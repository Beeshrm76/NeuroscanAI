import os
import json
import glob
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2
import pydicom
from PIL import Image as PILImage
from preprocess import get_data_generators
from ensemble_utils import ensemble_predict_proba, ENSEMBLE_MODEL_PREPROCESS, ENSEMBLE_MODEL_IMG_SIZE
from recommendation_utils import get_doctor_recommendations

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Directories for evaluation metrics & training curves
EVALUATION_RESULTS_DIR = 'evaluation_results'
EVALUATION_RESULTS_NATIVE_DIR = 'evaluation_results_native'
EVALUATION_RESULTS_STANDARD_DIR = 'evaluation_results_standard'

TRAINING_HISTORY_DIR = 'training_history'
TRAINING_HISTORY_NATIVE_DIR = 'training_history_native'
TRAINING_HISTORY_STANDARD_DIR = 'training_history_standard'

CONFUSION_MATRIX_FILENAME = "{model}_confusion_matrix.png"

# Global session variables
latest_results = {}
latest_top3 = []
latest_image_file = None
latest_cam_file = None
latest_patient_id = "N/A"
latest_patient_age_gender = "N/A"
latest_scan_modality = "Standard MRI"


def resolve_model_path(model_filename):
    """Resolve model weights path with graceful fallbacks:
    models_native/ -> models/ -> models_standard/
    """
    candidates = [
        os.path.join('models_native', model_filename),
        os.path.join('models', model_filename),
        os.path.join('models_standard', model_filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join('models', model_filename)


# Load all 6 trained architectures. Every prediction combines ALL 6 via a
# fixed, accuracy-weighted average (see ensemble_utils.py).
models = {
    'VGG16': load_model(resolve_model_path('vgg16_model.h5')),
    'ResNet50': load_model(resolve_model_path('resnet50_model.h5')),
    'MobileNetV2': load_model(resolve_model_path('mobilenet_model.h5')),
    'DenseNet121': load_model(resolve_model_path('densenet_model.h5')),
    'EfficientNetB0': load_model(resolve_model_path('efficientnet_model.h5')),
    'InceptionV3': load_model(resolve_model_path('inception_model.h5')),
}

# Dynamically load all class labels
train_gen, _, _ = get_data_generators()
CLASSES = list(train_gen.class_indices.keys())


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def build_model_inputs(filepath):
    """
    Build a SEPARATE, correctly-preprocessed input array per model, since
    each of the 6 architectures (VGG16, ResNet50, MobileNetV2,
    DenseNet121, EfficientNetB0, InceptionV3) requires its own
    preprocess_input function and native input resolution.
    """
    x_by_model = {}
    for name in models.keys():
        img_size = ENSEMBLE_MODEL_IMG_SIZE[name]
        preprocess_fn = ENSEMBLE_MODEL_PREPROCESS[name]
        img = image.load_img(filepath, target_size=img_size)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        x_by_model[name] = preprocess_fn(arr)
    return x_by_model


def analyze_image(filepath, filename):
    """
    Execute inference across all 6 architectures, compute accuracy-weighted
    ensemble predictions, identify top 3 confident models, and generate Grad-CAM.
    Updates global state and returns (results, top3, image_file, cam_file).
    """
    global latest_results, latest_top3, latest_image_file, latest_cam_file

    x_by_model = build_model_inputs(filepath)
    CONFIDENCE_THRESHOLD = 50.0

    results = {}
    for name, model in models.items():
        preds = model.predict(x_by_model[name], verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0])) * 100

        if confidence < CONFIDENCE_THRESHOLD:
            results[name] = {
                'prediction': 'UNCERTAIN / LOW CONFIDENCE',
                'confidence': round(confidence, 2)
            }
        else:
            results[name] = {
                'prediction': CLASSES[class_idx],
                'confidence': round(confidence, 2)
            }

    ensemble_proba, weights_used, top_3_confidences = ensemble_predict_proba(models, x_by_model)
    ensemble_class_idx = np.argmax(ensemble_proba)
    ensemble_confidence = float(np.max(ensemble_proba)) * 100

    ensemble_label = "Ensemble (All 6, accuracy-weighted)"
    if ensemble_confidence < CONFIDENCE_THRESHOLD:
        results[ensemble_label] = {
            'prediction': 'UNCERTAIN / LOW CONFIDENCE',
            'confidence': round(ensemble_confidence, 2)
        }
    else:
        results[ensemble_label] = {
            'prediction': CLASSES[ensemble_class_idx],
            'confidence': round(ensemble_confidence, 2)
        }

    top_3_display = [
        {'model': name, 'confidence': round(conf * 100, 2)}
        for name, conf in top_3_confidences
    ]

    latest_results = results
    latest_top3 = top_3_display
    latest_image_file = filename

    LAST_CONV_LAYER_BY_MODEL = {
        'VGG16': 'block5_conv3',
        'ResNet50': 'conv5_block3_out',
        'MobileNetV2': 'out_relu',
        'DenseNet121': 'conv5_block16_concat',
        'EfficientNetB0': 'top_conv',
        'InceptionV3': 'mixed10',
    }

    cam_filename = None
    try:
        cam_model_name = top_3_confidences[0][0]
        cam_model = models[cam_model_name]
        x_cam = x_by_model[cam_model_name]
        last_conv_layer_name = LAST_CONV_LAYER_BY_MODEL[cam_model_name]

        heatmap = make_gradcam_heatmap(x_cam, cam_model, last_conv_layer_name)

        cam_filename = "cam_" + filename
        cam_path = os.path.join(app.config['UPLOAD_FOLDER'], cam_filename)

        original_img = cv2.imread(filepath)
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        cv2.imwrite(cam_path, superimposed)
    except Exception as e:
        print("Grad-CAM generation error:", e)

    latest_cam_file = cam_filename
    return results, top_3_display, filename, cam_filename


@app.route('/')
def home():
    if latest_results and latest_image_file:
        return render_template(
            'index.html',
            results=latest_results,
            top3=latest_top3,
            image_file=latest_image_file,
            cam_file=latest_cam_file
        )
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global latest_patient_id, latest_patient_age_gender, latest_scan_modality

    latest_patient_id = request.form.get('patient_id', 'N/A')
    latest_patient_age_gender = request.form.get('patient_age_gender', 'N/A')
    latest_scan_modality = request.form.get('scan_modality', 'Standard MRI')

    if 'file' not in request.files or request.files['file'].filename == '':
        if latest_image_file:
            return redirect(url_for('reanalyze'))
        return redirect(request.url)

    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # DICOM Handling (.dcm files)
        if filename.lower().endswith('.dcm'):
            try:
                dicom_data = pydicom.dcmread(filepath)

                if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
                    latest_patient_id = str(dicom_data.PatientID)

                p_age = getattr(dicom_data, 'PatientAge', '')
                p_sex = getattr(dicom_data, 'PatientSex', '')
                if p_age or p_sex:
                    latest_patient_age_gender = f"{p_age} / {p_sex}".strip(' /')

                if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
                    latest_scan_modality = f"DICOM {dicom_data.Modality} Scan"

                pixel_array = dicom_data.pixel_array
                if pixel_array.max() != pixel_array.min():
                    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                else:
                    pixel_array = pixel_array.astype(np.uint8)

                img_pil = PILImage.fromarray(pixel_array).convert('RGB')
                new_filename = filename.rsplit('.', 1)[0] + '.png'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                img_pil.save(filepath)
                filename = new_filename
            except Exception as e:
                print("DICOM parsing error:", e)

        results, top3, image_file, cam_file = analyze_image(filepath, filename)
        return render_template('index.html', results=results, top3=top3, image_file=image_file, cam_file=cam_file)


@app.route('/reanalyze', methods=['GET', 'POST'])
def reanalyze():
    """Re-analyze current MRI scan across all models and regenerate Grad-CAM."""
    global latest_image_file
    if not latest_image_file:
        return redirect(url_for('home'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], latest_image_file)
    if not os.path.exists(filepath):
        return redirect(url_for('home'))
    results, top3, image_file, cam_file = analyze_image(filepath, latest_image_file)
    return render_template('index.html', results=results, top3=top3, image_file=image_file, cam_file=cam_file)


@app.route('/reset')
def reset():
    """Clear active scan results to cleanly start the next scan upload."""
    global latest_results, latest_top3, latest_image_file, latest_cam_file
    global latest_patient_id, latest_patient_age_gender, latest_scan_modality
    latest_results = {}
    latest_top3 = []
    latest_image_file = None
    latest_cam_file = None
    latest_patient_id = "N/A"
    latest_patient_age_gender = "N/A"
    latest_scan_modality = "Standard MRI"
    return redirect(url_for('home'))


@app.route('/api/recommend_doctors', methods=['POST'])
def recommend_doctors_api():
    data = request.get_json()
    city = data.get('city')
    tumor_class = data.get('tumor_class')
    
    if not city or not tumor_class:
        return {"error": "Missing city or tumor_class"}, 400
        
    print(f"AJAX: Fetching doctors for {tumor_class} in {city}...")
    recommendations, err_msg = get_doctor_recommendations(tumor_class, city)
    
    if recommendations:
        return {"success": True, "data": recommendations}, 200
    else:
        return {"success": False, "error": err_msg or "Failed to fetch recommendations from LLM"}, 500


@app.route('/performance')
def performance():
    """
    Model-performance dashboard: accuracy/precision/recall/F1 comparison
    across all 6 models + ensemble, confusion matrices, and per-model
    train/val loss+accuracy curves.
    """
    summary_records = []
    is_native = False

    # Check for active or native results first, fall back to standard if needed
    summary_candidates = [
        (os.path.join(EVALUATION_RESULTS_NATIVE_DIR, 'summary_metrics.csv'), True),
        (os.path.join(EVALUATION_RESULTS_DIR, 'summary_metrics.csv'), False),
        (os.path.join(EVALUATION_RESULTS_STANDARD_DIR, 'summary_metrics.csv'), False),
    ]

    for cand_path, is_nat in summary_candidates:
        if os.path.exists(cand_path):
            try:
                df = pd.read_csv(cand_path)
                if not df.empty:
                    summary_records = df.to_dict(orient='records')
                    is_native = is_nat
                    break
            except Exception as e:
                print(f"Error reading {cand_path}: {e}")

    model_names = [row.get('run_name', row.get('model', row.get('name', ''))) for row in summary_records]

    confusion_matrices = {}
    search_dirs = [EVALUATION_RESULTS_DIR, EVALUATION_RESULTS_NATIVE_DIR, EVALUATION_RESULTS_STANDARD_DIR]

    for name in model_names:
        if not name:
            continue
        found = False
        for sdir in search_dirs:
            candidate = os.path.join(sdir, CONFUSION_MATRIX_FILENAME.format(model=name))
            if os.path.exists(candidate):
                confusion_matrices[name] = os.path.basename(candidate)
                found = True
                break

    training_curves = {}
    history_dirs = [TRAINING_HISTORY_DIR, TRAINING_HISTORY_NATIVE_DIR, TRAINING_HISTORY_STANDARD_DIR]

    for hdir in history_dirs:
        if not os.path.exists(hdir):
            continue
        for path in glob.glob(os.path.join(hdir, '*_history.json')):
            model_key = os.path.basename(path).replace('_history.json', '')
            if model_key not in training_curves:
                try:
                    with open(path) as f:
                        training_curves[model_key] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

    return render_template(
        'dashboard.html',
        summary_records=summary_records,
        confusion_matrices=confusion_matrices,
        training_curves=training_curves,
        is_native=is_native,
    )


@app.route('/evaluation_results/<path:filename>')
def evaluation_result_file(filename):
    """Serves confusion matrix images with fallback across evaluation directories."""
    for folder in [EVALUATION_RESULTS_DIR, EVALUATION_RESULTS_NATIVE_DIR, EVALUATION_RESULTS_STANDARD_DIR]:
        target = os.path.join(folder, filename)
        if os.path.exists(target):
            return send_from_directory(folder, filename)
    return "File not found", 404


@app.route('/download_report')
def download_report():
    if not latest_image_file or not latest_results:
        print("Export failed: No latest_image_file or latest_results found.")
        return redirect(url_for('home'))

    try:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'diagnostic_report.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0f172a'), spaceAfter=4)
        subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#64748b'), spaceAfter=15)
        heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=4)

        story.append(Paragraph("NeuroScan AI &bull; Clinical Diagnostic Report", title_style))
        story.append(Paragraph("Automated Multi-Model Ensemble Brain Tumor Classification System", subtitle_style))

        data_meta = [
            [Paragraph(f"<b>Patient ID:</b> {latest_patient_id}", styles['Normal']),
             Paragraph(f"<b>Modality:</b> {latest_scan_modality}", styles['Normal'])],
            [Paragraph(f"<b>Age / Gender:</b> {latest_patient_age_gender}", styles['Normal']),
             Paragraph("<b>Status:</b> Verified Analysis", styles['Normal'])]
        ]
        t_meta = Table(data_meta, colWidths=[270, 270])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
            ('PADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Ensemble Model Predictions", heading_style))

        cell_head = ParagraphStyle('TH', parent=styles['Normal'], fontSize=9, leading=12, fontName='Helvetica-Bold', textColor=colors.white)
        cell_head_center = ParagraphStyle('THC', parent=cell_head, alignment=1)
        cell_body = ParagraphStyle('TD', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#1e293b'))
        cell_body_bold = ParagraphStyle('TDBold', parent=styles['Normal'], fontSize=9, leading=12, fontName='Helvetica-Bold', textColor=colors.HexColor('#0f172a'))
        cell_body_center = ParagraphStyle('TDCenter', parent=cell_body, alignment=1)
        cell_body_center_bold = ParagraphStyle('TDCenterBold', parent=cell_body_bold, alignment=1)

        table_data = [[
            Paragraph("Model Architecture", cell_head),
            Paragraph("Predicted Classification", cell_head),
            Paragraph("Confidence Score", cell_head_center)
        ]]

        for model_name, res in latest_results.items():
            is_ens = 'Ensemble' in model_name
            m_style = cell_body_bold if is_ens else cell_body
            p_style = cell_body_bold if is_ens else cell_body
            c_style = cell_body_center_bold if is_ens else cell_body_center

            table_data.append([
                Paragraph(model_name, m_style),
                Paragraph(str(res['prediction']), p_style),
                Paragraph(f"{res['confidence']}%", c_style)
            ])

        t_results = Table(table_data, colWidths=[190, 230, 120])
        t_results.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ]))
        story.append(t_results)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Visual Analysis & Grad-CAM Heatmap", heading_style))
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_image_file)
        cam_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_cam_file) if latest_cam_file else None

        img_cells = []
        caption_cells = []
        col_widths = []
        caption_style = ParagraphStyle('ImgCaption', parent=styles['Normal'], fontSize=8, leading=10, alignment=1, textColor=colors.HexColor('#475569'))

        if os.path.exists(img_path):
            img_cells.append(RLImage(img_path, width=130, height=130))
            caption_cells.append(Paragraph("<b>Input MRI Scan</b>", caption_style))
            col_widths.append(270)
        if cam_path and os.path.exists(cam_path):
            img_cells.append(RLImage(cam_path, width=130, height=130))
            caption_cells.append(Paragraph("<b>AI Grad-CAM Heatmap</b>", caption_style))
            col_widths.append(270)

        if img_cells:
            t_imgs = Table([img_cells, caption_cells], colWidths=col_widths)
            t_imgs.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,1), (-1,1), 4),
                ('BOTTOMPADDING', (0,1), (-1,1), 6),
            ]))
            story.append(t_imgs)

        story.append(Spacer(1, 15))

        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#94a3b8'))
        story.append(Paragraph("<b>Disclaimer:</b> This report is generated automatically by a computer vision research project prototype (NeuroScan AI). It serves as a preliminary diagnostic aid and must be reviewed by certified medical professionals prior to clinical decisions.", disclaimer_style))

        doc.build(story)
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return f"An error occurred while generating the report: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)