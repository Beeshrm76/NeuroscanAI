import os
import json
import requests
from typing import List, Dict, Optional

def get_doctor_recommendations(tumor_class: str, city: str, api_key: str = None) -> Optional[List[Dict]]:
    """
    Uses the Gemini REST API to search for top specialist recommendations based on the tumor class and city.
    
    Args:
        tumor_class: The predicted class of the tumor (e.g., 'Glioma', 'Meningioma').
        city: The city where the user is looking for doctors (e.g., 'Delhi', 'Mumbai').
        api_key: The Google Gemini API key. If None, it will look for 'GEMINI_API_KEY' in env variables.
        
    Returns:
        A list of dictionaries containing doctor recommendations, or None if there was an error.
    """
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key:
        error_msg = "Gemini API key not provided. Set GEMINI_API_KEY environment variable."
        print(f"Error: {error_msg}")
        return None, error_msg

    api_key = api_key.strip()

    prompt = f"""
    You are a helpful healthcare assistant. A patient has been diagnosed with a {tumor_class} brain tumor 
    and is looking for the best specialist doctors in {city}, India.
    
    Search for and recommend 3 highly-rated specialists (like Neuro-oncologists or Neurosurgeons) 
    who specialize in treating {tumor_class} in {city}.
    
    Return the result STRICTLY as a JSON array of objects. 
    Do not include any markdown formatting, backticks, or extra text. Just the raw JSON array.
    
    Each object must have the following keys:
    - "name": (string) Name of the doctor
    - "specialties": (array of strings) Their specific medical specialties or titles (e.g. ["Senior Consultant", "Neurosurgeon"])
    - "hospital": (string) The primary hospital or clinic they are associated with
    - "address": (string) The area or full address of the hospital in {city}
    - "rating": (string) Estimated numeric rating (e.g. "4.8/5")
    - "reviews": (string) Estimated number of reviews (e.g. "(120+ reviews)")
    - "why_recommended": (string) VERY SHORT, maximum 10-15 words explaining why they are recommended.
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text from the Gemini response structure
        result_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Clean up the response in case Gemini includes markdown code blocks (e.g., ```json ... ```)
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        result_text = result_text.strip()
        
        # Parse the JSON
        recommendations = json.loads(result_text)
        return recommendations, None
        
    except Exception as e:
        error_msg = f"Error calling Gemini REST API: {e}"
        print(error_msg)
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Raw response was: {response.text}")
        return None, error_msg

# --- Example Usage ---
if __name__ == "__main__":
    # To test this, you need to set your API key
    # os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
    
    test_tumor = "Glioma"
    test_city = "Mumbai"
    
    print(f"Fetching recommendations for {test_tumor} in {test_city}...")
    results = get_doctor_recommendations(test_tumor, test_city)
    
    if results:
        print(json.dumps(results, indent=2))
    else:
        print("Failed to get recommendations. Make sure you set GEMINI_API_KEY.")
