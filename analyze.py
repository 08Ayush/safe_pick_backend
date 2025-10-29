import base64
import os
from dotenv import load_dotenv
import requests
import json
import re # Import regular expression module

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
api_url = os.getenv("GEMINI_API_URL")

# Check if the API key is found, otherwise exit with an error message
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file!")
    print("Please ensure your .env file is in the same directory and contains 'GEMINI_API_KEY=\"YOUR_API_KEY_HERE\"'.")
    exit()

if not api_url:
    print("ERROR: GEMINI_API_URL not found in .env file!")
    exit()

def encode_image(image_path):
    """
    Encodes an image file to a base64 string.
    This is necessary to send the image data to the AI model.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_label(image_path, user_allergies=[]):
    """
    Analyzes a food product label image using Gemini AI to detect allergens.

    Args:
        image_path (str): The file path to the image of the food label.
        user_allergies (list): A list of strings representing the user's known allergies.

    Returns:
        str: A JSON string containing extracted ingredients, detected allergens,
             risk level, and a recommendation. Returns a structured error JSON
             if parsing the AI response fails.
    """
    base64_image = encode_image(image_path)
    # Format user allergies for the AI prompt
    allergy_info = ", ".join(user_allergies) if user_allergies else "none provided"

    # Define the prompt for the AI model, asking for a structured JSON response
    prompt = f"""
You are an expert food allergen detection AI. Analyze this food product label image carefully.

TASK:
1. *Read and extract ALL ingredients* listed on the label (look for "Ingredients:", "Contains:", or similar sections)
2. *Identify allergens* that match the user's allergy profile
3. *Assess risk level* based on detected allergens
4. *Provide a recommendation*

User's Allergies: {allergy_info}

IMPORTANT INSTRUCTIONS:
- Look carefully at the entire label for ingredient lists
- Extract every single ingredient you can read
- Pay special attention to small text and fine print
- If the image is unclear, state that in the recommendation
- Common allergens include: milk, eggs, fish, shellfish, tree nuts, peanuts, wheat, soybeans, sesame

OUTPUT FORMAT (JSON only, no markdown):
{{
  "extracted_ingredients": ["ingredient1", "ingredient2", "..."],
  "detected_allergens": ["allergen1", "allergen2"],
  "risk_level": "Low" or "Moderate" or "High",
  "recommendation": "Clear safety advice for the user"
}}

If you cannot read the label clearly, return:
{{
  "extracted_ingredients": [],
  "detected_allergens": [],
  "risk_level": "Unknown",
  "recommendation": "Unable to read the label clearly. Please ensure the image is well-lit, in focus, and the ingredients list is visible."
}}
"""

    # Prepare the request for Gemini API
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 2048
        }
    }

    # Make the API call to Gemini
    try:
        response = requests.post(
            f"{api_url}?key={api_key}",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Parse the Gemini response
        result = response.json()
        
        # Extract the text from Gemini's response structure
        if 'candidates' in result and len(result['candidates']) > 0:
            response_text = result['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception("No response from Gemini API")
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        error_response = {
            "extracted_ingredients": [],
            "detected_allergens": [],
            "risk_level": "Unknown",
            "recommendation": f"API Error: {str(e)}"
        }
        return json.dumps(error_response)

    # --- Robust JSON Extraction and Error Handling ---
    json_string_to_parse = ""

    # 1. Try to find JSON within markdown code blocks (json ...  or  ... )
    match = re.search(r"(?:json)?\s*(\{.*?\})\s*", response_text, re.DOTALL)
    if match:
        json_string_to_parse = match.group(1).strip()
    else:
        # 2. If no markdown block, try to find the first and last curly braces
        # This is a common pattern for LLMs that output JSON without wrappers
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_string_to_parse = response_text[first_brace : last_brace + 1].strip()
        else:
            # 3. If still no clear JSON, use the raw response and hope for the best
            # (This is the least reliable but necessary fallback)
            json_string_to_parse = response_text.strip()

    try:
        # Attempt to parse the (potentially extracted) string as JSON
        json.loads(json_string_to_parse)
        return json_string_to_parse # Return the clean, valid JSON string
    except json.JSONDecodeError as e:
        # If parsing fails, it means the AI did not return valid JSON.
        # Construct a structured error response that Flutter can always parse.
        print(f"Warning: AI response was not valid JSON. Error: {e}. Raw output: {response_text.strip()}")
        error_response = {
            "extracted_ingredients": [],
            "detected_allergens": [],
            "risk_level": "Unknown",
            "recommendation": f"Could not fully parse AI response. Raw AI output (for debug): {response_text.strip()}"
        }
        return json.dumps(error_response)
    # --- END Robust JSON Extraction and Error Handling ---

# Example usage for local testing (not used when Flask app is running)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <path_to_image.jpg>")
        sys.exit(1)
    
    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        sys.exit(1)

    test_allergies = ["milk", "peanuts"]
    print(f"Analyzing image: {image_path} for allergies: {test_allergies}")
    result = analyze_label(image_path, test_allergies)
    print("\n--- Analysis Result (Raw JSON String) ---")
    print(result)
    try:
        parsed_result = json.loads(result)
        print("\n--- Analysis Result (Pretty Printed JSON) ---")
        print(json.dumps(parsed_result, indent=2))
    except json.JSONDecodeError:
        print("Result is not valid JSON.")
