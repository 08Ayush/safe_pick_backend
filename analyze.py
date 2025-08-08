import base64
import os
from dotenv import load_dotenv
from together import Together
import json
import re # Import regular expression module

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

# Check if the API key is found, otherwise exit with an error message
if not api_key:
    print("ERROR: TOGETHER_API_KEY not found in .env file!")
    print("Please ensure your .env file is in the same directory and contains 'TOGETHER_API_KEY=\"YOUR_API_KEY_HERE\"'.")
    exit()

# Initialize the Together AI client with the API key
client = Together(api_key=api_key)

def encode_image(image_path):
    """
    Encodes an image file to a base64 string.
    This is necessary to send the image data to the AI model.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_label(image_path, user_allergies=[]):
    """
    Analyzes a food product label image using an AI model to detect allergens.

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
You are an AI allergen detector.
Given an image of a food product label, extract the *ingredients* and identify any ingredients that may pose an allergy risk to this user.

User Allergy Profile: {allergy_info}

Return a valid JSON object with the following keys:
1. "extracted_ingredients" (as a list of strings)
2. "detected_allergens" (as a list of strings)
3. "risk_level": "Low" | "Moderate" | "High"
4. "recommendation" (as a string)

Ensure the output is ONLY the JSON object, without any additional text, markdown formatting (like ```json), or conversational elements.
"""

    # Make the API call to the Together AI model
    # Using stream=True, we accumulate the response in chunks
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        stream=True
    )

    response_text = ""
    for token in response:
        # Concatenate content from streamed tokens
        if hasattr(token, 'choices') and token.choices and hasattr(token.choices[0].delta, 'content'):
            response_text += token.choices[0].delta.content or ""

    # --- Robust JSON Extraction and Error Handling ---
    json_string_to_parse = ""

    # 1. Try to find JSON within markdown code blocks (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
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

