from flask import Flask, request, jsonify
from analyze import analyze_label # Import the analyze_label function
import os
import json # Import json for potential error handling or direct JSON return

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    API endpoint to receive an image and user allergies,
    then analyze the image for allergens using the AI model.
    """
    # Check if an image file was included in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    # Get allergies from form data, default to empty string if not provided
    allergies = request.form.get('allergies', '')

    # Define a temporary path to save the uploaded image
    temp_path = "temp_image.jpg"
    try:
        # Save the uploaded image to the temporary path
        image.save(temp_path)

        # Convert the comma-separated allergies string into a list
        allergy_list = [a.strip() for a in allergies.split(',')] if allergies else []

        # Call the analyze_label function from analyze.py
        # This function is now designed to return a clean JSON string
        response_text = analyze_label(temp_path, allergy_list)

        # Return the response_text directly. It's expected to be a valid JSON string.
        # Set Content-Type header to application/json and status code to 200 OK.
        return response_text, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        # Catch any exceptions that occur during processing
        print(f"An error occurred: {e}") # Print error to server console for debugging
        # Return a structured JSON error response to the client
        return jsonify({"error": str(e)}), 500
    finally:
        # Ensure the temporary image file is always removed, regardless of success or failure
        if os.path.exists(temp_path):
            os.remove(temp_path)

# This block ensures the Flask development server runs when the script is executed directly
if __name__ == "__main__":
    # IMPORTANT: For production, do not use debug=True.
    # Also, ensure the host is accessible from your mobile device/emulator.
    # '0.0.0.0' makes it accessible on your local network.
    # Replace '192.168.0.105' with your computer's actual local IP address
    # in the Flutter app's scan_screen.dart file.
    app.run(host="0.0.0.0", port=5500, debug=True)

