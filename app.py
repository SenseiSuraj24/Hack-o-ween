# Assuming the previous 'predict_heart_risk.py' code is imported or part of this file
import predict_heart_risk
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/predict_risk', methods=['POST'])
def handle_prediction_request():
    # Expects a JSON body with the patient data
    data = request.get_json()
    
    # Validate the data structure (optional, but highly recommended)
    # Check if all required keys are present in the 'data' dictionary

    try:
        # Call the prediction function
        result = predict_heart_risk(data)
        
        # Return the prediction result as a JSON response
        return jsonify(result), 200
    except Exception as e:
        # Handle any errors gracefully
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# To run the web server:
if __name__ == '__main__':
#     # You would normally not run in debug mode for production
     app.run(debug=True)