from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('churn_model.pkl')  # Ensure model exists

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return jsonify({"prediction": int(model.predict([[data['tenure']]])[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

!nohup python app.py &  # Runs in background
!curl http://localhost:5000/predict -X POST -H "Content-Type: application/json" -d '{"tenure": 12}'

import requests

try:
    response = requests.post(
        'http://localhost:5000/predict',
        json={'tenure': 24},
        timeout=5  # Prevents hanging
    )
    print(response.json())
except ConnectionRefusedError:
    print("Server not running! Start Flask first.")
except Exception as e:
    print(f"Error: {e}")
