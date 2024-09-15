from flask import Flask, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load("flask-backend/WESAD_binary_xgboost.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not "heart_rate" in data:
        return {"error": "Missing heart_rate parameter"}, 400
    if not "x_acceleration" in data:
        return {"error": "Missing x_acceleration parameter"}, 400
    if not "y_acceleration" in data:
        return {"error": "Missing y_acceleration parameter"}, 400
    if not "z_acceleration" in data:
        return {"error": "Missing z_acceleration parameter"}, 400

    return {"prediction": "normal" if model.predict([[data["heart_rate"], data["x_acceleration"], data["y_acceleration"], data["z_acceleration"]]])[0] == 0 else "anomaly"}
    

if __name__ == '__main__':
    app.run(debug=True, port=8080)