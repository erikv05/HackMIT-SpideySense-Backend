from flask import Flask, request
from flask_cors import CORS
import joblib
import sklearn

app = Flask(__name__)
CORS(app)
model = joblib.load("flask-backend/WESAD_binary_xgboost.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    #TODO: add better error handling
    data = request.json
    if not "heart_rate" in data:
        return {"error": "Missing heart_rate parameter"}, 400
    if not "x_acceleration" in data:
        return {"error": "Missing x_acceleration parameter"}, 400
    if not "y_acceleration" in data:
        return {"error": "Missing y_acceleration parameter"}, 400
    if not "z_acceleration" in data:
        return {"error": "Missing z_acceleration parameter"}, 400
    if not "temperature" in data:
        temperature = 0.5
    else:
        temperature = int(data["temperature"])
    
    #TODO: get correct data shape
    #TODO: convert temperature from 1 to 10 scale

    heart_rate = data["heart_rate"]
    x_acceleration = data["x_acceleration"]
    y_acceleration = data["y_acceleration"]
    z_acceleration = data["z_acceleration"]

    print(heart_rate, x_acceleration, y_acceleration, z_acceleration)

    prediction = model.predict_proba([[0,0,0,0]])[0][1]
    print(prediction)

    return {"prediction": "normal" if prediction.item() >= temperature else "anomaly"}
    

if __name__ == '__main__':
    app.run(debug=True, port=8080)