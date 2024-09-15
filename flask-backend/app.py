from flask import Flask, request
from flask_cors import CORS
import joblib
import sklearn

app = Flask(__name__)
CORS(app)
model = joblib.load("flask-backend/WESAD_binary_xgboost.pkl")

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Confirm data exists
    if not "heart_rate" in data:
        return {"error": "Missing heart_rate parameter"}, 400
    if not "x_acceleration" in data:
        return {"error": "Missing x_acceleration parameter"}, 400
    if not "y_acceleration" in data:
        return {"error": "Missing y_acceleration parameter"}, 400
    if not "z_acceleration" in data:
        return {"error": "Missing z_acceleration parameter"}, 400
    if "temperature" not in data:
        temperature = 5

    # Confirm data is strings
    if not isinstance(data["heart_rate"], (str, float, int)):
        print("1")
        return {"error": "Invalid heart_rate parameter: data type"}, 400
    if not isinstance(data["x_acceleration"], (str, float, int)):
        print("2")
        return {"error": "Invalid heart_rate parameter: data type"}, 400
    if not isinstance(data["y_acceleration"], (str, float, int)):
        print("3")
        return {"error": "Invalid heart_rate parameter: data type"}, 400
    if not isinstance(data["z_acceleration"], (str, float, int)):
        print("4")
        return {"error": "Invalid heart_rate parameter: data type"}, 400
    
    # Confirm data can be cast
    if not is_float(data["heart_rate"]):
        print(1)
        return {"error": "Invalid heart_rate parameter"}, 400
    if not is_float(data["x_acceleration"]):
        print(2)
        return {"error": "Invalid x_acceleration parameter"}, 400
    if not is_float(data["y_acceleration"]):
        print(3)
        return {"error": "Invalid y_acceleration parameter"}, 400
    if not is_float(data["z_acceleration"]):
        print(4)
        return {"error": "Invalid z_acceleration parameter"}, 400
    if not isinstance(data, str) or not data["temperature"].isnumeric():
        temperature = 5

    # Convert temperature to cutoff 
    if (int(temperature) < 1 or int(temperature) > 10):
        temperature = 5
    proba_cutoff = 0.5 + (0.05 * (5 - int(temperature)))
    
    #TODO: get correct data shape

    heart_rate = float(data["heart_rate"])
    x_acceleration = float(data["x_acceleration"])
    y_acceleration = float(data["y_acceleration"])
    z_acceleration = float(data["z_acceleration"])

    prediction = model.predict_proba([[heart_rate, x_acceleration, y_acceleration, z_acceleration]])[0][1]
    print(prediction.item())

    return {"prediction": "normal" if prediction.item() < proba_cutoff else "anomaly"}
    

if __name__ == '__main__':
    app.run(debug=True, port=8080)