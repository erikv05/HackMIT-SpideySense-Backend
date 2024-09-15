from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True, port=8000)