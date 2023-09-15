import flask
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle  

app = Flask(__name__)

def load_model():
    try:
        with open('./pickle Hackathon_model.hin/saved_model.pb', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        return str(e)

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form('imageInput')

        prediction = model.predict(np.array(input_data).reshape(1, -1))

        result = f'The prediction is: {prediction[0]}'
        return render_template('index.html', result=result)
    except Exception as e:
        return str(e)

 

if __name__ == '__main__':
    app.run(debug=True)