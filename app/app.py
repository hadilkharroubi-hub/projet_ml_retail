from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ajouter src/ au path pour importer predict.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_client

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data   = request.get_json()
        result = predict_client(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'API opérationnelle'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)