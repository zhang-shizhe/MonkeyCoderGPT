from flask import render_template, request, jsonify
from app import app
import sys
import os
# sys.path.insert(1, '../src/monkeycodergpt')   # Include model directory in the path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src', 'monkeycodergpt')
sys.path.insert(1, src_dir)
from generate import generate  # Import generation function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_code():
    # Call your model's generation function
    generated_code = generate()
    return jsonify(generated_code)
