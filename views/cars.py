"""
Car APIs
"""
import json
import time

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

cars = Blueprint('cars', __name__)


@cars.route('/mark', methods=['POST'])
def mark():
    from app import app
    m = app.config['MODEL_Vehicle']

    file = request.files['file']
    # Save to mem
    file_str = file.read()
    # => NumPy
    np_img = np.frombuffer(file_str, np.uint8)
    # use cv2.imread() to read it
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # jobs
    results = m.mark_vehicles(img, convert_np_data=True)

    return jsonify(results)
