from flask import Flask, request, jsonify
import cv2
import numpy as np
from functions import carROI, textROI, processROI, textRead

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect_objects():
    # 이미지 파일 받기
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # 이미지 읽기
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image. Please check the input file format.")
    

    # 차량 ROI 감지
    car_roi = carROI(image)
    if car_roi is None:
        return jsonify({"error": "Vehicle not detected"}), 400
    
    ([x, y, w, h], car_image) = car_roi

    # 텍스트 ROI 감지
    text_roi = textROI(car_image)
    if text_roi is None:
        return jsonify({"error": "License plate not detected"}), 400
    
    ([startX, startY, endX, endY], text_image) = text_roi

    # 이미지 처리
    process_image = processROI(text_image)

    # 텍스트 읽기
    text = textRead(process_image)

    # 결과 반환
    return jsonify({"detections": text})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
