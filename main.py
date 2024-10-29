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

    # 차량 ROI 감지
    ([x, y, w, h], car_image) = carROI(image)

    # 텍스트 ROI 감지
    ([startX, startY, endX, endY], text_image) = textROI(car_image)

    # 이미지 처리
    process_image = processROI(text_image)

    # 텍스트 읽기
    text = textRead(process_image)

    # 결과 반환
    return jsonify({"detections": text})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
