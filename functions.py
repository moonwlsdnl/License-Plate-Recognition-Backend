import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract

# 설정 변수
min_confidence = 0.5 
frame_size = 320
padding = 0.05

# EAST 디텍터
east_detector = 'yolo/frozen_east_text_detection.pb'

# YOLO 모델 로드
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Tesseract의 설치 경로 지정
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' # Docker 실행 시
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract' # 로컬 실행 시

# 자동차 검출 함수
def carROI(image):
    # 이미지 크기 정보 저장
    height, width, channels = image.shape
    
    # YOLO 입력을 위한 이미지 전처리
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []  # 객체 신뢰도 저장 리스트
    boxes = []        # 감지된 객체의 좌표 저장 리스트

    # 네트워크 출력 결과를 반복하여 객체 탐지
    for out in outs:                 
        for detection in out:       
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 신뢰도가 일정 수준 이상인 '차량'만 필터링 (YOLO에서 car class는 2번)
            if class_id == 2 and confidence > min_confidence:
                center_x = int(detection[0] * width)  # 박스 중심 x 좌표
                center_y = int(detection[1] * height) # 박스 중심 y 좌표
                w = int(detection[2] * width)         # 박스 너비
                h = int(detection[3] * height)        # 박스 높이

                # 실제 이미지 내 박스 좌표 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])    
                confidences.append(float(confidence))
    
    if confidences:
        # 비최대 억제(NMS) 적용하여 가장 신뢰도 높은 박스 선택
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
        
        # indexes가 비어 있는지 확인
        if len(indexes) > 0:
            for i in indexes.flatten():  # Flatten을 사용하여 1D로 변환
                x, y, w, h = boxes[i]
                return (boxes[i], image[y:y+h, x:x+w])  # 차량 이미지 잘라내기

    return None  # 감지 실패 시 None 반환
    
        
# 번호판 검출 함수
def textROI(image):
    # 원본 이미지를 복사하고, 이미지의 높이와 너비를 가져온다.
    orig = image.copy()
    (origH, origW) = image.shape[:2]
 
    # 차량 이미지를 잘라냈을 때 크기가 다르면 번호판이 왜곡될 수 있다. (번호판은 차량 중앙에 위치)
    # 왜곡 없이 정사각형 이미지(320x320)로 맞추기 위해서 다음 작업을 진행.
    rW = origW / float(frame_size)    # 원본 너비와 frame_size의 비율 계산
    rH = origH / float(frame_size)    # 원본 높이와 frame_size의 비율 계산
    newW = int(origW / rH)            # 높이에 맞춰 정사각형 형태로 새로운 너비 계산
    center = int(newW / 2)            # 이미지 중앙 계산
    start = center - int(frame_size / 2)  # 정사각형을 만들기 위한 시작점 계산

    # 이미지를 리사이즈하고 새로운 크기의 이미지를 가져온다.
    image = cv2.resize(image, (newW, frame_size))  
    # scale_image = image[0:frame_size, start:start+frame_size]  # 잘라내어 정사각형 이미지 생성
    # (H, W) = scale_image.shape[:2]

    # 이미지를 확인하기 위해 창에 띄운다.
    # cv2.imshow("orig", orig)
    # cv2.imshow("resize", image)
    # cv2.imshow("scale_image", scale_image)
    
    # EAST 텍스트 검출 모델을 위한 두 개의 출력 레이어 이름 정의
    layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
    
    # 미리 학습된 EAST 텍스트 검출기 로드
    net = cv2.dnn.readNet(east_detector)

    # 이미지에서 blob을 생성
    blob = cv2.dnn.blobFromImage(image, 1.0, (frame_size, frame_size),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]  # 점수의 행과 열 가져오기
    rects = []       # 텍스트 박스 좌표 저장 리스트
    confidences = [] # 텍스트 박스 신뢰도 저장 리스트

    # 행 개수만큼 반복하여 스코어 추출
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # 열 개수만큼 반복하여 텍스트 검출
        for x in range(0, numCols):
                # 신뢰도가 낮은 검출은 무시
                if scoresData[x] < min_confidence:
                        continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)  # 텍스트 검출 오프셋 좌표

                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))  # 검출된 텍스트 영역 추가
                confidences.append(scoresData[x])           # 해당 텍스트 영역의 신뢰도 추가
    
    # 비최대 억제 적용하여 신뢰도 높은 박스들만 선택
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # 검출된 텍스트 박스 순회
    for (startX, startY, endX, endY) in boxes:
            # 원본 크기에 맞추기 위해 박스 좌표 조정
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # 박스 주변에 패딩 추가
            dX = int((endX - startX) * padding)
            dY = int((endY - startY) * padding)

            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))

            # 실제 패딩된 영역 추출 및 반환
            return ([startX, startY, endX, endY], orig[startY:endY, startX:endX])
    
    return None  # 감지 실패 시 None 반환


# 이미지 전처리 함수
def processROI(image):
    # 이미지를 회색조로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러를 적용하여 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 이진화 (Thresholding) 적용
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

# 글씨 인식 함수
def textRead(image):
    # Tesseract v4를 사용하여 OCR 적용
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(image, config=config)
    
    # Tesseract로 OCR된 텍스트 출력
    print("OCR TEXT : {}\n".format(text))
    
    # 비ASCII 문자를 제거하고 텍스트 정리
    text = "".join([c if c.isalnum() else "" for c in text]).strip()
    print("알파벳 숫자 텍스트 : {}\n".format(text))
    
    return text