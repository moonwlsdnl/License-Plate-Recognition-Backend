# License-Plate-Recognition-Project(차량 번호판 인식 시스템)

이 프로젝트는 YOLO와 EAST 텍스트 검출기를 사용하여 차량의 번호판을 인식하는 시스템입니다. OpenCV, NumPy 및 Tesseract OCR을 활용하여 이미지에서 차량과 번호판을 감지하고, 텍스트를 추출합니다.

이 프로젝트는 차량 번호판 인식 시스템으로, 다음 단계로 진행됩니다:

## 동작 과정
1. 사용자가 이미지 파일을 업로드합니다.
2. YOLO 모델을 사용하여 이미지에서 차량을 감지합니다.
3. 감지된 차량의 영역에서 번호판 영역을 찾아냅니다.
4. Tesseract OCR을 사용하여 번호판 이미지에서 텍스트를 인식합니다.
5. 인식된 텍스트를 JSON 형태로 반환합니다.


## 함수 설명
1. carROI(image)
    - 입력된 이미지에서 차량을 감지하고, 차량의 영역을 잘라냅니다.
    - 입력: image (numpy array) - 차량이 포함된 이미지
    - 출력: 차량의 좌표와 잘라낸 차량 이미지
2. textROI(image)
    - 차량 이미지에서 번호판을 감지하고, 번호판의 영역을 잘라냅니다.
    - 입력: image (numpy array) - 차량 이미지
    - 출력: 번호판의 좌표와 잘라낸 번호판 이미지
3. processROI(image)
    - 입력된 이미지에 대해 전처리(회색조 변환, 가우시안 블러, 이진화)를 수행합니다.
    - 입력: image (numpy array) - 번호판 이미지
    - 출력: 전처리된 이진화 이미지
4. textRead(image)
    - Tesseract OCR을 사용하여 입력된 이미지에서 텍스트를 인식합니다.
    - 입력: image (numpy array) - 전처리된 이미지
    - 출력: 인식된 텍스트

## Flask API

### `/detect` [POST]

- **이미지 파일을 받아 차량 번호판을 인식합니다.**
- **요청 바디**: 이미지 파일 (key: `file`)
- **응답**: 인식된 텍스트가 포함된 JSON 객체

### 예시 요청

```bash
curl -X POST -F 'file=@path/to/image.jpg' http://localhost:5001/detect
```

### 예시 응답
```json
{
  "detections": "ABC1234"
}
```

## 출처
- 자동차 번호판 인식 프로젝트: https://mldlcvmjw.tistory.com/330

- YOLO (You Only Look Once): https://traumees.tistory.com/172

- frozen_east_text_detection.pb/frozen_east_text_detection.pb at master · oyyd/frozen_east_text_detection.pb: https://github.com/oyyd/frozen_east_text_detection.pb/blob/master/frozen_east_text_detection.pb

- OCR 입문기 - tessearct / pytesseract / 사진 한국어로 읽기: https://scribbled-diary.tistory.com/44

- tesserocr: https://pypi.org/project/tesserocr/

- [오류 노트] libGL.so.1: cannot open shared object file: No such file or directory: https://velog.io/@zxxzx1515/%EC%98%A4%EB%A5%98-%EB%85%B8%ED%8A%B8-libGL.so.1-cannot-open-shared-object-file-No-such-file-or-directory

- [OpenCV] libgthread-2.0.so.0: cannot open shared object file 해결: https://shuka.tistory.com/31






