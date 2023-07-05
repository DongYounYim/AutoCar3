import cv2
import numpy as np
from pop import Pilot

# 실시간 캠 객체 생성
cap = Pilot.Camera(320, 240)

# Darknet 설정 파일과 가중치 파일 경로
config_file = "./yolov4-tiny-settings/yolov4-tiny-custom.cfg"
weight_file = "./weight/yolov4-tiny-custom_best.weights"
meta_file = "./yolov4-tiny-settings/obj.names"

# 객체 탐지 모델 초기화
YOLO_net = cv2.dnn.readNet(weight_file, config_file)

classes = []
with open(meta_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

# 캠에서 프레임 읽기
while True:
    frame = cap.value

    # 프레임 크기 가져오기
    h, w, c = frame.shape
    
    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # class_id : 클래스 넘버링 / confidence는 확률?
            # print(scores, class_id, confidence)

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

            # 경계상자와 클래스 정보 이미지에 입력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
            (255, 255, 255), 1)

    cv2.imshow("YOLOv4-tiny", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.close()
cv2.destroyAllWindows()