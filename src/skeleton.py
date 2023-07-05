import cv2, math, time, os, sys, signal
from pop import Camera
from pop import Pilot
from pop import LiDAR
import numpy as np
from darknet import darknet

#---------------------------------------------------
kernel = np.ones((4,4),np.uint8)    #의미없으면 지울 예정

# 민감도 관련 변수
sensitivity = 90   
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,30+sensitivity])

# 차량 제어 변수
curr_steering_angle = 0.0   # 차량 바퀴 각도
current_speed = None    # 차량 스피드

drv = Pilot.AutoCar()   # 차량 객체
cam = Pilot.Camera(320, 240)    # 차량 내장 카메라 객체
drv.steering = 0    #바퀴 직선으로 초기화
#---------------------------------------------------

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

# 탐지된 객체 값
detect = None

# 신호등 횟수
traffic_count = 0

# ===============================
# Ctrl + C를 감지하여 처리하는 코드
def signal_handler():
    global drv
    cv2.destroyAllWindows()
    drv.stop()
    time.sleep(2)
    os.system("killall -9 python")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
# ===============================

# 엣지 탐색 (차선 찾기) - detect_line 내부 1
def detect_edges(frame):
    frame = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)    
    edges = cv2.Canny(mask, 200, 400)

    return edges

# 카메라 내에서 관찰할 영역 지정 - detect_line 내부 2
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height * 1 / 4),
        (width, height * 1 / 4),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image  = cv2.bitwise_and(edges, mask)
    return masked_image

# 엣지 (차선) 그리기 - detect_line 내부 3
def detect_line_segments(cropped_edges):
    rho = 1
    angle = np.pi / 180
    min_threshold = 50 
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=10, maxLineGap=4)

    return line_segments

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2)

    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

# 기울기? (라인의 기울기 혹은 경사) - detect_line 내부 4
def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        return lane_lines
        
    __height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2: #skipping vertical line segment
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

# 차선 라인 그리기(이미지) - detect_line 내부 5
def draw_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def detect_lane(frame):
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = draw_lines(frame, lane_lines)

    return lane_lines, lane_lines_image

# 라인에 따라 차의 이동방향(바퀴방향) 계산하기
def calc_steering_angle(frame, lane_lines, camera_mid_offset_percent = 0.0):
    if len(lane_lines) == 0:
        return 90

    height, width = frame.shape[:2]
    if len(lane_lines) == 1: #only detected one lane line
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:  
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    y_offset = int(height / 3)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi) 
    steering_angle = angle_to_mid_deg + 90

    return steering_angle

# 현재 바퀴 각도와 계산한 바퀴각도를 종합하여 계산
def stabilize_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, two_lines=6, one_lines=2):
    if num_of_lane_lines == 2 :
        max_angle_deviation = two_lines
    else :
        max_angle_deviation = one_lines
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    return stabilized_steering_angle

# 라인이 그려진 이미지 화면에 보여줌
def preview_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

# 각도 계산 및 라인 따라 이미지 생성
def control(frame, lane_lines):
    global curr_steering_angle

    if len(lane_lines) == 0:
        if drv.getSpeed() != 0: 
            drv.setSpeed(0)
        return frame
    else:
        if drv.getSpeed() == 0:
            drv.setSpeed(current_speed)

    new_steering_angle = calc_steering_angle(frame, lane_lines)
    curr_steering_angle = stabilize_angle(curr_steering_angle, new_steering_angle, len(lane_lines))
        
    #------------------------------------------------------
    car_steering = (curr_steering_angle - 90)/90
    
    drv.steering = car_steering 
    print(car_steering)
    #------------------------------------------------------
    
    return preview_line(frame, curr_steering_angle)

def calc_object(class_ids, confidences):
    # 탐지된 값이 없으면 넘김
    if class_ids == []:
        return None, 1, False
    
    # 가장 확률이 높은 것 한 개만 계산
    maxConfidence = max(confidences)
    maxDetectionIndex = confidences.index(maxConfidence)
    classNum = class_ids[maxDetectionIndex]
    
    # 변환 맵
    classNumMap = {0: "plastic_bag", 1: "elk", 2: "person", 3: "traffic_red", 4: "traffic_green", 5: "left_sign", 6: "right_sign", 7: "stop_sign"}
    objectMap = {'stop_sign': 7, 'right_sign': 4, 'left_sign': 4, 'plastic_bag': 2, 'elk': 2, 'person': 2, 'traffic_red': 7, 'traffic_green': 7}
    
    # 확률 threshold 결정 (50 or 60)
    if float(maxConfidence) > 0.7:
        class_name = classNumMap[classNum]
        mode = objectMap[class_name]
        return class_name, mode, True
    
    return None, 1, False


def detect_object(frame):
    global cam, detect
    mode = 1
    flag = False
    
    h, w, c = frame.shape
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    YOLO_net.setInput(blob)
    detections = YOLO_net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

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
    
    # 탐지된 객체 중 가장 높은 확률의 객체 계산
    detect, mode, flag = calc_object(class_ids, confidences)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

            # 경계상자와 클래스 정보 이미지에 입력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

    
    return frame, flag, mode

def line_trace_drive():
    global current_speed, cam

    mode = 1    # 다음 어떤 모드로 전환할지의 대한 변수
    current_speed = 40
    drv.setSpeed(current_speed)
    drv.forward()

    while True:
        frame = cam.value

        # 라인 트레이싱 진행
        lane_lines, lane_lines_image = detect_lane(frame)
        line_preview_image = control(lane_lines_image, lane_lines)

        cv2.imshow("CSI Camera", line_preview_image)
        
        # 객체 탐지 진행
        detect_preview_image, flag, mode = detect_object(frame)
        cv2.imshow("Object Detect", detect_preview_image)

        cv2.waitKey(1)

        if flag == True:
            drv.stop()
            cv2.destroyWindow("CSI Camera")
            cv2.destroyWindow("Object Detect")
            break

    return mode

# 돌발상황 주행
def unexpected_drive():
    global current_speed, detect
    
    drive_mode = 2
    
    # ====================================
    # plastic_bag, elk, person을 선택하여 미션을 수행합니다.
    # ====================================
    if detect == "":
        pass
    else:
        drive_mode = 1
        return drive_mode
            
    # 오르막을 구현했다면 drive_mode = 6
    # 구현하지 못했다면 drive_mode = 1
    drive_mode = 1
    return drive_mode

# 경사로 주행
def ramp_drive():
    global current_speed, detect, cam
    
    drive_mode = 6
    speed = current_speed
    
    # ====================================
    # TODO: 오르막 구간에서는 빠른 주행으로 속도를 유지합니다.
    # TODO: 평지 구간에서는 이전 속도를 유지합니다.
    # TODO: 내리막 구간에서는 느린 주행으로 속도를 유지합니다.
    # ====================================
    return drive_mode

# 장애물 주행
def obstacle_drive():
    global detect, cam
    
    drive_mode = 3
    
    # ====================================
    # TODO: 좁은 길 장애물 주행을 합니다.
    # TODO: 두번째 빨간 불을 탐지 후에 line_tracing으로 복귀합니다.
    # ====================================
    
    return drive_mode

# 회전 주행
def turn_drive():
    global detect
    
    # 이 아래는 모두 스켈레톤 입니다.
    # ====================================
    # TODO: 좌회전 혹은 우회전 표지판을 탐지하여 주행을 수행합니다.
    # TODO: 회전 후에는 전면 주차 모드로 전환합니다.
    # ====================================
    if detect == "":
        pass
    
    # 전면 주차 모드로 전환
    return 

# 전면 주차
def front_parking():
    global current_speed
    
    # ====================================
    # TODO: 전면 주차를 수행합니다.
    # ====================================
    # 종료상태로 모드 전환
    return 10

def traffic_light():
    global detect, traffic_count, cam
    
    if detect == "traffic_red":
        print("======detect_red_light=======")
        drv.stop()
        # 표지판 탐지 진행 (초록불까지)
        while True:
            frame = cam.value
            
            detect_preview_image, flag, drive_mode = detect_object(frame)
            cv2.imshow("Object Detect", detect_preview_image)
        
            cv2.waitKey(1)
            if flag == True and detect == "traffic_green":
                cv2.destroyWindow("Object Detect")
                return 7
    elif detect == "traffic_green":
        print("======detect_green_light=======")
        drv.forward()
        time.sleep(1)
        if traffic_count == 0:
            traffic_count += 1
            # 장애물 주행
            return 3
        else:
            # 라인 트레이싱
            return 1
    else:
        print("======= error handle =======")
        return 1
                
    
def main():
    # ===============================
    # 차량 제어 시작
    # ===============================
    LINE_TRACE = 1
    EVENT1 = 2
    EVENT2 = 3
    EVENT3 = 4
    EVENT4 = 5
    HIDDEN = 6
    LIGHT = 7
    FINISH = 10
    drive_mode = LINE_TRACE 
    # ===========================================================
    # 기본적인 line_trace가 동작.
    # 이 후, 표지판을 마주치면 마주친 표지판을 토대로 이벤트 함수 실행
    # ===========================================================
    
    while drive_mode != FINISH:
        if drive_mode == LINE_TRACE:
            print("------- line_trace_driving --------")
            drive_mode = line_trace_drive()
        elif drive_mode == EVENT1:
            print("------- unexpected_situation --------")
            drive_mode = unexpected_drive()
        elif drive_mode == EVENT2:
            print("------- passing_obstacles --------")
            drive_mode = obstacle_drive()
        elif drive_mode == EVENT3:
            print("------- turn_section --------")
            drive_mode = turn_drive()
        elif drive_mode == EVENT4:
            print("------- parking_mode --------")
            drive_mode = front_parking()
        elif drive_mode == HIDDEN:
            print("------- ramp_section --------")
            drive_mode = ramp_drive()
        elif drive_mode == LIGHT:
            print("------- traffic_light_wait --------")
            drive_mode = traffic_light()

    drv.stop()
    # ========================================
    # 주행 완료시 앞뒤 불빛을 깜빡거리게 함.
    # ========================================
    print("------- FINISH --------")
    for _ in range(2):
        drv.setLamp(1, 1)
        time.sleep(2)
        drv.setLamp(0, 0)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()