import cv2
from pop import Camera
from pop import Pilot
import numpy as np
import math

#---------------------------------------------------
kernel = np.ones((4,4),np.uint8)

sensitivity = 90
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,30+sensitivity])

curr_steering_angle = 0.0
current_speed = None

drv = Pilot.AutoCar()
drv.forward()
#---------------------------------------------------

def detect_edges(frame):
    frame = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)    
    edges = cv2.Canny(mask, 200, 400)

    return edges

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

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        return lane_lines
        
    __height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 2
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

def control(frame, lane_lines):
    global curr_steering_angle

    if len(lane_lines) == 0:
        if drv.getSpeed() != 0: 
            drv.setSpeed(30)
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

def main():
    global current_speed

    current_speed = 30
    drv.setSpeed(current_speed)
    
    cap = Camera(320, 240)

    while True:
        frame = cap.value

        lane_lines, lane_lines_image = detect_lane(frame)
        preview_image = control(lane_lines_image, lane_lines)
        
        cv2.imshow("basic camera", frame);
        cv2.imshow("CSI Camera", preview_image)
        keyCode = cv2.waitKey(10) & 0xFF
        if keyCode == 27 or keyCode == ord('q'):
            break

    drv.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()