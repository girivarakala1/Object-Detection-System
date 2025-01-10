import cv2, cvzone
import numpy as np
from ultralytics import YOLO
import math
from sort import *

model = YOLO("C:\\Users\\giriv\\Downloads\\best.pt")

classNames = ['baby-boy', 'baby-girl', 'boy', 'girl', 'man', 'old-man', 'old-women', 'women']


def intersection_points(points):
    line = []
    points.append(points[0])
    for num in range(len(points)-1):
        point1 = points[num]
        point2 = points[num+1]
        # ((89, 422), (342, 249))
        x1, y1, x2, y2 = point1[0][0], point1[0][1], point1[1][0], point1[1][1]
        x3, y3, x4, y4 = point2[0][0], point2[0][1], point2[1][0], point2[1][1]
        if (x2-x1) == 0:
            x1 = x2-1

        if (x3-x4) == 0:
            x3 = x4 -1

        m1 = (y2-y1)/(x2-x1)
        m2 = (y4-y3)/(x4-x3)
        c1 = (m1 * x1) - y1
        c2 = (m2 * x3) - y3
        if m1 == m2:
            return None
        # let (x0, y0) is intersecting point between 2 lines
        x0 = (c1 - c2)/(m1 - m2)
        y0 = ((c1*m2)-(c2 * m1))/(m1 - m2)

        # to find p(x0, y0) in between first line points
        ab = math.sqrt((x2-x1)**2 + ((y2-y1)**2))
        bp = math.sqrt((x2-x0)**2 + ((y2-y0)**2))
        ap = math.sqrt((x0-x1)**2 + ((y0-y1)**2))

        if round(ab, 6) == round(bp + ap, 6):
            line.append([int(x0), int(y0)])
        else:
            print(ab, bp+ap, bp, ap)
    if len(line) == len(points)-1:
        global_co_ordinates.append(line)
        return line
    print(len(line), len(points)-1)
    return None



def mask_coordinates(frame_pic = ""):
    # Function to handle mouse events
    def draw_line(event, x, y, flags, param):
        global drawing, line_start, line_end, co_ordinates, line_points

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            line_start = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            line_end = (x, y)
            co_ordinates.append((line_start, line_end))
            cv2.line(img, line_start, line_end, (0, 255, 0), 2)
            #cv2.imshow('Image with Lines', img)

    # Read the uploaded image
    img = cv2.imread(frame_pic)

    # Initialize global variables
    drawing = False
    line_start = (-1, -1)
    line_end = (-1, -1)


    # Create a window and set the callback function for mouse events
    cv2.namedWindow('Image with Lines')
    cv2.setMouseCallback('Image with Lines', draw_line)

    while True:
        cv2.imshow('Image with Lines', img)
        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to capture line coordinates
        if key == ord('c') and line_start != (-1, -1) and line_end != (-1, -1):
            print(f'Line Start: {line_start}, Line End: {line_end}')
            break

        # Press 'esc' to exitc
        elif key == 27:
            break

    cv2.destroyAllWindows()



def masked_fun(multi_points, height, width):
    # Create a black image
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image = cv2.resize(image, (width, height))

    for points in multi_points:
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw a filled irregular shape on the black image

        cv2.fillPoly(image, [pts], color=(255, 255, 255))


    # Display the result
    cv2.imwrite("mask.jpg", image)


def video_setup(success, first_frame):
    if success:
        # Save the first frame as an image
        cv2.imwrite("first_frame.jpg", first_frame)
        height, width, _ = first_frame.shape

    frame_pic = "C:\\Users\\giriv\\Opencv_practice\\ch_5 Running yolo\\project1 Count\\first_frame.jpg"
    mask_coordinates(frame_pic)
    cv2.imshow('Image with Lines', first_frame)
    lines = intersection_points(co_ordinates)
    if lines == None:
        print("draw lines to create enclosed area")
        return None
    masked_fun(global_co_ordinates, height, width)
    return 1

def inside(point, cx, cy):
    x1, y1, x2, y2 = point[0][0], point[0][1], point[1][0], point[1][1]
    m = (y2 - y1) / (x2 - x1)
    c = y1 - (m * x1)
    # y = mx+c == y-c/m
    x0, y0 = cx,cy
    if x2 == x1 and min(y1, y2) <= y0 <= max(y1, y2):
        dist = abs(x1 - x0)
    elif y2 == y1 and min(x1, x2) <= x0 <= max(x1, x2):
        dist = abs(y1 - y0)
    else:
        alpha = math.atan(m)
        angle = math.pi / 2 - alpha
        dx = 50 * math.sin(angle)
        dy = 50 * math.cos(angle)
        min_x = min(x1 - dx, x1 + dx, x2 - dx, x2 + dx)
        max_x = max(x1 - dx, x1 + dx, x2 - dx, x2 + dx)
        min_y = min(y1 - dy, y1 + dy, y2 - dy, y2 + dy)
        max_y = max(y1 - dy, y1 + dy, y2 - dy, y2 + dy)
        if min_x <= x0 <= max_x and min_y <= y0 <= max_y:
            dist = abs(y0 - (m * x0) - c) / math.sqrt(1 + m ** 2)
        else:
            return False
    if dist <= 50:
        return True
    return False
def calculate_area(value, cx,cy):
    value.append(value[0])
    a = 0
    # A = (1/2) |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|,
    for i in range(len(value)-1):

        a = a + (1/2) * abs(cx*(value[i][1] - value[i+1][1]) + value[i][0]*(value[i+1][1] - cy) + value[i+1][0]*(cy - value[i][1]))
    return a


def region_area(data):
    for value in data:
        cx = cy = 0
        for point in value:
            cx+=point[0]
            cy+=point[1]
        cx = cx/len(value)
        cy = cy/len(value)
        area = calculate_area(value, cx,cy)
        reg_area.append(area)



def main():
    while True:
        print(*classNames)
        print("Enter 1 or more class names with space separated to detect :  ")
        class_list = input().split(" ")
        error_names = []
        for name in class_list:
            if name not in classNames:
                error_names.append(name)
        if len(error_names) == 0:
            break
        else:
            print(f"{error_names} not mached in yolo class names here is list of class names")
            print(classNames)
            exit_opt = input("Do you want to exit y/n")
            if exit_opt in ["y", "yes"]:
                break

    option = input(" Do you have video file path y/n : ")
    if option.lower() == "y" or option.lower() == "yes":
        #path = input(r"Enter video file path: ")
        path = "C:\\Users\\giriv\\Object-Detection-System\\Video.mp4"
        #path = path.replace("\\", "\\\\")
        cap = cv2.VideoCapture(path)
    else:
        path = input("Enter 0 for web cam or 1 for external cam")
        cap = cv2.VideoCapture(int(path))  # for webcam
        cap.set(3, 1280)
        cap.set(4, 720)

    # Create a white image
    h, w = 512, 512
    white_image = np.ones((h, w, 3), dtype=np.uint8) * 255

    print("Do you want to detect full video screen or only some part")
    masked_pic = input("only some part y/n : ")
    if masked_pic == "y" or masked_pic == "yes":
        num_of_regions = input("how many regions you want to create 1 or more : ")
    else:
        num_of_regions = "1"

    if num_of_regions.isdigit() and int(num_of_regions) > 1:
        num_of_regions = int(num_of_regions)
    else:
        num_of_regions = 1
    n = num_of_regions

    while True:
        global co_ordinates, global_co_ordinates, reg_area
        co_ordinates = []
        if masked_pic.lower() == "y" or masked_pic.lower() == "yes":
            success, first_frame = cap.read()
            res = video_setup(success, first_frame)
            if res != 1:
                request = input("Do you want to try again y/n : ")
                if request.lower() == "n" or request.lower() == "no":
                    print("problem")
                    return
                continue
            n -=1
            if n == 0:
                break
        else:
            cv2.imwrite("mask.jpg", white_image)
            break
    co_ordinates = []
    totalCounts = []
    decision = "no"
    if decision.lower() == "y" or decision.lower() == "yes":
        for i in range(num_of_regions):
            totalCounts.append([])
        n = num_of_regions
        frame_path = "C:\\Users\\giriv\\Object-Detection-System\\first_frame.jpg"
        while True:
            mask_coordinates(frame_path)
            n-=1
            if n == 0:
                break
    else:
        if masked_pic == "y" or masked_pic == "yes":
            count_decision = input("Count seperatly based on regions y/n : ")
        else:
            count_decision = "no"
        if count_decision.lower() == "y" or count_decision.lower() == "yes":
            region_area(global_co_ordinates)
            for i in range(num_of_regions):
                totalCounts.append([])

    mask = cv2.imread("C:\\Users\\giriv\\Object-Detection-System\\mask.jpg")
    tracker = Sort(max_age=200, min_hits=3, iou_threshold=0.3)

    while True:
        success, img = cap.read()
        # Resize mask to match the dimensions of img
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        imRegion = cv2.bitwise_and(img, mask)

        results = model(imRegion, stream=True)
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=5)

                # confidence and class
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                currentClass = classNames[cls]

                if currentClass in class_list and conf > 0.3:
                    # cvzone.putTextRect(img, f"{currentClass} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=5)
                    currentArray = np.array([x1, y1, x2, y2, conf])

                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        if num_of_regions == len(co_ordinates):
            for point in co_ordinates:
                cv2.line(img, point[0], point[1], (0, 0, 255), 5)
        else:
            print(co_ordinates)

        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            if decision.lower() == "y" or decision.lower() == "yes":
                for i, point in enumerate(co_ordinates):
                    if inside(point,cx,cy):
                        if totalCounts[i].count(Id) == 0:
                            totalCounts[i].append(Id)
                            cv2.line(img, point[0], point[1], (0, 255, 0), 5)

            elif count_decision.lower() == "y" or count_decision.lower() == "yes":
                for i in range(len(global_co_ordinates)):
                    value = global_co_ordinates[i]
                    result = calculate_area(value, cx, cy)
                    if reg_area[i] >= result:
                        if totalCounts[i].count(Id) == 0:
                            totalCounts[i].append(Id)

            else:
                if totalCounts.count(Id) == 0:
                    totalCounts.append(Id)
                    # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        if num_of_regions > 1:
            for i in range(1, len(totalCounts) + 1):
                y = 50 * i
                cvzone.putTextRect(img, f"Count_{i}: {len(totalCounts[i-1])}", (50,y))
        else:
            cvzone.putTextRect(img, f"Count: {len(totalCounts)}", (50, 50))
        # cv2.putText(img, str(len(totalCounts)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,255), 8)
        cv2.imshow("image", img)
        cv2.waitKey(1)
        print(totalCounts)

    cap.release()
    cv2.destroyAllWindows()




co_ordinates = []
global_co_ordinates = []
reg_area = []

main()





