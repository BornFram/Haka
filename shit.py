import cv2
import numpy as np

import time

CONFIDENCE = 0.7
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "cfg/yolov3.cfg"
weights_path = "yolov3.weights"
font_scale = 1
thickness = 1
LABELS = open("data/coco.names").read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# ln = net.getLayerNames()
# ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

ln = ['yolo_82', 'yolo_94', 'yolo_106']

print(ln)

cap = cv2.VideoCapture('123.avi')

timer = 0

while True:

    timer += 1

    print(timer)

    if timer:
        _, image = cap.read()
        image = cv2.resize(image, (800, 600))
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (800, 800), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.perf_counter()
        layer_outputs = net.forward(ln)
        time_took = time.perf_counter() - start
        print("Time took:", time_took)
        boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness = 1

    if len(idxs) > 0:

        for i in idxs.flatten():

            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            #[[y+h,x],[y+h+20,x-10],[y+h+20,x+w+10],[y+h,x+w]]
            coef = 6; 
            coord_zon = lu, ld, rd, ru = \
            (() , ()),\
            #(x-(x*np.cos(20)    -(y+h-h//6)*np.sin(20)) , y+h-((y+h-h//6)*np.cos(20)  + x*np.sin(20))),\
            #(x+(1*np.cos(-20)   - (y+h+h//6)*np.sin(-20)) , y+h+((y+h+h//6)*np.cos(-20)   + x*np.sin(-20))),\
            #(x+w-(1*np.cos(20)  - (y+h-h//6)*np.sin(20)) , y+h-((y+h-h//6)*np.cos(20) + (x+w)*np.sin(20))),\
            #(x+w+(1*np.cos(-20) - (y+h+h//6)*np.sin(-20)) , y+h+((y+h-h//6)*np.cos(-20)   + (x+w)*np.sin(-20)))
            #((x+w)*np.cos(-20)+ (y+h+h//coef)*np.sin(-20) , (y+h+h//coef)*np.cos(-20)- (x+w)*np.sin(-20)),\
            #((x+w)*np.cos(20) + (y+h-h//coef)*np.sin(20) , (y+h-h//coef)*np.cos(20)  - (x+w)*np.sin(20))
            
            pts = np.array([(x-50,y+h-30),(x+50,y+h+30),[x+w+50,y+h+30],[x+w-50,y+h-30]], np.int32) 
            #pts = np.array([coord_zon], np.int32) 
            pts = pts.reshape((-1,1,2)) 
            cv2.polylines(image,[pts],True,(0,0,255))

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"

            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)

            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    cv2.imshow("image", image)
    if cv2.waitKey(1) == 27:
        break
print(ln)
cap.release()
cv2.destroyAllWindows()