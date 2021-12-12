import cv2

cap = cv2.VideoCapture('1232.avi')

lowerBody = cv2.CascadeClassifier('person.xml')


while True:

    success, img = cap.read()

    res = lowerBody.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1)

    for (x, y, w, h) in res:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("sda")
        break