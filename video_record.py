import cv2

cap = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('captured.avi',codec, 25.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) == 27 or ret == False:
        break
    cv2.imshow('frame', frame)
    out.write(frame)
out.release()
cap.release()
cv2.destroyAllWindows()

