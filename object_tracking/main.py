import cv2 
cap = cv2.VideoCapture("highway_small.mp4")
# obj detection from stable camera
obj_dec = cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=10)

while True:
    ret,frame = cap.read()
    height,width, _ = frame.shape
    print(height,width)
    # ROI
    roi = frame[360:720,320:1280]
    mask = obj_dec.apply(roi)
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # print(frame.shape)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >100:
            # cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Frame", frame)
    # cv2.imshow("ROI", roi)
    cv2.imshow("mask",mask)
    key = cv2.waitKey(30)
    if key ==27:
        break 

cap.release()
cap.destroyAllWindows()  