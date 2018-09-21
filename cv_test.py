import cv2

cap = cv2.VideoCapture(0)                                        # 打开摄像头

while True:

    ret, frame = cap.read()                                      # 读摄像头
    cv2.imshow("video", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):                        # 按q退出
        break

cap.release()            
cv2.destroyAllWindows()                                          # 基本操作
