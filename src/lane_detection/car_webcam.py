import cv2 
from cv_bridge import CvBridge, CvBridgeError

# Open the ZED camera
bridge = CvBridge()
cap = cv2.VideoCapture(1)
if cap.isOpened() == 0:
    exit(-1)

cap.set(3, 1280) # Modify webcam resolution
cap.set(4, 720)

ret, frame = cap.read()


height, width, layers = frame.shape

size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 20, size)

if not out.isOpened():
    print("error")
    cap.release()
    sys.exit()

count = 0

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    
    out.write(frame)
    print("write")
    count = count + 1
    if count == 100:
        out.release()
        break

print("end")
