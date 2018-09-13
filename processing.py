import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
option = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 2000,
    'threshold': 0.08,
}

TFNet=TFNet(option) #initialises model
capture = cv2.VideoCapture('test22.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()#start time
    ret, frame = capture.read() #ret is true when video is playing
    if ret:
        results = TFNet.return_predict(frame)
        for color, result in zip(colors, results): #makes a list
            #top left and bottom right for box
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)#display frame
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break