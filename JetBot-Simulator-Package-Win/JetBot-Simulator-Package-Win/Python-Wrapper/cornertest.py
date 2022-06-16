import os
import cv2
import natsort

feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 50 )

Y_BIAS = 250
X_BIAS = 640

def obstacle_avoidance(frame,reference_row):
    height,width,_ = frame.shape
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray_frame, mask = None, **feature_params)
    cv2.line(frame,(width//2-X_BIAS,reference_row+Y_BIAS-20),(width//2+X_BIAS,reference_row+Y_BIAS-20),(0,255,0))
    cv2.line(frame,(width//2-X_BIAS,reference_row-Y_BIAS),(width//2+X_BIAS,reference_row-Y_BIAS),(0,255,0))
    left,right = 0,0
    for coord in p0:
        x,y = coord.ravel()
        if reference_row+Y_BIAS >= y >= reference_row-Y_BIAS and width//2+X_BIAS >= x >= width//2-X_BIAS:
            cv2.circle(frame,(int(x),int(y)),5,(255,0,0))
            if x > width//2:
                right += 1
            else:
                left += 1
        else:
            pass
            cv2.circle(frame,(int(x),int(y)),5,(0,0,255))
    
    movement = 'M'
    if left+right > 2:
        if left > right:
            # text = '>>>>'
            movement = "R"
        else:
            movement = "L"
            # text = '<<<<'
        # cv2.putText(frame,text,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
    return movement
if __name__ == '__main__':
    for root,dirs,files in os.walk('../object1/'):
        for f in natsort.natsorted(files):
            frame = cv2.imread(root+f)
            obstacle_avoidance(frame,470)
            cv2.imshow('frame',frame)
            cv2.waitKey(0)