from sympy import im
from jetbotSim import Robot, Camera
import numpy as np
import cv2
from time import perf_counter, sleep
from copy import deepcopy
import cornertest

prev_time = perf_counter()
WIDTH = 1280
HEIGHT = 720
HALF_WIDTH = 640
STOP_FLAG = 0
REFERENCE_ROW = -250
FRAME_COUNTER = 0
PROPORTION = 0.1
AVOIDANCE_MOVEMENT = {'M':0,'L':150,'R':-150}
        
class WindDown:
    def __init__(self,speed) -> None:
        self.total_steps = 30
        self.steps = self.total_steps
        self.original_speed = speed
    def reset(self):
        self.steps = self.total_steps
    def next_value(self):
        out = self.original_speed*(self.steps/self.total_steps)
        if self.steps > 0:
            self.steps -= 1
        # else:
        #     print("stop")
        return out

class OptiFlowV2:
    def __init__(self):
        self.prev_gray = None
        self.frame_count = 0
        self.p0 = None
        self.colors = np.random.randint(0, 255, (100, 3))
        self.feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    def flow(self,frame_gray):
        if self.frame_count%10 == 0:
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **self.feature_params)
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return np.zeros_like(self.prev_gray)
        mask = np.zeros_like(self.prev_gray)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.p0, None, **self.lk_params)
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), self.colors[i].tolist(), 2)
        self.prev_gray = frame_gray
        self.frame_count += 1
        return mask

robot = Robot()
camera = Camera()
wind = WindDown(0.1)
optical = OptiFlowV2()

def execute(change):
    global prev_time,STOP_FLAG
    curr_time = perf_counter()
    time_step = curr_time-prev_time
    curr_frame = change['new']
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV) #Use HSV instead of BGR for easier color filtration
    red_frame = cv2.inRange(hsv_frame, (160, 50, 20), (180, 255, 255))
    red_frame = red_frame + cv2.inRange(hsv_frame, (0,50,20), (10,255,255)) #Filter out non-red colors, since red~orange spans from hue 0~20 and 160~180, a two step process is needed
    black_frame = cv2.inRange(hsv_frame,(0,0,0),(180,255,150)) #Black colors, for object avoidance
    ref_row = red_frame[REFERENCE_ROW,:] #Referenced row for red line tracking
    lateral_bias = 0 # +250 to 250 should be safely within boundry
    left = -1
    best_len,mid = 0,-1
    for x,dot in enumerate(ref_row): # Find the geometric median of the largest red patch in the referenced row
        if dot > 0:
            if left == -1:
                left = x
            if x-left+1 > best_len:
                best_len = x-left+1
                mid = (x+left)//2
        else:
            left = -1
    a_move = cornertest.obstacle_avoidance(curr_frame,600)
    if a_move == 'L':
        lateral_bias = 150
    elif a_move == 'R':
        lateral_bias = -150
    target = HALF_WIDTH+lateral_bias # The target point that we try to match to the red patch median
    mid_diff = abs(mid-target)/WIDTH
    if best_len == 0: # Stop the robot gradually if we can't detect the red reference line
        # robot.stop()
        robot.forward(wind.next_value())
        STOP_FLAG = 5
    elif STOP_FLAG > 0:
        robot.forward(wind.next_value())
        STOP_FLAG -= 1
    else:
        robot.forward(0.1)
        wind.reset()
        if mid > target+20: #Proportion controller
            robot.add_motor(0.025*mid_diff,-0.025*mid_diff)
        elif mid < target-20:
            robot.add_motor(-0.025*mid_diff,0.025*mid_diff)
    
    out_str = str(mid)
    if mid > target:
        out_str = "   " + out_str + ">>>"
    elif mid < target:
        out_str = "<<<" + out_str + "   "
    else:
        out_str = "   " + out_str + "   "
    
    prev_time = curr_time
    #print(out_str)
    cv2.imshow('frame',curr_frame)
    #mask_frame = optical.flow(cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY))
    #cv2.imshow('mask',cv2.cvtColor(mask_frame,cv2.COLOR_GRAY2BGR)+curr_frame)

camera.observe(execute)