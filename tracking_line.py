from jetbot import Robot, Camera
import numpy as np
import cv2
from time import perf_counter, sleep
from copy import deepcopy

prev_time = perf_counter()
WIDTH = 1280
HEIGHT = 720
HALF_WIDTH = 640
STOP_FLAG = 0
REFERENCE_ROW = -250
FRAME_COUNTER = 0
PROPORTION = 0.1

class motor1(Robot):
    def __init__(self):
        super().__init__()
        self.l = 0  
        self.r = 0

    def Forward(self, speed):
        self.l = speed
        self.r = speed
        self.forward(speed)

    def ShowSpeed(self):
        print("Speed: ", (self.l, self.r))
        
    
    def AddMotor(self, sl, sr):
        self.l += sl
        self.r += sr
        self.set_motors(self.l, self.r)
        
    def SetMotor(self, sl, sr):
        self.l = sl
        self.r = sr
        self.set_motors(self.l, self.r)
    
    def Stop(self):
        self.stop()
        
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
    
robot = motor1()   
wind = WindDown(0.1)

def execute(frame):
    global prev_time, STOP_FLAG, FRAME_COUNTER
    curr_time = perf_counter()
    time_step = curr_time-prev_time
    FRAME_COUNTER += 1
    curr_frame = frame
    curr_frame[:,:,2]  = (curr_frame[:,:,2] // 5) * 4
    
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
    target = HALF_WIDTH + lateral_bias # The target point that we try to match to the red patch median
    # mid_diff = abs(mid-target)/WIDTH
    mid_diff = (mid - target) / WIDTH
    if best_len == 0: # Stop the robot gradually if we can't detect the red reference line
        # robot.stop()
        robot.Forward(wind.next_value())
        STOP_FLAG = 5
    elif STOP_FLAG > 0:
        robot.Forward(wind.next_value())
        STOP_FLAG -= 1
    else:
        robot.Forward(0.1)
        wind.reset()
        # if mid > target+20: #Proportion controller
        #     robot.add_motor(0.05*mid_diff,-0.05*mid_diff)
        # elif mid < target-20:
        #     robot.add_motor(-0.05*mid_diff,0.05*mid_diff)
        if abs(mid_diff) > (20 / WIDTH):
            robot.AddMotor(PROPORTION * mid_diff, -PROPORTION * mid_diff)

    out_str = str(mid)
    if mid > target:
        out_str = "   " + out_str + ">>>"
    elif mid < target:
        out_str = "<<<" + out_str + "   "
    else:
        out_str = "   " + out_str + "   "
    prev_time = curr_time
    print(out_str)

try:
    camera = Camera.instance(width=1280, height=720)
    while True:
        execute(camera.value)
        # print(camera.value)
        # sleep(1)
        
finally:
    camera.stop()