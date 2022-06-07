from jetbotSim import Robot, Camera
import numpy as np
import cv2
from time import perf_counter

prev_time = perf_counter()
WIDTH = 1280
HEIGHT = 720
HALF_WIDTH = 640
STOP_FLAG = 0
REFERENCE_ROW = -250

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
        return out

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
        robot.forward(0.2)
        wind.reset()
        if mid > target+20: #Proportion controller
            robot.add_motor(0.05*mid_diff,-0.05*mid_diff)
        elif mid < target-20:
            robot.add_motor(-0.05*mid_diff,0.05*mid_diff)

    red_frame_out = cv2.cvtColor(red_frame+black_frame,cv2.COLOR_GRAY2BGR)
    out_str = str(mid)
    if mid > target:
        out_str = "   " + out_str + ">>>"
    elif mid < target:
        out_str = "<<<" + out_str + "   "
    else:
        out_str = "   " + out_str + "   "
    cv2.rectangle(red_frame_out,(mid-10,HEIGHT+REFERENCE_ROW-20),(mid+10,HEIGHT+REFERENCE_ROW),(0,0,255),3)
    cv2.rectangle(red_frame_out,(target-10,HEIGHT+REFERENCE_ROW-20),(target+10,HEIGHT+REFERENCE_ROW),(0,255,0),3)
    cv2.putText(red_frame_out,out_str,(mid-75,HEIGHT+REFERENCE_ROW-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow("camera", red_frame_out)
    prev_time = curr_time

robot = Robot()
camera = Camera()
wind = WindDown(0.2)
camera.observe(execute)
