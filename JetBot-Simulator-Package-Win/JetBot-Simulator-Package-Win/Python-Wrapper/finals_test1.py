from jetbotSim import Robot, Camera
import numpy as np
import cv2
from time import perf_counter

prev_time = perf_counter()
# KERNEL = np.ones((5,5),np.uint8)
def execute(change):
    global prev_time
    curr_time = perf_counter()
    time_step = curr_time-prev_time
    curr_frame = change['new']
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    red_frame = cv2.inRange(hsv_frame, (160, 50, 20), (179, 255, 255))
    red_frame = red_frame + cv2.inRange(hsv_frame, (0,50,20), (10,255,255))
    # print(red_frame.shape)
    last_row = red_frame[-1,:]

    left = -1
    best_len,mid = 0,-1
    for x,dot in enumerate(last_row):
        if dot > 0:
            if left == -1:
                left = x
            if x-left+1 > best_len:
                best_len = x-left+1
                mid = (x+left)//2
        else:
            left = -1
    if mid > 600:
        robot.right(0.05)
    elif mid < 560:
        robot.left(0.05)
    else:
        robot.forward(0.2)
    cv2.imshow("camera", red_frame)

    prev_time = curr_time
robot = Robot()
camera = Camera()
camera.observe(execute)