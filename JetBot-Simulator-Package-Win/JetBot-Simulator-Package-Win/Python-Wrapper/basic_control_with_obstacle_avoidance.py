from jetbotSim import Robot, Camera
import numpy as np
import cv2
from time import perf_counter
from copy import deepcopy # (hosin)

prev_time = perf_counter()
WIDTH = 1280
HEIGHT = 720
HALF_WIDTH = 640
STOP_FLAG = 0
REFERENCE_ROW = -250

# (hosin)
FRAME_COUNTER = 0
argmax_x_start = 0
argmax_x_end = 0
argmax_y_start = 0
argmax_y_end = 0

OBSTACLE = False # (hosin) 

def FindBox(frame): # (hosin)
    global argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, OBSTACLE, FRAME_COUNTER
    small = cv2.resize(frame, (80,45))
    kernel = cv2.Mat(np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]]))
    filtered = cv2.filter2D(small, -1, kernel)
    segmented = filtered.copy()
    obstacle_map = [[None for x in range(80)] for y in range(45)]
    for y in range(45):
        for x in range(80):
            pixel = filtered[y][x]
            blue = int(pixel[0])
            green = int(pixel[1])
            red = int(pixel[2])
            avg = (red + green + blue) / 3 + 1
            IS_LANE = False
            if abs(red-avg)/avg < 0.12 and abs(green-avg)/avg < 0.12 and abs(blue-avg)/avg < 0.12:
                IS_FLOOR = True
            else:
                IS_FLOOR = False
            if red < 170 or green < 180 or blue < 190:
                IS_FLOOR = False
            if red < 170 and green < 180 and blue < 190:
                if abs(red-avg)/avg < 0.12 and abs(green-avg)/avg < 0.12 and abs(blue-avg)/avg < 0.12:
                    IS_LANE = True # black
            if red > avg * 1.5:
                IS_LANE = True # red
            if IS_FLOOR:
                segmented[y][x] = [255,255,255]
                obstacle_map[y][x] = 'F'
            elif IS_LANE:
                segmented[y][x] = [0,0,0]
                obstacle_map[y][x] = 'L'
            else: # obstacle
                segmented[y][x] = [0,255,255]
                obstacle_map[y][x] = 'O'
    #cv2.imshow('segmented', segmented)
    yellow_y = [-3 for y in range(30)]
    accum_y = 0
    max_y = 0
    temp_start_y = 0
    for y in range(15, 45):
        obs_y = 0
        black_y = [1 for y in range(81)]
        for x in range(80):
            if obstacle_map[y][x] == 'O':
                obs_y += 1
            elif obstacle_map[y][x] == 'L':
                obs_y -= 0.5
                black_y[x+1] = black_y[x] * 2
                if black_y[x+1] > 500:
                    black_y[x+1] = 256
        if obs_y > 5 or np.max(black_y) > np.power(2, y * 0.2 + 3):
            yellow_y[y-15] = 1
        accum_y += yellow_y[y-15]
        if accum_y < 0:
            accum_y = 0
        if accum_y == yellow_y[y-15]:
            temp_start_y = y
        if accum_y >= max_y:
            max_y = accum_y
            argmax_y_end = y
            argmax_y_start = temp_start_y
    yellow_x = [-5 for x in range(80)]
    accum_x = 0
    max_x = 0
    temp_start_x = 0
    for x in range(80):
        obs_x = 0
        black_x = [1 for x in range(31)]
        for y in range(15, 45):
            if obstacle_map[y][x] == 'O':
                obs_x += 1
            elif obstacle_map[y][x] == 'L':
                obs_x -= 0.2
                black_x[y-14] = black_x[y-15] * 2
                if black_x[y-14] > 100:
                    black_x[y-14] = 64
        if obs_x > 5 or np.max(black_x) > 50:
            yellow_x[x] = 1
        accum_x += yellow_x[x]
        if accum_x < 0:
            accum_x = 0
        if accum_x == yellow_x[x]:
            temp_start_x = x
        if accum_x >= max_x:
            max_x = accum_x
            argmax_x_end = x
            argmax_x_start = temp_start_x
    cv2.rectangle(segmented, (argmax_x_start,argmax_y_start), (argmax_x_end,argmax_y_end), (255,0,0), 1)
    cv2.imwrite(str(FRAME_COUNTER) + '.png', segmented)
    cv2.imshow('obstacle',segmented)
    area = (argmax_x_end - argmax_x_start) * (argmax_y_end - argmax_y_start)
    if area > 25 and argmax_y_start >= 15 and argmax_y_end < 38 and argmax_y_end > 20 and argmax_x_end - argmax_x_start > argmax_y_start * 0.2 and argmax_y_end > argmax_y_start * 1.2:
        whites = 0
        for y in range(argmax_y_start, argmax_y_end):
            for x in range(argmax_x_start, argmax_x_end):
                if obstacle_map[y][x] == 'F':
                    whites += 1
        if whites > area / 2:
            OBSTACLE = False
        else:
            OBSTACLE = True
    else:
        OBSTACLE = False
    return

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
    global prev_time,STOP_FLAG, FRAME_COUNTER, argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, OBSTACLE # (hosin)
    curr_time = perf_counter()
    time_step = curr_time-prev_time
    FRAME_COUNTER += 1
    curr_frame = change['new']
    if FRAME_COUNTER % 2 == 0: # (hosin)
        FindBox(deepcopy(curr_frame))
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV) #Use HSV instead of BGR for easier color filtration
    red_frame = cv2.inRange(hsv_frame, (160, 50, 20), (180, 255, 255))
    red_frame = red_frame + cv2.inRange(hsv_frame, (0,50,20), (10,255,255)) #Filter out non-red colors, since red~orange spans from hue 0~20 and 160~180, a two step process is needed
    black_frame = cv2.inRange(hsv_frame,(0,0,0),(180,255,150)) #Black colors, for object avoidance
    ref_row = red_frame[REFERENCE_ROW,:] #Referenced row for red line tracking
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
    if OBSTACLE and FRAME_COUNTER % 2 == 0:
        left = -1
        best_length,bottom_mid = 0,-1
        for x,dot in enumerate(red_frame[max(argmax_y_end, 25)*16,:]): # Find the geometric median of the largest red patch in the referenced row
            if dot > 0:
                if left == -1:
                    left = x
                if x-left+1 > best_length:
                    best_length = x-left+1
                    bottom_mid = (x+left)//2
            else:
                left = -1
        target = (argmax_x_start + argmax_x_end) * 8
        if bottom_mid > target:
            shift = argmax_x_end * 16 - bottom_mid + argmax_y_end * 13 - 150
        else:
            shift = bottom_mid - argmax_x_start * 16 + argmax_y_end * 13 - 150
        mid_diff = max(shift, 0) / WIDTH / 10
    else:
        target = HALF_WIDTH
        mid_diff = abs(mid - target) / WIDTH
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
        if mid > target+20: # Proportion controller
            robot.add_motor(0.025*mid_diff,-0.025*mid_diff)
        elif mid < target-20:
            robot.add_motor(-0.025*mid_diff,0.025*mid_diff)

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
    if OBSTACLE and FRAME_COUNTER % 2 == 0:
        print('obstacle at time ' + str(FRAME_COUNTER))
        print(argmax_x_end, argmax_x_start, argmax_y_end, argmax_y_start, mid_diff, shift, bottom_mid)
        cv2.rectangle(red_frame_out, (argmax_x_start*16,argmax_y_start*16), (argmax_x_end*16,argmax_y_end*16), (255,0,0), 3) # (hosin)
    cv2.putText(red_frame_out,out_str,(mid-75,HEIGHT+REFERENCE_ROW-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow("camera", red_frame_out)
    prev_time = curr_time

robot = Robot()
camera = Camera()
wind = WindDown(0.2)
camera.observe(execute)
