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
REFERENCE_ROW = -300
GOAL = False

# (hosin)
FRAME_COUNTER = 0
argmax_x_start = -1
argmax_x_end = -1
argmax_y_start = -1
argmax_y_end = -1
obs_history = 0
handler_history = 0

OBSTACLE = False 
HANDLER_MODE = False

def GetMedian(frame, ref_row): # Find the geometric median of the largest red patch in the referenced row
    left = -1
    best_len,mid = 0,-1
    for x, dot in enumerate(frame[ref_row,:]): 
        if dot > 0:
            if left == -1:
                left = x
            if x-left+1 > best_len:
                best_len = x-left+1
                mid = (x+left)//2
        else:
            left = -1
    return best_len, mid

def FindBox(frame): # (hosin)
    global argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, OBSTACLE, FRAME_COUNTER, obs_history
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
            IS_BLACK = False
            IS_RED = False
            if abs(red-avg)/avg < 0.12 and abs(green-avg)/avg < 0.12 and abs(blue-avg)/avg < 0.12:
                IS_FLOOR = True
            else:
                IS_FLOOR = False
            if red < 170 or green < 180 or blue < 190:
                IS_FLOOR = False
            if red < 170 and green < 180 and blue < 190:
                if abs(red-avg)/avg < 0.12 and abs(green-avg)/avg < 0.12 and abs(blue-avg)/avg < 0.12:
                    IS_BLACK = True # black
            if red > avg * 1.5:
                IS_RED = True # red
            if IS_FLOOR:
                segmented[y][x] = [255,255,255]
                obstacle_map[y][x] = 'F'
            elif IS_BLACK:
                segmented[y][x] = [0,0,0]
                obstacle_map[y][x] = 'B'
            elif IS_RED:
                segmented[y][x] = [0,0,255]
                obstacle_map[y][x] = 'R'
            else: # obstacle
                segmented[y][x] = [0,255,255]
                obstacle_map[y][x] = 'O'
    accum_y = 0
    max_y = 0
    heat_y = [-3 for x in range(30)]
    temp_start_y = 0
    for y in range(15, 45):
        yellow_y = [1 for y in range(81)]
        black_y = [1 for y in range(81)]
        threshold = np.power(2, y * 0.2 + 4)
        for x in range(80):
            if obstacle_map[y][x] == 'O':
                yellow_y[x+1] = yellow_y[x] * (x * x * (-0.000333) + x * 0.02667 + 1.4667)
                if yellow_y[x+1] >= threshold:
                    yellow_y[x+1] = threshold
            elif obstacle_map[y][x] == 'B':
                black_y[x+1] = black_y[x] * (x * x * (-0.000333) + x * 0.02667 + 1.4667)
                if black_y[x+1] >= threshold:
                    black_y[x+1] = threshold
        if np.max(yellow_y) == threshold or np.max(black_y) == threshold:
            heat_y[y-15] = 1
        accum_y += heat_y[y-15]
        if accum_y < 0:
            accum_y = 0
        if accum_y == heat_y[y-15]:
            temp_start_y = y
        if accum_y >= max_y:
            max_y = accum_y
            argmax_y_end = y
            argmax_y_start = temp_start_y
    heat_x = [-5 for x in range(80)]
    accum_x = 0
    max_x = 0
    temp_start_x = 0
    for x in range(80):
        yellow_x = [1 for y in range(31)]
        black_x = [1 for x in range(31)]
        for y in range(15, 45):
            if obstacle_map[y][x] == 'O':
                yellow_x[y-14] = yellow_x[y-15] * (2.5 - 0.03 * y)
                if yellow_x[y-14] > 100:
                    yellow_x[y-14] = 64
            elif obstacle_map[y][x] == 'B':
                black_x[y-14] = black_x[y-15] * (2.5 - 0.03 * y)
                if black_x[y-14] > 100:
                    black_x[y-14] = 64
        if np.max(yellow_x) > 50 or np.max(black_x) > 50:
            heat_x[x] = 1
        accum_x += heat_x[x]
        if accum_x < 0:
            accum_x = 0
        if accum_x == heat_x[x]:
            temp_start_x = x
        if accum_x >= max_x:
            max_x = accum_x
            argmax_x_end = x
            argmax_x_start = temp_start_x
    cv2.rectangle(segmented, (argmax_x_start,argmax_y_start), (argmax_x_end,argmax_y_end), (255,0,0), 1)
    #print((argmax_x_start,argmax_y_start), (argmax_x_end,argmax_y_end))
    cv2.imwrite(str(FRAME_COUNTER) + '.png', segmented)
    #cv2.imshow('obstacle',segmented)
    area = (argmax_x_end - argmax_x_start) * (argmax_y_end - argmax_y_start)
    if area > 30 and argmax_y_start >= 15 and argmax_y_end > 30 and argmax_x_end - argmax_x_start > 5 and argmax_y_end > argmax_y_start * 1.2:
        miss = 0
        for y in range(argmax_y_start, argmax_y_end):
            for x in range(argmax_x_start, argmax_x_end):
                if obstacle_map[y][x] == 'F' or  obstacle_map[y][x] == 'R':
                    miss += 1
        if miss > area * 0.3:
            OBSTACLE = False
        else:
            OBSTACLE = True
            obs_history = 0
    else:
        OBSTACLE = False
    return

def execute(change):
    global prev_time, STOP_FLAG, FRAME_COUNTER, argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, GOAL, OBSTACLE, HANDLER_MODE, motion_plan, motion_span, motion_init, obs_history, handler_history # (hosin)
    if GOAL:
        return
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
    best_len, mid = GetMedian(red_frame, REFERENCE_ROW)
    target = HALF_WIDTH

    if handler_history > 10 and not HANDLER_MODE and GetMedian(red_frame, -100)[0] == 0: # Stop the robot instantly if we can't detect the red reference line
        print('\nenter parking area at time ' + str(FRAME_COUNTER))
        robot.stop()
        GOAL = True
        return
    if not HANDLER_MODE and OBSTACLE and best_len > 0:
        HANDLER_MODE = True
        print('\n\nenter handler mode at time ' + str(FRAME_COUNTER))
        target = (argmax_x_start + argmax_x_end) * 8 # center of obstacle
        bottom_side_y = min(44, int(min(1, (argmax_y_end/30 - 1/3)) * (argmax_y_end - argmax_y_start) + argmax_y_start))
        roll = argmax_y_end * argmax_y_end * 0.0000267 - argmax_y_end * 0.00533 + 0.196
        best_length, bottom_mid = GetMedian(red_frame, bottom_side_y * 16) 
        if bottom_mid > target:
            print('right shift')
            rotate = 0.01
            shift = min(0.1, max(0.03, (argmax_x_end * 16 - bottom_mid + argmax_y_end * 4 + 150) * 0.0005))
            if mid > HALF_WIDTH + 50:
                shift += 0.02
            elif mid < HALF_WIDTH - 50:
                shift -= 0.01
        else:
            print('left shift')
            rotate = -0.01
            shift = min(0.1, max(0.03, abs(argmax_x_start * 16 - bottom_mid - argmax_y_end * 4 - 150) * 0.0005))
            if mid < HALF_WIDTH - 50:
                shift += 0.02
            elif mid > HALF_WIDTH + 50:
                shift -= 0.01
        motion_plan = [(rotate,-rotate,'no obstacle'), (shift,shift,'deviate'), (-rotate,rotate,'back and straight'), (roll,roll,'y distance to obstacle')] # operation for each motion planning step
        motion_span = [5, 2, 5, 2] # default upper limits of each step of motion plan in seconds
        motion_init = perf_counter()
        print(bottom_mid, target, shift, motion_init, roll)

    if HANDLER_MODE:
        handler_history = 0
        motion = motion_plan[0]
        robot.set_motor(motion[0], motion[1])
        print(FRAME_COUNTER, '--- handler: ',  str(motion), 't = ', perf_counter())
        if motion[2] == 'no obstacle' and not OBSTACLE and obs_history > 2:
            motion_plan.pop(0)
            motion_span.pop(0)
            motion_span[1] = perf_counter() - motion_init
            motion_init = perf_counter()
        elif motion[2] == 'y distance to obstacle' and abs(GetMedian(red_frame, -300)[1] - HALF_WIDTH) / WIDTH > 0.25:
            print('--- rolling stopped early ---')
            motion_plan.pop(0)
            motion_span.pop(0)
            motion_init = perf_counter()
        elif perf_counter() > motion_init + motion_span[0] or (motion[2] == 'deviate' and GetMedian(red_frame, 600)[1] - GetMedian(red_frame, 400)[1] > 300 and GetMedian(red_frame, 650)[0] == 0):
            if motion[2] == 'back and straight' and (perf_counter() > motion_init + 5 or abs(GetMedian(red_frame, -300)[1] - HALF_WIDTH) / WIDTH > 0.3):
                motion_span[1] = 0.5
                print('--- roll time cut to 0.5 ---')
            motion_plan.pop(0)
            motion_span.pop(0)
            motion_init = perf_counter()
        if len(motion_plan) == 0:
            HANDLER_MODE = False
            print('\nexit handler mode at time ' + str(FRAME_COUNTER))
    else:
        mid_diff = min(0.05, abs(mid - HALF_WIDTH) / WIDTH / 10)
        robot.forward(0.05)
        if mid > target + 20: #Proportion controller
            robot.add_motor(0.025*mid_diff,-0.025*mid_diff)
        elif mid < target - 20:
            robot.add_motor(-0.025*mid_diff,0.025*mid_diff)
        print('guiding mode', mid_diff)

    red_frame_out = cv2.cvtColor(red_frame+black_frame,cv2.COLOR_GRAY2BGR)
    out_str = str(mid)
    if mid > target:
        out_str = "   " + out_str + ">>>"
    elif mid < target:
        out_str = "<<<" + out_str + "   "
    else:
        out_str = "   " + out_str + "   "
    if HANDLER_MODE:
        out_str = str(motion)
    cv2.rectangle(red_frame_out,(mid-10,HEIGHT+REFERENCE_ROW-20),(mid+10,HEIGHT+REFERENCE_ROW),(0,0,255),3)
    cv2.rectangle(red_frame_out,(target-10,HEIGHT+REFERENCE_ROW-20),(target+10,HEIGHT+REFERENCE_ROW),(0,255,0),3)
    if OBSTACLE and FRAME_COUNTER % 2 == 0:
        print('obstacle at time ' + str(FRAME_COUNTER) + '\n')
        cv2.rectangle(red_frame_out, (argmax_x_start*16,argmax_y_start*16), (argmax_x_end*16,argmax_y_end*16), (255,0,0), 3) # (hosin)
    cv2.putText(red_frame_out,out_str,(mid-75,HEIGHT+REFERENCE_ROW-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow("camera", red_frame_out)
    prev_time = curr_time
    obs_history += 1
    handler_history += 1

robot = Robot()
camera = Camera()
camera.observe(execute)
