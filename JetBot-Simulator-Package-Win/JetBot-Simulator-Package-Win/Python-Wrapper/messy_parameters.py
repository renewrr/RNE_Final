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
history = 999 # how many time frames since the last obstacle detected
left_dir = True # does the vehicle want to turn left or not
last_rotation = 0 # used to rotate back during obstacle avoidance
stuck = 0

OBSTACLE = False 
LOST = False 
FINISHED = False

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
    global argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, OBSTACLE, FRAME_COUNTER, history
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
    yellow_y = [-3 for y in range(30)]
    accum_y = 0
    max_y = 0
    temp_start_y = 0
    for y in range(15, 45):
        obs_y = 0
        black_y = [1 for y in range(81)]
        threshold = np.power(2, y * 0.2 + 4)
        for x in range(80):
            if obstacle_map[y][x] == 'O':
                obs_y += 1
            elif obstacle_map[y][x] == 'B':
                obs_y -= 0.5
                black_y[x+1] = black_y[x] * (x * x * (-0.000333) + x * 0.02667 + 1.4667)
                if black_y[x+1] >= threshold:
                    black_y[x+1] = threshold
        if obs_y > 5 or np.max(black_y) == threshold:
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
            elif obstacle_map[y][x] == 'B':
                obs_x -= 0.2
                black_x[y-14] = black_x[y-15] * (2.5 - 0.03 * y)
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
    #cv2.imshow('obstacle',segmented)
    area = (argmax_x_end - argmax_x_start) * (argmax_y_end - argmax_y_start)
    if area > 30 and argmax_y_start >= 15 and argmax_y_end > 20 and argmax_x_end - argmax_x_start > 5 and argmax_y_end > argmax_y_start * 1.2:
        miss = 0
        for y in range(argmax_y_start, argmax_y_end):
            for x in range(argmax_x_start, argmax_x_end):
                if obstacle_map[y][x] == 'F' or  obstacle_map[y][x] == 'R':
                    miss += 1
        if miss > area / 2:
            OBSTACLE = False
        else:
            OBSTACLE = True
            history = 0
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
    global prev_time, STOP_FLAG, REFERENCE_ROW, FRAME_COUNTER, argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, OBSTACLE, LOST, history, left_dir, stuck, FINISHED, last_rotation # (hosin)
    curr_time = perf_counter()
    time_step = curr_time-prev_time
    FRAME_COUNTER += 1
    history += 1
    curr_frame = change['new']
    if FRAME_COUNTER % 2 == 0: # (hosin)
        FindBox(deepcopy(curr_frame))
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV) #Use HSV instead of BGR for easier color filtration
    red_frame = cv2.inRange(hsv_frame, (160, 50, 20), (180, 255, 255))
    red_frame = red_frame + cv2.inRange(hsv_frame, (0,50,20), (10,255,255)) #Filter out non-red colors, since red~orange spans from hue 0~20 and 160~180, a two step process is needed
    black_frame = cv2.inRange(hsv_frame,(0,0,0),(180,255,150)) #Black colors, for object avoidance
    best_len, mid = GetMedian(red_frame, REFERENCE_ROW)

    if not OBSTACLE and GetMedian(red_frame, 650)[1] - GetMedian(red_frame, 400)[1] > 200:
        deviate = True
        print('deviate')
    elif abs(mid - HALF_WIDTH) / WIDTH > argmax_y_end * 0.0145 - 0.14:
        deviate = True
        print('deviate')
    else:
        deviate = False

    if OBSTACLE:
        target = (argmax_x_start + argmax_x_end) * 8 # center of obstacle
    else:
        target = HALF_WIDTH
        
    if OBSTACLE and FRAME_COUNTER % 2 == 0 and not deviate:
        bottom_side_y = min(44, int((2.04 - argmax_y_end * 0.0038) * (argmax_y_end - argmax_y_start) + argmax_y_start))
        best_length, bottom_mid = GetMedian(red_frame, bottom_side_y) # bottom mid is the middle of the red line at y = bottom_side_of_obstacle
        if bottom_mid > 0:
            shift = 0
        if bottom_mid > target:
            left_dir = False
            shift = (argmax_x_end * 16 - bottom_mid + argmax_y_end * 25 - 50) 
        else:
            left_dir = True
            shift = (bottom_mid - argmax_x_start * 16 + argmax_y_end * 25 - 50)
        if shift > 0:
            shift *= (argmax_y_end * argmax_y_end * 0.000011 - argmax_y_end * 0.0065 + 1)
            mid_diff = shift / WIDTH * 2.7
        else:
            mid_diff = abs(mid - HALF_WIDTH) / WIDTH
        upper_limit = argmax_y_end * argmax_y_end * argmax_y_end * (-0.0000667) + argmax_y_end * argmax_y_end * 0.004 + argmax_y_end * (-0.03833) + 0
        mid_diff = min(mid_diff, max(upper_limit, 0))
        if argmax_y_end > 40 and (target - HALF_WIDTH) / WIDTH > 0.9:
            print('passed an obstacle')
            mid_diff = 0
    else:
        if OBSTACLE and not deviate:
            mid_diff = 0
        else:
            turn = history * history * history * 0.0034 - history * history * 0.08 + history * 0.54 - 0.05
            mid_diff = min(abs(mid - target) / WIDTH, 0.4 * min(turn, 1))
    
    if LOST: # in the process of readjustment
        stuck += 1
        if stuck > 30:
            FINISHED = True
        elif best_len > 0 and abs(mid - HALF_WIDTH) / WIDTH < 0.3:
            LOST = False
            robot.forward(0.1)
            print('end adjustment')
        else:
            if left_dir: # rotate to find red line
                print('rotating left..')
                robot.set_motor(-0.01,0.01)
            else:
                print('rotating right..')
                robot.set_motor(0.01,-0.01)

    if best_len == 0:
        print('\nlost track of red line')
        speed = 0
        if history < 20: # lost track of red line during or after obstacle avoidance
            if not LOST:
                LOST = True
                stuck = 0
                print('beginning readjustment...')
                robot.backward(0.02)
                speed = -0.02
                if left_dir: # rotate to find red line
                    print('rotating left..')
                    robot.set_motor(-0.01,0.01)
                else:
                    print('rotating right..')
                    robot.set_motor(0.01,-0.01)
            elif OBSTACLE and FRAME_COUNTER % 2 == 0:
                if target > HALF_WIDTH:
                    print('rotating left..')
                    robot.set_motor(-0.01,0.01)
                else:
                    print('rotating right..')
                    robot.set_motor(0.01,-0.01)
        elif FINISHED or not LOST: # Stop the robot gradually if we can't detect the red reference line
            print('reaching goal...')
            robot.forward(wind.next_value())
            STOP_FLAG = 3
    elif STOP_FLAG > 0:
        speed = wind.next_value()
        robot.forward(speed)
        STOP_FLAG -= 1
    else:
        wind.reset()
        if history < 3:
            speed = max(0.01 - mid_diff / 10, 0)
            offset_param = 0.9
        else:
            speed = max(0.1 - mid_diff / 5, 0.01)
            offset_param = 0.1
        if deviate:
            speed /= 2
        robot.forward(speed)
        if mid > target + 10: # Proportion controller
            mid_diff -= last_rotation * offset_param
            mid_diff = max(mid_diff, 0)
            robot.add_motor(0.025*mid_diff,-0.025*mid_diff)
            last_rotation = mid_diff
            print('rotate right')
        elif mid < target - 10:
            mid_diff += last_rotation * offset_param
            mid_diff = max(mid_diff, 0)
            robot.add_motor(-0.025*mid_diff,0.025*mid_diff)
            last_rotation = -mid_diff
            print('rotate left')
        else:
            print('straight walk')

    if not LOST:
        if mid < target:
            left_dir = True
        else:
            left_dir = False

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
    if OBSTACLE and FRAME_COUNTER % 2 == 0  and not deviate: # obstacle handler mode
        print('\nobstacle at time ' + str(FRAME_COUNTER))
        print(round(mid_diff, 3), round(upper_limit, 3), round(shift, 3), round(speed, 3))
        cv2.rectangle(red_frame_out, (argmax_x_start*16,argmax_y_start*16), (argmax_x_end*16,argmax_y_end*16), (255,0,0), 3)
    else: # red line guiding mode
        print('guiding mode at time ' + str(FRAME_COUNTER))
        print(round(mid_diff,3), round(speed, 3))
    cv2.putText(red_frame_out,out_str,(mid-75,HEIGHT+REFERENCE_ROW-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow("camera", red_frame_out)
    prev_time = curr_time

robot = Robot()
camera = Camera()
wind = WindDown(0.05)
camera.observe(execute)
