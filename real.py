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
visibility = -1
last_frame = None

OBSTACLE = False 
HANDLER_MODE = False
PARKING_MODE = False

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

def FeaturePoints(image): # count number of feature points to find parking line
    frame = cv2.resize(image, (144,256))[60:, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black_frame = cv2.inRange(frame,(0,0,0),(180,255,150))
    kernel = np.ones((7,7),np.uint8)
    black_frame = cv2.erode(black_frame, kernel, iterations = 1)
    gray = np.float32(black_frame)
    harris_corners = cv2.cornerHarris(gray, 5, 5, 0.05)
    kernel = np.ones((5,5),np.uint8)
    harris_corners = cv2.dilate(harris_corners, kernel, iterations = 2)
    return np.count_nonzero(harris_corners)

def Visibility(obstacle_map, y_start=15, y_end=45): # visibility decides how fast car can run
    visibility = 0
    for y in range(y_start, y_end):
        spots = np.where(np.array(obstacle_map[y]) == 'R')[0]
        visibility += min(5, max(0, np.sum([(3.25 - y * 0.05) * (x * x * (125/3) + x * (-27.5) + 29/6) for x in spots])))
    if y_start == y_end - 1:
        if len(spots) == 0:
            spots = [0]
        return visibility, np.average(spots)
    return visibility

def RedGone(obstacle_map): # check occurence of red points in the frontal visual field
    for x in range(80):
        for y in range(15, 45):
            mask = (abs(x - 40) / 80) * (abs(x - 40) / 80) * (-36.6) + (abs(x - 40) / 80) * (-5.9) + 40
            if y > mask:
                continue
            else:
                if obstacle_map[y][x] == 'R':
                    return False
    return True

def FindBox(before, after): # assign an image to track optical flow
    global argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, OBSTACLE, FRAME_COUNTER, obs_history
    obstacle_map = np.array([[0.0 for x in range(80)] for y in range(45)])
    if before is None:
        return obstacle_map
    points = 2000 # [tunable parameter 2000]
    original = deepcopy(after)
    small = cv2.resize(original, (80,45))
    before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(before, points, 0.01, 10)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(before, after, corners, (30,30))
    points = len(corners)

    old = [(int(corners[p][0][0]), int(corners[p][0][1])) for p in range(points)]
    new = [(int(nextPts[p][0][0]), int(nextPts[p][0][1])) for p in range(points)]
    vector_x = [new[p][0] - old[p][0] for p in range(points)] # vector_x = new_x - old_x
    drawn = [False for p in range(points)]
    for p in range(points):
        if status[p][0] == 0:
            continue
        if ((new[p][0] - 640) / 660)**2 + ((new[p][1] - 310) / 160)**2 > 1: # masked by a top portion of an ellipse
            continue
        if (new[p][0] - old[p][0])**2 + (new[p][1] - old[p][1])**2 > 5000: # large length taken as error [tunable parameter 5000]
            continue
        if abs(new[p][1] - old[p][1]) / abs(new[p][0] - old[p][0] + 0.01) > 0.5: # large verdical flows are not regarded as objects [tunable parameter 0.5]
            continue
        #cv2.line(before, old[p], new[p], (0, 255, 0), 2)
        #cv2.line(after, old[p], new[p], (0, 255, 0), 2)
        drawn[p] = True

    mean = np.mean([abs(vector_x[p[0]]) for p in np.argwhere(drawn).tolist()])
    stdev = np.std([abs(vector_x[p[0]]) for p in np.argwhere(drawn).tolist()])
    concentration_map = np.array([[0.0 for x in range(80)] for y in range(45)])
    for p in range(points):
        if drawn[p] and abs(vector_x[p]) > 10: # notable object points [tunable parameter 10]
            value = abs(vector_x[p]) / 20 # [tunable parameter 'value']
            obstacle_map[min(44, int(new[p][1] / 16))][min(79, int(new[p][0] / 16))] += value
            obstacle_map[min(44, int(old[p][1] / 16))][min(79, int(old[p][0] / 16))] += value
            concentration_map[min(44, int(new[p][1] / 16))][min(79, int(new[p][0] / 16))] += np.power(2, (abs(vector_x[p]) - mean) / (stdev + 0.01)) # [tunable parameter 2]
            concentration_map[min(44, int(old[p][1] / 16))][min(79, int(old[p][0] / 16))] += np.power(2, (abs(vector_x[p]) - mean) / (stdev + 0.01)) # [tunable parameter 2]

    heat_y = [0 for y in range(45)]
    accum_y = 0
    max_y = 0
    temp_start_y = 0
    for y in range(45):
        heat_y[y] = 1 if np.argwhere(obstacle_map[y][:]).shape[0] >= 12 else -1  # [tunable parameter 12]
        accum_y += heat_y[y]
        if accum_y < 0:
            accum_y = 0
        if accum_y == heat_y[y]:
            temp_start_y = y
        if accum_y >= max_y:
            max_y = accum_y
            argmax_y_end = y
            argmax_y_start = temp_start_y
    heat_x = [0 for x in range(80)]
    accum_x = 0
    max_x = 0
    temp_start_x = 0
    for x in range(80):
        heat_x[x] = 1 if np.argwhere([obstacle_map[y][x] for y in range(45)]).shape[0] >= 7 else -1 # [tunable parameter 7]
        accum_x += heat_x[x]
        if accum_x < 0:
            accum_x = 0
        if accum_x == heat_x[x]:
            temp_start_x = x
        if accum_x >= max_x:
            max_x = accum_x
            argmax_x_end = x
            argmax_x_start = temp_start_x

    left_red = None
    right_red = None
    for x in range(argmax_x_start, argmax_x_end):
        blue = int(small[argmax_y_end][x][0])
        green = int(small[argmax_y_end][x][1])
        red = int(small[argmax_y_end][x][2])
        if red < 50: # [tunable parameter 50]
            continue
        if (red > green * 1.4 and red > blue * 1.4) or red > (green + blue) * 0.8: # [tunable parameters 1.4, 1.4, 0.8]
            left_red = x
            break
    for x in range(argmax_x_end, argmax_x_start, -1):
        blue = int(small[argmax_y_end][x][0])
        green = int(small[argmax_y_end][x][1])
        red = int(small[argmax_y_end][x][2])
        if red < 50: # [tunable parameter 50]
            continue
        if (red > green * 1.4 and red > blue * 1.4) or red > (green + blue) * 0.8: # [tunable parameters 1.4, 1.4, 0.8]
            right_red = x
            break
    if left_red is not None and right_red is not None:
        if left_red - argmax_x_start > argmax_x_end - right_red:
            argmax_x_end = left_red
        else:
            argmax_x_start = right_red
    #cv2.rectangle(after, (argmax_x_start * 16, argmax_y_start * 16), (argmax_x_end * 16, argmax_y_end * 16), (0, 255, 0), 3)
    #print(argmax_x_start, argmax_y_start, argmax_x_end, argmax_y_end)

    area = (argmax_x_end - argmax_x_start) * (argmax_y_end - argmax_y_start)
    #print('Area: ', area)
    if area < 100: # [tunable parameter 100]
        OBSTACLE = False
    else:
        # miss rate and capture rate tells about the false positive and false negative of accurately cropping the flow, not the real obstacle
        misses = np.argwhere([[obstacle_map[y][x] <= 1.2 for x in range(argmax_x_start, argmax_x_end)] for y in range(argmax_y_start, argmax_y_end)]).shape[0]  # [tunable parameter 1.2]
        miss_rate = round(misses / area, 3)
        #print('False positive: ', miss_rate)
        all_flow = np.sum([[concentration_map[y][x] for x in range(80)] for y in range(45)])
        cropped_flow = np.sum([[concentration_map[y][x] for x in range(argmax_x_start, argmax_x_end)] for y in range(argmax_y_start, argmax_y_end)])
        capture_rate = round(cropped_flow / all_flow, 3)
        #print('False negative: ', 1 - capture_rate)
        #cv2.rectangle(after, (0, 0), (650, 70), (0, 0, 0), -1)
        rates_text = 'False positive: ' + str(miss_rate) + '   False negative: ' + str(round(1 - capture_rate, 3))
        #cv2.putText(after, rates_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        if miss_rate > 0.45: # [tunable parameter 0.45]
            OBSTACLE = False
        elif 1 - capture_rate > 0.4: # [tunable parameter 0.4]
            OBSTACLE = False
        elif argmax_x_end - argmax_x_start < 10 or argmax_y_end - argmax_y_start < 5 or argmax_x_end - argmax_x_start > 40: # [tunable parameters 10, 5, 40]
            OBSTACLE = False
        else:
            OBSTACLE = True
            print('Obstacle detected')

    obstacle_map = obstacle_map.tolist()
    for y in range(15, 45):
        for x in range(80):
            blue = int(small[argmax_y_end][x][0])
            green = int(small[argmax_y_end][x][1])
            red = int(small[argmax_y_end][x][2])
            if red < 50: # [tunable parameter 50]
                continue
            if (red > green * 1.4 and red > blue * 1.4) or red > (green + blue) * 0.8: # [tunable parameters 1.4, 1.4, 0.8]
                obstacle_map[y][x] = 'R'
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)          
    return obstacle_map

def execute(change):
    global last_frame, prev_time, STOP_FLAG, FRAME_COUNTER, argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, GOAL, OBSTACLE, HANDLER_MODE, PARKING_MODE, visibility, motion_plan, motion_span, motion_init, obs_history, handler_history # (hosin)
    if GOAL:
        robot.stop()
        return
    curr_time = perf_counter()
    time_step = curr_time-prev_time
    FRAME_COUNTER += 1
    curr_frame = change['new']
    if FRAME_COUNTER % 2 == 0 and not PARKING_MODE: # (hosin)
        obstacle_map = FindBox(deepcopy(last_frame), deepcopy(curr_frame))
        visibility = Visibility(obstacle_map)
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV) #Use HSV instead of BGR for easier color filtration
    red_frame = cv2.inRange(hsv_frame, (160, 50, 20), (180, 255, 255))
    red_frame = red_frame + cv2.inRange(hsv_frame, (0,50,20), (10,255,255)) #Filter out non-red colors, since red~orange spans from hue 0~20 and 160~180, a two step process is needed
    black_frame = cv2.inRange(hsv_frame,(0,0,0),(180,255,150)) #Black colors, for object avoidance
    ref_row = red_frame[REFERENCE_ROW,:] #Referenced row for red line tracking
    best_len, mid = GetMedian(red_frame, REFERENCE_ROW)
    target = HALF_WIDTH
    print('\nfeature points: ', FeaturePoints(curr_frame))

    if FRAME_COUNTER % 2 == 0 and handler_history > 5 and not HANDLER_MODE and not PARKING_MODE and (FeaturePoints(curr_frame) > 4000 or RedGone(obstacle_map)):
        print('\nfeature points: ', FeaturePoints(curr_frame))
        print('enter parking area at time ' + str(FRAME_COUNTER))
        PARKING_MODE = True
    elif not HANDLER_MODE and not PARKING_MODE and OBSTACLE and best_len > 0:
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
        print(FRAME_COUNTER, '--- handler: ',  str(motion), 't = ', round(perf_counter(), 3))
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
            print('\nexit handler mode at time: ' + str(FRAME_COUNTER))
    else:
        if PARKING_MODE:
            feature_points = FeaturePoints(curr_frame)
            print('\nfeature points: ', feature_points)
            obstacle_map = FindBox(deepcopy(last_frame), deepcopy(curr_frame))
            mid_diff = 'none'
            for y in range(30, 45):
                visibility, center = Visibility(obstacle_map, y, y + 1)
                if visibility > 0:
                    x = abs(center - 40) / 80
                    robot.forward(max(0, 0.02 - x * 0.2))
                    inv_slope = (center - Visibility(obstacle_map, y, y + 1)[1]) / (44.1 - y)
                    mid_diff = min(0.1, max(0.01, x / 2)) # rotate to make straight
                    break
            if feature_points > 7000:
                robot.stop()
            if mid_diff == 'none':
                GOAL = True
                robot.forward(0.5) # final rush
                print('visibility: ', Visibility(obstacle_map))
                print('--- stop ---')
                return
            if inv_slope > 0.2: 
                robot.add_motor(0.05*mid_diff,-0.05*mid_diff)
            elif inv_slope < -0.2:
                robot.add_motor(-0.05*mid_diff,0.05*mid_diff)
            elif center * 16 > HALF_WIDTH + 30: #Proportion controller
                robot.add_motor(0.05*mid_diff,-0.05*mid_diff)
            elif center * 16 < HALF_WIDTH - 30:
                robot.add_motor(-0.05*mid_diff,0.05*mid_diff)
            else:
                robot.set_motor(0.01,0.01)
            print('parking mode at time: ', str(FRAME_COUNTER))
            print(round(mid_diff, 6), round(visibility, 2), round(center * 16, 0), x)
        else:
            mid_diff = min(0.05, abs(mid - HALF_WIDTH) / WIDTH / 10)
            print('guiding mode', round(mid_diff, 6), round(visibility, 2))
            robot.forward(min(0.05, max(visibility * 0.0005 - 0.025, visibility / 600 - 0.0667)))
            if mid > target + 20: #Proportion controller
                robot.add_motor(0.025*mid_diff,-0.025*mid_diff)
            elif mid < target - 20:
                robot.add_motor(-0.025*mid_diff,0.025*mid_diff)

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
    last_frame = curr_frame

robot = Robot()
camera = Camera()
camera.observe(execute)
