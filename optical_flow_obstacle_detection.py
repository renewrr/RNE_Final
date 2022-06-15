import cv2
import numpy as np
import math

directory = 'pass_left/'

def OpticalFlow(i): # assign an image to track optical flow
    global argmax_x_start, argmax_x_end, argmax_y_start, argmax_y_end, OBSTACLE
    points = 2000 # [tunable parameter 2000]
    before = cv2.imread(directory + 'object' + str(i) + '.png')
    after = cv2.imread(directory + 'object' + str(i+1) + '.png')
    original = cv2.imread(directory + 'object' + str(i+1) + '.png')
    small = cv2.resize(original, (80,45))
    before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(before, points, 0.01, 10)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(before, after, corners, (30,30))

    old = [(int(corners[p][0][0]), int(corners[p][0][1])) for p in range(points)]
    new = [(int(nextPts[p][0][0]), int(nextPts[p][0][1])) for p in range(points)]
    vector_x = [new[p][0] - old[p][0] for p in range(points)] # vector_x = new_x - old_x
    obstacle_map = np.array([[0.0 for x in range(80)] for y in range(45)])
    drawn = [False for p in range(points)]
    #print(corners)
    #print(nextPts)
    #print(status)
    #print(err)
    for p in range(points):
        if status[p][0] == 0:
            continue
        if ((new[p][0] - 640) / 660)**2 + ((new[p][1] - 310) / 160)**2 > 1: # masked by a top portion of an ellipse
            continue
        if (new[p][0] - old[p][0])**2 + (new[p][1] - old[p][1])**2 > 5000: # large length taken as error [tunable parameter 5000]
            continue
        if abs(new[p][1] - old[p][1]) / abs(new[p][0] - old[p][0] + 0.01) > 0.5: # large verdical flows are not regarded as objects [tunable parameter 0.5]
            continue
        cv2.line(before, old[p], new[p], (0, 255, 0), 2)
        cv2.line(after, old[p], new[p], (0, 255, 0), 2)
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
        #print(y, np.argwhere(obstacle_map[y][:]).shape[0])
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
        #print(x, np.argwhere([obstacle_map[y][x] for y in range(45)]).shape[0])
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
            print(small[argmax_y_end][x])
            break
    for x in range(argmax_x_end, argmax_x_start, -1):
        blue = int(small[argmax_y_end][x][0])
        green = int(small[argmax_y_end][x][1])
        red = int(small[argmax_y_end][x][2])
        if red < 50: # [tunable parameter 50]
            continue
        if (red > green * 1.4 and red > blue * 1.4) or red > (green + blue) * 0.8: # [tunable parameters 1.4, 1.4, 0.8]
            right_red = x
            print(small[argmax_y_end][x])
            break
    if left_red is not None and right_red is not None:
        if left_red - argmax_x_start > argmax_x_end - right_red:
            argmax_x_end = left_red
        else:
            argmax_x_start = right_red
    cv2.rectangle(after, (argmax_x_start * 16, argmax_y_start * 16), (argmax_x_end * 16, argmax_y_end * 16), (0, 255, 0), 3)
    print(argmax_x_start, argmax_y_start, argmax_x_end, argmax_y_end)

    area = (argmax_x_end - argmax_x_start) * (argmax_y_end - argmax_y_start)
    print('Area: ', area)
    if area < 100: # [tunable parameter 100]
        OBSTACLE = False
    else:
        # miss rate and capture rate tells about the false positive and false negative of accurately cropping the flow, not the real obstacle
        misses = np.argwhere([[obstacle_map[y][x] <= 1.2 for x in range(argmax_x_start, argmax_x_end)] for y in range(argmax_y_start, argmax_y_end)]).shape[0]  # [tunable parameter 1.2]
        miss_rate = round(misses / area, 3)
        print('False positive: ', miss_rate)
        all_flow = np.sum([[concentration_map[y][x] for x in range(80)] for y in range(45)])
        cropped_flow = np.sum([[concentration_map[y][x] for x in range(argmax_x_start, argmax_x_end)] for y in range(argmax_y_start, argmax_y_end)])
        capture_rate = round(cropped_flow / all_flow, 3)
        print('False negative: ', 1 - capture_rate)
        cv2.rectangle(after, (0, 0), (650, 70), (0, 0, 0), -1)
        rates_text = 'False positive: ' + str(miss_rate) + '   False negative: ' + str(round(1 - capture_rate, 3))
        cv2.putText(after, rates_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        if miss_rate > 0.45: # [tunable parameter 0.45]
            OBSTACLE = False
        elif 1 - capture_rate > 0.4: # [tunable parameter 0.4]
            OBSTACLE = False
        elif argmax_x_end - argmax_x_start < 10 or argmax_y_end - argmax_y_start < 5 or argmax_x_end - argmax_x_start > 40: # [tunable parameters 10, 5, 40]
            OBSTACLE = False
        else:
            OBSTACLE = True
            print('Obstacle detected')
            
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    cv2.imwrite(directory + 'flow' + str(i) + '.png', after)
    return obstacle_map


for i in range(20):
    OpticalFlow(i+1)
