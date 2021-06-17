from datascience import *
import pandas as pd
import matplotlib
matplotlib.use ('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import numpy as np
from astropy.table import QTable, Table, Column
from astropy import units as u
from operator import itemgetter
import scipy.fftpack



"input: pandas dataframe"
"output: time starts at 0"
def normalize_time (frame):
    t = frame["world_timestamp"].values
    length = len(frame["world_timestamp"])
    k = t[0]
    frame.world_timestamp = frame.world_timestamp - k

"input: pandas dataframe"
"output: plots x and y coordinates"
def plotCoordinates (frame):
    # right eye is 0 and left is 1
    pupil_right = frame.loc[frame["confidence"] >= .7].loc[frame["eye_id"] == 0]
    pupil_left = frame.loc[frame["confidence"] >= .7].loc[frame["eye_id"] == 1]

    x_r = make_array(pupil_right["norm_pos_x"])
    # y_r = make_array(pupil_right["norm_pos_y"])
    # x_l = make_array(pupil_left["norm_pos_x"])
    # y_l = make_array(pupil_left["norm_pos_y"])

    t = make_array(pupil_right["world_timestamp"])
    #print (t)

    #fig, axs = plt.subplots(1, 1,  figsize=(20, 10), sharey=True)
    # plt.ylim((0, 1))
    plt.scatter(t, x_r, marker = '.')
    # axs[1].scatter(x_r, t, marker = '.')
    # axs[2].scatter(y_l, t, marker = '.')
    # axs[3].scatter (x_l, t, marker = '.')
    plt.xlabel('time (s)')
    plt.ylabel('x position (pixel)')
    
    plt.suptitle('Non-Focused Pupil Position Plot')
    # plt.ylim((0, 5))
    plt.xlim((0, 500))
    plt.show()

"input: pandas dataframe"
"output: a matrix of all the velocities and times"
def getVelocitiesTimes (frame):
    x_pos = frame["norm_pos_x"].values
    y_pos = frame["norm_pos_y"].values
    t = frame["world_timestamp"].values
    length = len(frame["eye_id"])
    velocities = []
    avg_times = []
    for i in range(0, length - 1):
        x_dif = x_pos[i+1] - x_pos[i]
        y_dif = y_pos[i+1] - y_pos[i]
        dt = t[i+1] - t[i]
        avg_time = (t[i+1] + t[i])/2
        avg_times.append (avg_time)
        pos_change = np.sqrt(pow(x_dif, 2) + pow(y_dif, 2))
        velocity = pos_change/dt
        velocities.append(velocity)
    return [velocities, avg_times]

def getAcclerationsTimes (frame):
    velocities = getVelocitiesTimes(frame)[0]
    times = getVelocitiesTimes(frame)[1]
    length = len(velocities)
    avg_times = []
    accelerations = []
    for i in range(0, length - 1):
        dv = velocities[i+1] - velocities[i]
        dt = times[i+1] - times[i]
        avg_time = (times[i+1] + times[i])/2
        avg_times.append (avg_time)
        acceleration = abs(dv/dt)
        accelerations.append(acceleration)
    return [accelerations, avg_times]

"input: pandas dataframe"
"output: plots velocity (pixels/s) over time (s)"
def plotVelocities (frame):
    velocities = getVelocitiesTimes(frame)[0]
    avg_times = getVelocitiesTimes(frame)[1]
    plt.scatter (avg_times, velocities)
    plt.xlabel('time (s)')
    plt.ylabel('velocity (pixel/s)')
    plt.suptitle('Velocity over Time Plot')
    #plt.ylim((0, 5))
    #plt.xlim((200, 1000))
    plt.show()

def plotAccelerations (frame):
    # plt.ylim((50, 600))
    # plt.xlim((200, 1000))
    velocities = getAcclerationsTimes(frame)[0]
    avg_times = getAcclerationsTimes(frame)[1]
    plt.scatter (avg_times, velocities)
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (pixel/s^2)')
    plt.suptitle('Acceleration over Time Plot')
    #plt.ylim((0, 5))
    plt.show()

"input: pandas dataframe, interval expressed in seconds"
"output: percentage focused"
def percentFocused(frame, t1, t2, v):
    "For intervals of 100 ms, determines if the interval is a period of focus or not"
    "If max velocity is above 2 pixels/sec, it is unfocused."
    data = frame

    velocities = getVelocitiesTimes(data)[0]
    velocities.append (0)
    data["Velocity"] = velocities

    time_focused = time_not_focused = 0
    start = t1
    final = t2

    while start < final:
        end = start + .1
        interval = data.loc[data["world_timestamp"] >= start].loc[data["world_timestamp"] <= end]
        if len(interval["eye_id"]) > 0:
            interval_max = interval["Velocity"].max()
            # print (interval)
            # print ("*******************")
            # print (interval_max)
            # Check if the max is below the threshold
            if interval_max < v:
                time_focused += 1
            else:
                time_not_focused += 1
        start = end
    percentage = (time_focused / (time_focused + time_not_focused))
    return percentage

def percentFocusedBoth(frame_l, frame_r, t1, t2):
    "For intervals of 100 ms, determines if the interval is a period of focus or not"
    "If max velocity is above 2 pixels/sec, it is unfocused."
    data_l = frame_l
    data_r = frame_r

    velocities_l = getVelocitiesTimes(data_l)[0]
    velocities_l.append (0)
    data_l["Velocity"] = velocities_l

    velocities_r = getVelocitiesTimes(data_r)[0]
    velocities_r.append (0)
    data_r["Velocity"] = velocities_r

    time_focused = time_not_focused = 0
    start = t1
    final = t2

    while start < final:
        end = start + .1
        interval_l = data_l.loc[data_l["world_timestamp"] >= start].loc[data_l["world_timestamp"] <= end]
        interval_r = data_r.loc[data_r["world_timestamp"] >= start].loc[data_r["world_timestamp"] <= end]
        if len(interval_l["eye_id"]) > 0:
            interval_max_l = interval_l["Velocity"].max()
            interval_max_r = interval_r["Velocity"].max()
            # print (interval)
            # print ("*******************")
            # print (interval_max)
            # Check if the max is below the threshold
            if interval_max_l < 2 and interval_max_r < 2:
                time_focused += 1
            else:
                time_not_focused += 1
        start = end
    percentage = (time_focused / (time_focused + time_not_focused))
    return percentage

def allFocus (v):
    print(v)
    for i in range (17):
        pupils = pd.read_csv("/Users/munte029/Desktop/eye_tracking/data" + str(i) + ".csv",
                usecols=["world_timestamp","eye_id", "norm_pos_x", "confidence", "norm_pos_y"])
        normalize_time (pupils)

        pupils_l = pupils.loc[pupils["confidence"] >= 0.7].loc[pupils["eye_id"] == 0]
        velocities_l = getVelocitiesTimes(pupils_l)[0]
        velocities_l.append (0)
        pupils_l["Velocity"] = velocities_l

        pupils_r = pupils.loc[pupils["confidence"] >= 0.7].loc[pupils["eye_id"] == 1]
        velocities_r = getVelocitiesTimes(pupils_r)[0]
        velocities_r.append (0)
        pupils_r["Velocity"] = velocities_r
        
        #pupils = pupils.loc[pupils["confidence"] >= 0.7].loc[pupils["Velocity"] <= 7000].loc[pupils["eye_id"] == 0]
        
        residency = [0,0,1,1,1,2,2,5,5,5,6,6,6,7,7,7,7]

        #focused = percentFocusedBoth (pupils_l, pupils_r, 480, 900)
        focused_l = percentFocused (pupils_l,  480, 900, v)
        focused_r = percentFocused (pupils_r,  480, 900, v)
        #print ("Residency year: " + str(residency[i]) + ". Percentage focused: " + str(focused))
        Max = max(focused_l, focused_r)
        print(Max)
        #print ("Sample " + str(i) + ". Residency year: " + str(residency[i]) + ". Left: " + str(focused_l) + ". Right: " + str(focused_r) + ". Average: " + str((focused_l + focused_r)/2))

def AccFocused(frame, t1, t2, a):
    "For intervals of 100 ms, determines if the interval is a period of focus or not"
    "If max velocity is above 2 pixels/sec, it is unfocused."
    data = frame

    accelerations = getAcclerationsTimes(data)[0]
    accelerations.append (0)
    accelerations.append (0)
    # print(len(data))
    # print(len(accelerations))
    data["Acc"] = accelerations

    time_focused = time_not_focused = 0
    start = t1
    final = t2

    while start < final:
        end = start + .1
        interval = data.loc[data["world_timestamp"] >= start].loc[data["world_timestamp"] <= end]
        if len(interval["eye_id"]) > 0:
            interval_max = interval["Acc"].max()
            # Check if the max is below the threshold
            if interval_max < a:
                time_focused += 1
            else:
                time_not_focused += 1
        start = end
    percentage = (time_focused / (time_focused + time_not_focused))
    return percentage

def allAccFocus (v):
    print(v)
    for i in range (17):
        pupils = pd.read_csv("/Users/munte029/Desktop/eye_tracking/data" + str(i) + ".csv",
                usecols=["world_timestamp","eye_id", "norm_pos_x", "confidence", "norm_pos_y"])
        normalize_time (pupils)

        pupils_l = pupils.loc[pupils["confidence"] >= 0.7].loc[pupils["eye_id"] == 0]
        acc_l = getAcclerationsTimes(pupils_l)[0]
        acc_l.append (0)
        acc_l.append (0)
        # print( len(pupils_l))
        # print( len(acc_l))
        pupils_l["Acc"] = acc_l

        pupils_r = pupils.loc[pupils["confidence"] >= 0.7].loc[pupils["eye_id"] == 1]
        acc_r = getAcclerationsTimes(pupils_r)[0]
        acc_r.append (0)
        acc_r.append (0)
        pupils_r["Acc"] = acc_r
        
        
        residency = [0,0,1,1,1,2,2,5,5,5,6,6,6,7,7,7,7]

        #focused = percentFocusedBoth (pupils_l, pupils_r, 480, 900)
        focused_l = AccFocused (pupils_l,  480, 900, v)
        focused_r = AccFocused (pupils_r,  480, 900, v)
        #print ("Residency year: " + str(residency[i]) + ". Percentage focused: " + str(focused))
        Max = max(focused_l, focused_r)
        print(Max)
        #print ("Sample " + str(i) + ". Residency year: " + str(residency[i]) + ". Left: " + str(focused_l) + ". Right: " + str(focused_r) + ". Average: " + str((focused_l + focused_r)/2))

def fft_velocity(frame):
    
    normalize_time (frame)
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / len(frame["eye_id"])
    yf = scipy.fftpack.fft(getVelocitiesTimes(frame)[0])
    xf = np.asarray(getVelocitiesTimes(frame)[1])
    print(yf.size)
    print(xf.size)
    z = 2.0/N * np.abs(yf[:N//2])
    print(z.size)
    fig, ax = plt.subplots()

    #ax.plot(x,y) #plot raw signal
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))  #plot fft
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (pixel)')
    #ax.set_yscale('log')
    plt.show()

def fft_acceleration(frame):
    
    normalize_time (frame)
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / len(frame["eye_id"])
    yf = scipy.fftpack.fft(getAcclerationsTimes(frame)[0])
    xf = np.asarray(getAcclerationsTimes(frame)[1])
    print(yf.size)
    print(xf.size)
    z = 2.0/N * np.abs(yf[:N//2])
    print(z.size)
    fig, ax = plt.subplots()

    #ax.plot(x,y) #plot raw signal
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))  #plot fft
    #ax.set_yscale('log')
    #plt.xlim((0,10))
    plt.show()

pupils = pd.read_csv("/Users/munte029/Desktop/eye_tracking/data10.csv",
                usecols=["world_timestamp","world_index","eye_id", "norm_pos_x", "confidence", "norm_pos_y"])
pupils = pupils.loc[pupils["confidence"] >= .6].loc[pupils["eye_id"] ==1]
normalize_time(pupils)
plotCoordinates (pupils)
pupilsscrubin = pupils.iloc[:166]
pupilssurgery = pupils.iloc[1200:1366]
fft_velocity(pupilsscrubin)
fft_velocity(pupilssurgery)
# fft_acceleration(pupilsscrubin)
# fft_acceleration(pupilssurgery)


