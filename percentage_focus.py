import numpy as np
import os
import math
import scipy
import pandas as pd
from datascience import *
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



def getVelocities(frame):

    x_pos = frame["norm_pos_x"].values
    y_pos = frame["norm_pos_y"].values
    t = frame["world_timestamp"].values
    length = len(frame["eye_id"])
    velocities = []
    for i in range(0, length-1):
        x_dif = x_pos[i+1] - x_pos[i]
        y_dif = y_pos[i+1] - y_pos[i]
        dt = t[i+1] - t[i]
        pos_change = np.sqrt(pow(x_dif, 2) + pow(y_dif, 2))
        velocity = pos_change/dt
        velocities.append(velocity)
    velocities.append(0)
    return velocities

def getTime(minutes, sec, table):
    time = (60 * minutes) + sec
    timestamp = table.column("world_timestamp").item(0) + time
    return timestamp

def percentFocused(time, table, min1, sec1, min2, sec2):
    time_focused = 0
    time_not_focused = 0
    
    start_time = getTime(min1, sec1, table)
    end_time = getTime(min2, sec2, table)
    middle_section = table.where(time, are.above(start_time))
    middle_section = middle_section.where(time, are.below(end_time))
    all_velocities = len(middle_section.column("Velocity"))
    
    #Divide into 100 millisecond portions
    start = middle_section.column(time).item(0)
    final = middle_section.column(time).item(all_velocities-1)
    intervals = (final - start)/.1

    while start < final:
        end = start + .1
        interval = middle_section.where(time, are.above(start)).where(time, are.below(end))
#       print(interval)
        if len(interval.column("Velocity")) > 0:
#           print(interval.column("Velocity"))
            interval_max = max(interval.column("Velocity"))
#           print(interval_max)
            # Check if the max is below the threshold
            if interval_max < 2:
                time_focused += 1
#               print("time focused: ", time_focused)
            else:
                time_not_focused += 1
#               print("time not focused: ", time_not_focused)
        start = end
    percentage = (time_focused / (time_focused + time_not_focused))
    print(percentage)
    return percentage


for i in range (1, 30):
     test_data = pd.read_csv("/Users/ioanamunteanu/Eye-Tracking/pupil_data/pupil_positions (" + str(i) + ").csv",
                      usecols=["world_timestamp", "world_index", "eye_id", "confidence", "norm_pos_x", "norm_pos_y"])

     test_data = test_data.loc[test_data["confidence"] >= .7].loc[test_data["eye_id"] == 0]
     test_v = getVelocities(test_data)
     v = {"Velocities": test_v}
     test_velocities = pd.DataFrame(data = v)
     velocities_table = Table.from_df(test_velocities)
     test_table = Table.from_df(test_data)
     test_table.append_column("Velocity", velocities_table.column("Velocities"))     

     percent = percentFocused("world_timestamp", test_table, 8,0,15,0)


