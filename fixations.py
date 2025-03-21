import numpy as np
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt

pupil_pd_frame = pd.read_csv('/Users/ioanamunteanu/Downloads/EyeData/pupil_data/pupil_positions1.csv')

# filter for 3d data
detector_3d_data = pupil_pd_frame[pupil_pd_frame.method == '3d c++']


# Normalize timestamps to start at 0
start_time = detector_3d_data['world_timestamp'].min()
print(start_time)
detector_3d_data['normalized_timestamp'] = detector_3d_data['world_timestamp'] - start_time

def vector_dispersion(vectors):
    distances = scipy.spatial.distance.pdist(vectors, metric='cosine')
    distances.sort()
    cut_off = np.max([distances.shape[0] // 5, 4])
    return np.arccos(1. - distances[-cut_off:].mean())

max_dispersion = np.deg2rad(1)
min_duration = 0.3

def gaze_dispersion(eye_data):
    eye0_data = [p for p in eye_data if p['eye_id'] == 0 and p['confidence'] > 0.95]
    eye1_data = [p for p in eye_data if p['eye_id'] == 1 and p['confidence'] > 0.95]
    base_data = eye1_data if len(eye1_data) > len(eye0_data) else eye0_data

    vectors = []
    for p in base_data:
        vectors.append((p['circle_3d_normal_x'], p['circle_3d_normal_y'], p['circle_3d_normal_z']))
    vectors = np.array(vectors, dtype=np.float32)

    if len(vectors) < 2:
        return float("inf")
    else:
        return vector_dispersion(vectors)

from collections import deque

def detect_fixations(gaze_data):
    # Convert Pandas data frame to list of Python dictionaries
    gaze_data = gaze_data.T.to_dict().values()

    candidate = deque()
    future_data = deque(gaze_data)
    while future_data:
        # print('future_data', future_data)
        # print('candidate', candidate)
        # check if candidate contains enough data
        if len(candidate) < 2 or candidate[-1]['normalized_timestamp'] - candidate[0]['normalized_timestamp'] < min_duration:
            datum = future_data.popleft()
            candidate.append(datum)
            continue

        # Minimal duration reached, check for fixation
        dispersion = gaze_dispersion(candidate)
        if dispersion > max_dispersion:
            # not a fixation, move forward
            candidate.popleft()
            continue

        # Minimal fixation found. Try to extend!
        while future_data:
            datum = future_data[0]
            candidate.append(datum)

            dispersion = gaze_dispersion(candidate)
            if dispersion > max_dispersion:
                # end of fixation found
                candidate.pop()
                break
            else:
                # still a fixation, continue extending
                future_data.popleft()
                
        yield (candidate[0]['normalized_timestamp'], candidate[-1]['normalized_timestamp'])
        candidate.clear()

fixations = list(detect_fixations(detector_3d_data))
fixation_durations = []

plt.rcParams['figure.figsize'] = (25, 6)
for fix in fixations:
    print('fix', fix)
    length = fix[1] - fix[0]
    fixation_durations.append(length)
    plt.bar(fix[0], 1.0, length, align='edge')
plt.xlabel("Timestamps [s]")
plt.ylim((0,1))
plt.gca().get_yaxis()
plt.title("Occurrences of Fixations")
plt.show()


# Number of fixations
num_fixations = len(fixations)

# Average fixation duration
avg_fixation_duration = np.mean(fixation_durations)

# Total time spent in fixations
total_fixation_time = sum(fixation_durations)

# Total recording time (from first to last timestamp)
total_time = detector_3d_data['normalized_timestamp'].max()

# Percentage of time spent in fixations
percent_time_fixation = (total_fixation_time / total_time) * 100 if total_time > 0 else 0

print(f"Number of Fixations: {num_fixations}")
print(f"Average Fixation Duration: {avg_fixation_duration:.2f} sec")
print(f"Percent Time in Fixation: {percent_time_fixation:.2f}%")