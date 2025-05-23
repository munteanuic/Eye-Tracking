import numpy as np
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import deque
from scipy.spatial.distance import pdist

map_file_to_surgery_time = {
    1: [
        (0, 46),
        (435, 450),
        (658, 668),
        (690, 696),
        (719, 723),
        (744, 757),
        (827, 829),
        (835, 839),
        (843, 845),
        (1129, 1135),
        (1250, 1253),
        (1284, 1288),
        (1296, 1305),
        (1357, 1361),
        (1478, 1498),
        (1562, 1564),
        (1568, 1570),
        (1592, 1597),
        (1831, 1834),
        (1843, 1847),
        (1857, 1860),
        (1920, 1925),
        (1993, 1996),
        (2012, 2015),
        (2049, 2061),
        (2093, 2098),
        (2119, 2122),
        (2151, 2158),
        (2247, 2250),
        (2295, 2300)
    ], 2: [
        (5, 9), (55, 57), (94, 102), (118, 120), (124, 128),
        (137, 140), (141, 144), (150, 152), (162, 164), (175, 179),
        (190, 192), (470, 479), (486, 494), (588, 590), (919, 921),
        (1000, 1004), (1099, 1105), (1255, 1256), (1276, 1277),
        (1335, 1336), (1508, 1545), (1550, 1551), (1576, 1580),
        (1958, 1958), (1990, 1994), (2050, 2052), (2093, 2094),
        (2224, 2226), (2268, 2370), (2623, 2625), (2646, 2647),
        (2691, 2696), (2705, 2708), (2745, 2725), (2736, 2737),
        (2790, 2798), (2847, 2848), (3250, 3252), (3317, 3322),
        (3336, 3340), (3494, 3500), (3544, 3547), (3632, 3638),
        (3703, 3707), (3718, 3722), (3732, 3747), (3798, 3802),
        (3864, 3872), (3882, 3884), (4057, 4069), (4164, 4170),
        (4552, 4556), (4756, 4759), (4832, 4832), (4893, 4896),
        (4924, 4944), (5065, 5068), (5160, 5172), (5179, 5199),
        (5223, 5230), (5302, 5306), (5315, 5319), (5380, 5404),
        (5409, 5416)
    ], 9: [
        (170, 175), (291, 297), (317, 321), (530, 532), (579, 581),
        (663, 672), (689, 692), (762, 767), (784, 788), (797, 799),
        (829, 832), (869, 874), (876, 877), (880, 882), (927, 929),
        (990, 1000), (1341, 1361), (1388, 1404), (1427, 1428), (1733, 1736),
        (1742, 1744), (1763, 1764), (1775, 1778), (1934, 1935), (1950, 1952),
        (1966, 1967), (2329, 2331), (2397, 2399), (2412, 2418), (2422, 2425),
        (2426, 2428), (2570, 2572), (2606, 2614)
    ], 10: [
        (2, 3), (13, 21), (55, 58), (97, 100), (128, 131),
        (186, 189), (194, 196), (296, 298), (299, 300), (307, 311),
        (313, 314), (338, 340), (364, 367), (400, 401), (429, 433),
        (448, 449), (531, 536), (572, 574), (597, 600), (654, 655),
        (668, 676), (705, 709), (717, 722), (749, 755), (822, 825),
        (863, 867), (907, 909), (929, 931), (941, 944), (960, 967),
        (971, 987), (1026, 1028), (1155, 1159), (1220, 1236), (1268, 1269),
        (1328, 1332), (1522, 1523), (1561, 1564), (1583, 1591), (1617, 1623),
        (1639, 1644), (1668, 1668), (1676, 1682), (1686, 1690), (1728, 1729),
        (1837, 1838), (1869, 1872), (1879, 1887)
    ], 14: [
        (14, 24), (28, 29), (34, 35), (40, 42), (76, 110),
        (180, 181), (188, 191), (199, 201), (208, 211), (234, 240),
        (254, 255), (295, 298), (345, 351), (356, 363), (430, 437),
        (461, 469), (473, 474), (481, 485)
    ], 17: [
        (15, 20), (66, 71), (89, 92), (101, 102), (173, 178),
        (212, 214), (243, 248), (255, 259), (332, 335), (341, 343),
        (348, 350), (367, 368), (421, 424), (443, 445), (598, 601),
        (609, 610), (652, 659), (668, 674), (681, 683), (711, 712),
        (713, 715), (722, 724), (730, 732), (744, 746)
    ], 22: [
        (183, 191), (191, 198), (217, 354), (419, 440), (463, 493),
        (503, 512), (544, 568), (609, 612), (654, 684), (697, 701),
        (709, 733), (763, 770), (775, 782), (792, 841), (848, 883),
        (856, 915), (968, 969), (991, 997), (1005, 1015), (1057, 1058),
        (1064, 1082)
    ], 25: [
        (65, 71), (76, 97), (113, 118), (137, 145), (169, 171),
        (180, 184), (349, 352), (355, 358), (360, 366), (401, 403),
        (416, 422), (904, 956), (964, 981), (1024, 1058), (1142, 1143)
    ], 26: [
        (15, 17), (80, 81), (149, 153), (772, 773), (848, 859), (917, 918)
    ],    27: [
        (143, 144), (257, 259), (263, 268), (325, 328), (332, 334),
        (338, 347), (351, 352), (358, 361), (368, 373), (381, 384)
    ],  28: [
        (17, 19), (24, 26), (37, 38), (448, 458), (482, 486),
        (587, 588), (703, 705), (722, 725), (892, 894), (904, 906),
        (1012, 1015)
    ], 35: [
        (0, 68), (178, 187), (504, 525), (585, 594)
    ],  37: [
        (59, 61), (520, 524), (578, 579), (586, 588), (594, 595),
        (621, 622), (632, 638), (923, 938), (977, 985), (1027, 1032),
        (1075, 1079), (1120, 1121), (1166, 1168), (1197, 1198),
        (1242, 1250), (1251, 1253)
    ], 
}

map_file_to_surgeon_and_level = {
    1: {"surgeon": "Jen", "level": 6},
    2: {"surgeon": "SLP", "level": 7},
    9: {"surgeon": "Jen", "level": 6},
    10: {"surgeon": "SLP", "level": 7},
    14: {"surgeon": "Jen", "level": 6},
    17: {"surgeon": "Jen", "level": 6},
    22: {"surgeon": "Bin", "level": 5},
    25: {"surgeon": "Margaret", "level": 1},
    26: {"surgeon": "Margaret", "level": 1},
    27: {"surgeon": "Doug", "level": 2},
    28: {"surgeon": "Doug", "level": 2},
    35: {"surgeon": "Bin", "level": 5},
    37: {"surgeon": "Bin", "level": 5}
}


# Your function definitions
def vector_dispersion(vectors):
    distances = pdist(vectors, metric='cosine')
    distances.sort()
    cut_off = np.max([distances.shape[0] // 5, 4])
    return np.arccos(1. - distances[-cut_off:].mean())

max_dispersion = np.deg2rad(1)
min_duration = 0.3

def gaze_dispersion(eye_data):
    eye0_data = [p for p in eye_data if p['eye_id'] == 0 and p['confidence'] > 0.95]
    eye1_data = [p for p in eye_data if p['eye_id'] == 1 and p['confidence'] > 0.95]
    base_data = eye1_data if len(eye1_data) > len(eye0_data) else eye0_data

    vectors = [(p['circle_3d_normal_x'], p['circle_3d_normal_y'], p['circle_3d_normal_z']) for p in base_data]
    vectors = np.array(vectors, dtype=np.float32)

    if len(vectors) < 2:
        return float("inf")
    return vector_dispersion(vectors)

def detect_fixations(gaze_data):
    gaze_data = gaze_data.T.to_dict().values()
    candidate = deque()
    future_data = deque(gaze_data)
    while future_data:
        if len(candidate) < 2 or candidate[-1]['normalized_timestamp'] - candidate[0]['normalized_timestamp'] < min_duration:
            candidate.append(future_data.popleft())
            continue

        if gaze_dispersion(candidate) > max_dispersion:
            candidate.popleft()
            continue

        while future_data:
            datum = future_data[0]
            candidate.append(datum)
            if gaze_dispersion(candidate) > max_dispersion:
                candidate.pop()
                break
            future_data.popleft()

        yield (candidate[0]['normalized_timestamp'], candidate[-1]['normalized_timestamp'])
        candidate.clear()

def compute_fixation_centers(fixations, eye_data):
    centers = []
    for start, end in fixations:
        points = eye_data[(eye_data['normalized_timestamp'] >= start) & (eye_data['normalized_timestamp'] <= end)]
        if not points.empty:
            avg_x = points['circle_3d_normal_x'].mean()
            avg_y = points['circle_3d_normal_y'].mean()
            avg_time = (start + end) / 2
            centers.append((avg_x, avg_y, avg_time))
    return centers

# updated code to measure average velocity from after the first fixation ends and before the second fixation starts
def compute_avg_velocity(fixations, fixation_centers):
    if len(fixations) < 2:
        return 0

    velocities = []
    for i in range(len(fixations) - 1):
        _, end1 = fixations[i]
        start2, _ = fixations[i + 1]
        x1, y1, _ = fixation_centers[i]
        x2, y2, _ = fixation_centers[i + 1]

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        saccade_time = start2 - end1

        if saccade_time > 0:
            velocities.append(distance / saccade_time)

    return np.mean(velocities) if velocities else 0

# Your mappings (ensure you have map_file_to_surgery_time and map_file_to_surgeon_and_level defined)
# Insert them here

# Define scrub-in as all data before the first surgery start
scrubin_times = {}
for pid, ranges in map_file_to_surgery_time.items():
    first_surgery_start = min(start for start, _ in ranges)
    scrubin_times[pid] = [(0, first_surgery_start)]

# Loop through participants
results = []
for i in map_file_to_surgeon_and_level.keys():
    file_path = f"/Users/ioanamunteanu/Downloads/EyeData/pupil_data/pupil_positions{i}.csv"
    if not os.path.exists(file_path):
        print(f"Missing file for participant {i}")
        continue

    df = pd.read_csv(file_path)
    if 'method' not in df or '3d c++' not in df['method'].unique():
        print(f"3d data missing in file {i}")
        continue

    df = df[df.method == '3d c++']
    start_ts = df['world_timestamp'].min()
    df['normalized_timestamp'] = df['world_timestamp'] - start_ts

    for phase, time_ranges in [("scrub", scrubin_times[i]), ("surgery", map_file_to_surgery_time[i])]:
        segment_data = pd.concat([
            df[(df['normalized_timestamp'] >= start) & (df['normalized_timestamp'] <= end)]
            for start, end in time_ranges
        ])
        if segment_data.empty:
            print(f"No data for {phase} phase in participant {i}")
            continue

        fixations = list(detect_fixations(segment_data))
        durations = [end - start for start, end in fixations]
        centers = compute_fixation_centers(fixations, segment_data)

        results.append({
            "participant": i,
            "phase": phase,
            "avg_fix_dur": np.mean(durations) if durations else 0,
            "avg_velocity": compute_avg_velocity(fixations, centers),
            "experience": map_file_to_surgeon_and_level[i]["level"]
        })

# Save or print results
results_df = pd.DataFrame(results)
results_df.to_csv("scrub_vs_surgery_metrics.csv", index=False)
print("Done. Saved to scrub_vs_surgery_metrics.csv")

