import numpy as np

list = []

with open('../some.txt') as f:
    for l in f:
        terms = l.split(" ")
        distance = float(str.rstrip(terms[len(terms) - 1]))
        frame = int(terms[len(terms) - 5])
        list.append([frame, distance])

dict = {}

with open('../VDataKamL.txt') as f:
    for i, string in enumerate(f):
        terms = string.split(" ")
        distance_term = str.rstrip(terms[len(terms) - 1])
        distance = float(distance_term)
        dict[i + 1] = distance

list2 = []

for el in list:
    frame_number = el[0]
    measured_distance = el[1]
    real_distance = dict.get(frame_number)
    if real_distance is not None:
        list2.append([frame_number, real_distance])
        measurement_error = measured_distance - real_distance
        # print(measurement_error)

import matplotlib.pyplot as plt

measures = np.array(list).T
reals = np.array(list2).T

plt.scatter(measures[0], measures[1], s=1, c='blue')
plt.scatter(reals[0], reals[1], s=2, color='red')

plt.show()

