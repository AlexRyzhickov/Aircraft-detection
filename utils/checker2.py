import numpy as np
import re
import math

def str2(str, s):
    isNotFirst = False
    for i in range(len(str)):
        if str[i] == s:
            if isNotFirst:
                return i
            if not isNotFirst:
                isNotFirst = True
    return -1


def axis_name(coordinate):
    if coordinate == 1:
        return "по оси Z"
    if coordinate == 2:
        return "по оси Y"
    if coordinate == 3:
        return "по оси X"


measured_data = []
measured_data_dict = {}

with open('../some3.txt') as f:  # some1 - middle point, some2 - left wing point, some3 - right wing point
    for l in f:
        i = str2(l, " ")
        frame_number = int(l[:i].split(" ")[1])
        vector_str = l[i:].strip()
        if vector_str != "None":
            vector_str = vector_str[2:len(vector_str) - 2]
            vector_str = " ".join(vector_str.split())
            vector = [float(el) for el in vector_str.split(" ")][:3]
            measured_data.append((frame_number, vector))
            measured_data_dict[frame_number] = vector

real_data = []
with open('../LogFile.txt') as f:
    lines = []
    for l in f:
        lines.append(l)

    for l in lines[2::3]: # 0 - left wing point, 1 - middle point, 2 - right wing point
        i = str2(l, " ")
        frame_number = int(l[:i].split(" ")[1])
        vector_str = l[i:].strip()
        vector_str2 = re.split(" ", vector_str)
        x = vector_str2[2]
        y = vector_str2[4]
        z = vector_str2[6]
        x = x[:len(x) - 1].replace(',', '.')
        y = y[:len(y) - 1].replace(',', '.')
        z = z.replace(',', '.')
        vector = [float(z), float(y), float(x)]
        real_data.append((frame_number, vector))

coordinate = 3  # 1 - z, 2 - y, 3 - x

measurement_error_data = []
for data in real_data:
    frame_number, real_vector = data[0], data[1]
    measured_vector = measured_data_dict.get(frame_number)
    if measured_vector is not None:
        measurement_error = np.array(real_vector) - np.array(measured_vector)
        measurement_error_data.append((frame_number, measurement_error[coordinate - 1]))  # 0 - z, 1 - y, 2 - x
        # measurement_error_data.append((frame_number, math.sqrt(math.pow(measurement_error[0],2) +
        #                                                        math.pow(measurement_error[1],2) +
        #                                                        math.pow(measurement_error[2],2))))

import matplotlib.pyplot as plt

measures = np.array([[data[0], data[1][0], data[1][1], data[1][2]] for data in measured_data]).T
reals = np.array([[data[0], data[1][0], data[1][1], data[1][2]] for data in real_data]).T

plt.scatter(reals[0], reals[coordinate], s=1, color='red', label='real data')
plt.scatter(measures[0], measures[coordinate], s=1, c='blue', label='measered data')
plt.legend()
plt.xlabel("Номер кадра в видеопотоке")
plt.ylabel("Дистанция до объекта" + axis_name(coordinate))
plt.show()

measurement_errors = np.array([measurement_error_data]).T
plt.scatter(measurement_errors[0], abs(measurement_errors[1]), s=2, color='red')
plt.xlabel("Номер кадра в видеопотоке")
plt.ylabel("Ошика в определении дистанции до объекта" + axis_name(coordinate))
plt.show()
