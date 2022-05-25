import numpy as np


def calculate_speed(coordinate_vector_curr, coordinate_vector_prev, frame_offset, frame_duration_s, frame_number):
    difference_vector = (coordinate_vector_curr - coordinate_vector_prev)
    print("______________________")
    print("Frame: ", frame_number, ", velocity calculate for last ", (frame_offset * frame_duration_s), " s")
    print("Velocity Vector: ", difference_vector / (frame_offset * frame_duration_s))
    print("______________________")


def calculate_landing_point(data, y):
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
    print(linepts)
    numerators = -linepts[0]
    denominators = linepts[1] - linepts[0]
    print(numerators)
    print(denominators)
    value = (y + numerators[1]) / denominators[1]
    x = value * denominators[0] - numerators[0]
    z = value * denominators[2] - numerators[2]
    return x, z