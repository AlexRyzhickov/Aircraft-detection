import numpy as np
from skspatial.objects import Line
from skspatial.objects import Points

def calculate_speed(coordinate_vector_curr, coordinate_vector_prev, frame_offset, frame_duration_s, frame_number):
    difference_vector = (coordinate_vector_curr - coordinate_vector_prev)
    print("______________________")
    print("Frame: ", frame_number, ", velocity calculate for last ", (frame_offset * frame_duration_s), " s")
    print("Velocity Vector: ", difference_vector / (frame_offset * frame_duration_s))
    print("______________________")


def calculate_landing_point(data, y):
    points = Points(data)
    line_fit = Line.best_fit(points)

    a = line_fit.point
    b = a + line_fit.direction

    numerators = -b
    denominators = a - b
    value = (y + numerators[1]) / denominators[1]
    z = value * denominators[0] - numerators[0]
    x = value * denominators[2] - numerators[2]
    return z, x