from skspatial.objects import Line
from skspatial.objects import Points

def calculate_speed(coordinate_vector_curr, coordinate_vector_prev, frame_offset, frame_duration_s):
    difference_vector = (coordinate_vector_curr - coordinate_vector_prev)
    return difference_vector / (frame_offset * frame_duration_s)

def calculate_landing_point(data, y):
    points = Points(data)
    line_fit = Line.best_fit(points)

    p1 = line_fit.point
    p2 = p1 + line_fit.direction

    numerators = -p2
    denominators = p1 - p2
    value = (y + numerators[1]) / denominators[1]
    z = value * denominators[0] - numerators[0]
    x = value * denominators[2] - numerators[2]
    return z, x