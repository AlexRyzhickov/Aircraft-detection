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
    # data = np.array(data)
    # datamean = data.mean(axis=0)
    # uu, dd, vv = np.linalg.svd(data - datamean)
    # linepts = vv[0] * np.mgrid[-1000:1:2j][:, np.newaxis]
    # numerators = -linepts[0]
    # denominators = linepts[1] - linepts[0]
    # print(numerators)
    # print(denominators)
    # value = (y + numerators[1]) / denominators[1]
    # x = value * denominators[0] - numerators[0]
    # z = value * denominators[2] - numerators[2]
    # linepts += datamean
    # numerators = -linepts[0]
    # denominators = linepts[1] - linepts[0]
    # value = (16 + numerators[1]) / denominators[1]
    # x = value * denominators[0] - numerators[0]
    # z = value * denominators[2] - numerators[2]

    # if frame_n == 1000 or frame_n == 1100 or frame_n == 1200 or frame_n == 1300 or frame_n == 1400:
    #     import matplotlib.pyplot as plt
    #     import mpl_toolkits.mplot3d as m3d
    #
    #     ax = m3d.Axes3D(plt.figure())
    #     ax.set(xlim=(-30, 30), ylim=(0, 1500), zlim=(16, 150))
    #     ax.set_xlabel("Z-axis")
    #     ax.set_ylabel("X-axis")
    #     ax.set_zlabel("Y-axis")
    #     ax.scatter3D(*data.T)
    #     ax.plot3D(*linepts.T)
    #     plt.show()
    #
    # return x, z