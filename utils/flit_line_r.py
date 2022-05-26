from skspatial.objects import Line
from skspatial.objects import Points
from skspatial.plotting import plot_3d
import re

def str2(str, s):
    isNotFirst = False
    for i in range(len(str)):
        if str[i] == s:
            if isNotFirst:
                return i
            if not isNotFirst:
                isNotFirst = True
    return -1

measured_data = []

with open('./some1.txt') as f:  # some1 - middle point, some2 - left wing point, some3 - right wing point
    for l in f:
        i = str2(l, " ")
        frame_number = int(l[:i].split(" ")[1])
        vector_str = l[i:].strip()
        if vector_str != "None":
            vector_str = vector_str[2:len(vector_str) - 2]
            vector_str = " ".join(vector_str.split())
            vector = [float(el) for el in vector_str.split(" ")][:3]
            vector = [vector[2], vector[1], vector[0]]
            measured_data.append(vector)


points = Points(measured_data[700:756])
line_fit = Line.best_fit(points)

a = line_fit.point
b = a + line_fit.direction

numerators = -b
denominators = a - b
value = (18 + numerators[1]) / denominators[1]
z = value * denominators[0] - numerators[0]
x = value * denominators[2] - numerators[2]

print("x", x, "z", z)

plot, ax = plot_3d(
    line_fit.plotter(c='k'),
    points.plotter(c='b', depthshade=False)
)
ax.set_xlabel("Z-axis")
ax.set_ylabel("X-axis")
ax.set_zlabel("Y-axis")
plot.show()