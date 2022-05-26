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

real_data = []
with open('./LogFile.txt') as f:
    lines = []
    for l in f:
        lines.append(l)

    for l in lines[1::3]: # 0 - left wing point, 1 - middle point, 2 - right wing point
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
        vector = [float(x), float(y), float(z)]
        real_data.append(vector)
        print(vector)

points = Points(real_data)
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