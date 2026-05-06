import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = None

dataset = []

M = 50
n_points = 100

for i in range(n_points):
    x = np.random.rand()*M
    y = np.random.rand()*M
    dataset.append(Point(x, y))

