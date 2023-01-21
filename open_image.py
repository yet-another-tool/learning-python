import cv2
import numpy as np


def plot_number(inputs, h=27):
    line = ""
    for p in inputs:
        line += ".░▒▓█"[round(p * 4)]
        if len(line) > h:
            print(line)
            line = ""


image = cv2.imread("./dataset/0.png", cv2.IMREAD_GRAYSCALE)

scale_percent = 100  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

plot_number([round(float(c/255), 2)
             for _ in resized for c in _], height-1)
plot_number([round(float(c/255), 2)
             for _ in np.invert(resized) for c in _], height-1)
