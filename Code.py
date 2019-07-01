import os
from PIL.Image import *

def image():
    image = open("TR.jpg")
    print(image.format, image.size, image.mode)
    (x, y) = image.size
    for i in range(x):
        for j in range(y):
            (r, g, b) = image.getpixel((i, j))
            image.putpixel((i, j), (2 * r, 2 * g, 2 * b))
    Image.show(image)
    return False
