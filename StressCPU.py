from random import *
from PIL import Image
from math import *
import numpy as np

fichier = "TR.jpeg"

def StressImg1():
    """affiche une image à partir de la matrice, des couleurs correspondantes et de la taille de chaque chunk."""
    image = Image.open('TR.jpg')
    height, width = image.size
    for y in range(width):
        for x in range(height):
            (r, g, b) = image.getpixel((x, y))
            image.putpixel((x, y), (g, b, r))
    #image.show()
    
Issou = np.vectorize(StressImg1)