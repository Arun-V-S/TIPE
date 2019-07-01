from opensimplex import OpenSimplex
from random import *
from math import *


def IssouC(height, width, coeffs, expo):
    """Crée l'image height*width avec coeffs comme coefficients des harmoniques et expo l'exposant correcteur."""
    n = len(coeffs)
    simplex = OpenSimplex(seed = int(random() * (10 ** 15)))
    mat = []
    min = 100
    max = -1
    sum = 0
    for i in range(n):
        sum += coeffs[i] #Calcule la somem des coeffs

    for y in range(0, height):
        mat.append([])
        for x in range(0, width):
            nx = x/width - 0.5 #Positions normalisées
            ny = y/width - 0.5 #entre -0.5 et +0.5
            e = 0
            for i in range(n):
                e += abs(coeffs[i] * simplex.noise2d((2 ** i) * nx, (2 ** i) * ny))
            e = e / sum #Renormalise avec la somme des coeffs
            e = pow(e, expo) #Sert à plus ou moins accentuer les variations
            mat[y].append(e)
            if e < min:
                min = e
            elif e > max:
                max = e
        os.system('cls')
        print("Bruit : " + str(int((y / width) * 100)) + "%")
    return (mat, min, max)
