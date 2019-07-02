from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def imageFourier(nomImage):
    matrice = np.ndarray.flatten(np.asarray(Image.open(nomImage).convert('LA')))
    transf = np.fft.fft(matrice)
    freq = np.fft.fftfreq(matrice.shape[-1])
    plt.plot(freq, transf.real)
    plt.show()
