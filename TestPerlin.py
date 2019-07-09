import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict
import matplotlib.image as matim
import visvis
import cv2
from vispy import app, scene
from vispy.util.filter import gaussian_filter
import vispy as vs
import sys
sys.setrecursionlimit(50000)

global CMAPGRAD

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_octaves(shape, freq, pers, oct): #Shape doit être de la forme (x, y, z) où z = 1 (généralement)
    n = np.zeros((shape[0], shape[1]))
    for i in range(oct):
        res = ((freq ** i), (freq ** i))
        n += (pers ** i) * generate_perlin_noise_2d(shape, res)
    return n


def surface_plot(matrice):
    global CMAPGRAD
    (X, Y) = np.meshgrid(np.arange(matrice.shape[0]), np.arange(matrice.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, matrice, cmap = cm.gist_earth, rcount=200, ccount = 200, linewidth=0, antialiased=False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()


def cmapGradient(points = 100):
    """Renvoie l'espace de couleurs lié à l'image de gradient avec un certain nb de points (donc résolution de couleur)."""
    global CMAPGRAD
    img = matim.imread('Gradient.png')
    vert, hor, depth = img.shape
    res = hor / points
    resultat = []
    for i in range(points - 1):
        couleur = mp.colors.to_hex(tuple(img[0, int(res * i)]))
        resultat.append((round(i / points, 2), couleur))
    resultat.append((1, mp.colors.to_hex(tuple(img[0, hor - 1]))))
    print(resultat)
    CMAPGRAD = resultat #LinearSegmentedColormap.from_list('Issou', resultat)

def cmapGradient2(points = 100):
    """Renvoie l'espace de couleurs lié à l'image de gradient avec un certain nb de points (donc résolution de couleur)."""
    global CMAPGRAD
    img = cv2.imread('gradient.png', -1)[1, :]
    vert, hor = img.shape
    res = vert / points
    resultat = np.zeros(points, dtype = '3i')
    for i in range(points - 1):
        couleur = img[int(i * res)]
        resultat[i][0] = couleur[2]
        resultat[i][1] = couleur[1]
        resultat[i][2] = couleur[0]
        #resultat[i][3] = 127
    resultat[points - 1] = img[vert - 1]
    #resultat[points - 1][3] = 127
    CMAPGRAD = resultat




"""cmapGradient2(100)"""
"""
LOL4 = generate_octaves((1024, 1024), 2, 0.4, 8)
LOL4 = (LOL4 + 1) * 127
LOL4 = LOL4.astype(int)
print(np.ndarray.max(LOL4))
xi = np.arange(0, 1024)
yi = np.arange(0, 1024)
f = visvis.gca()
m = visvis.grid(xi, yi, LOL4)
f.daspect = 1,1,1 # z x 10
m = visvis.surf(xi,yi,LOL4)
m.SetValues(np.linspace(0, 1, np.ndarray.max(LOL4) - np.ndarray.min(LOL4)))
m.colormap = CMAPGRAD
"""

def Vispy(matrice):
    def normalize(x, cmin=None, cmax=None, clip=True):
        """Normalize an array from the range [cmin, cmax] to [0,1],
        with optional clipping."""
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if cmin is None:
            cmin = x.min()
        if cmax is None:
            cmax = x.max()
        if cmin == cmax:
            return .5 * np.ones(x.shape)
        else:
            cmin, cmax = float(cmin), float(cmax)
            y = (x - cmin) * 1. / (cmax - cmin)
            if clip:
                y = np.clip(y, 0., 1.)
        return y
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='w')
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up='z', fov=60)
    matrice = (matrice + 1) * 127
    # Simple surface plot example
    # x, y values are not specified, so assumed to be 0:50
    couleurs = normalize(CMAPGRAD)
    couleurs = np.flip(couleurs, 0)
    p1 = scene.visuals.SurfacePlot(z=matrice)
    #couleurs[0]
    p1.transform = scene.transforms.MatrixTransform()
    p1.transform.scale([1/100, 1/100, 4/100])
    p1.transform.translate([0, 0, 0])

    view.add(p1)
    p1._update_data()
    # p1._update_data()  # cheating.
    cf = scene.filters.ZColormapFilter(vs.color.Colormap(couleurs, interpolation='linear'), zrange=(matrice.max(), matrice.min()))
    p1.attach(cf)


    xax = scene.Axis(pos=[[-0.5, -0.5], [0.5, -0.5]], tick_direction=(0, -1),
                    font_size=16, axis_color='k', tick_color='k', text_color='k',
                    parent=view.scene)
    xax.transform = scene.STTransform(translate=(0, 0, -0.2))

    yax = scene.Axis(pos=[[-0.5, -0.5], [-0.5, 0.5]], tick_direction=(-1, 0),
                    font_size=16, axis_color='k', tick_color='k', text_color='k',
                    parent=view.scene)
    yax.transform = scene.STTransform(translate=(0, 0, -0.2))

    # Add a 3D axis to keep us oriented
    axis = scene.visuals.XYZAxis(parent=view.scene)

    canvas.show()
    app.run()

def runVispy():
    print("Début de Perlin")
    TEST = generate_octaves((2048, 2048), 2, 0.6, 8)
    print("Début de préparation de l'affichage")
    Vispy(TEST)
