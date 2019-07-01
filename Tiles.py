TILES = [0, 1] #0 eau, 1 terrain
def ranTiles(size, tiles):
    """Génère une matrice aléatoire remplie d'éléments de tiles."""
    m = len(tiles)
    liste = [[]]
    for i in range(size):
        for j in range(size):
            n = map(random(), (0, 1), (0, m), 1)
            liste[i].append(tiles[n])
        liste.append([])
    return liste

def printMultTiles(taille, chunk, mat, couleurs, n):
    """Enregistre n images avec les paramètres entrés."""
    for _ in range(n):
        mat = ranTiles(10, TILES)
        printTiles(taille, chunk, mat, couleurs)
    return

def printTiles(taille, chunk, mat, couleurs):
    """affiche une image à partir de la matrice, des couleurs correspondantes et de la taille de chaque chunk."""
    image = new('RGB', (taille, taille))
    size = image.size[0] #Image carrée!
    n = size / chunk
    print(n)
    if n != int(n):
        print("Pas bon!")
        return False
    n = int(n)
    for i in range(n):
        for j in range(n):
            for y in range(chunk):
                for x in range(chunk):
                    image.putpixel((i * chunk + x, j * chunk + y), couleurs[mat[i][j]])
    backslash = "\\"
    print(backslash + str(taille) + backslash + str(int(random() * (10 ** 10))) + ".png")
    image.save(str(taille) + backslash + str(int(random() * (10 ** 10))) + ".png", "png")
