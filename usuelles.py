def moyenne(liste):
    """Renvoie la moyenne des éléments de la liste."""
    n = len(liste)
    comp = 0
    for element in liste:
        comp += element/n
    return comp

def ecartType(liste, m = 0):
    """Retourne l'écart-type des valeurs de liste."""
    n = len(liste)
    if m == 0:
        m = moyenne(liste)
    comp = 0
    for element in liste:
        comp += ((element - m) ** 2) / n
    return comp ** (0.5)

def testRandom(randomf, n = 1000000):
    """Teste le caractère aléatoire d'une fonction. Renvoie l'écart-type et la moyenne."""
    liste = []
    for _ in range(n):
        liste.append(randomf)
    m = moyenne(liste)
    return (m, ecartType(liste, m))

def map(n, dep, arr, I = 0):
    """Remape la valeur n compris dans l'intervalle dep sur l'intervalle arr."""
    (x, y) = dep
    (i, j) = arr
    res = ((n - x) * (j - i)) / (y - x) + i
    if I == 1:
        res = int(res)
    return res

def indice(element, liste):
    """Renvoie l'indice de l'element dans la liste, -1 s'il n'est pas dedans."""
    n = len(liste)
    for i in range(n):
        if liste[i] == element:
            return i
    return -1
