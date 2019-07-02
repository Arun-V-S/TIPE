def process(function, i, values, *args):
    values[i] = function(*args)
    print(values[i])

def compte(nombre, ):
    nombre = nombre[0]
    a = 0
    for i in range(nombre):
        a = (-1) ** i + i
    return a
