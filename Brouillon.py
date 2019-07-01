def image(): #Exemple de gestion d'une image
    image = open("TR.jpg")
    print(image.format, image.size, image.mode)
    (x, y) = image.size
    for i in range(x):
        for j in range(y):
            (r, g, b) = (map(random(), (0, 1), (0, 255), 1), map(random(), (0, 1), (0, 255), 1), map(random(), (0, 1), (0, 255), 1))
            image.putpixel((i, j), (r, g, b))
    Image.show(image)
    return False
