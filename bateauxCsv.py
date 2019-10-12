import csv
import os


ABSOLUTE = 'D:/Documents/Prepa/TIPE'

pathBateaux = ABSOLUTE + "/data/MASATI-v2/detail"
pathMer = ABSOLUTE + "/data/MASATI-v2/water"

NUMBERSHIP = 5 # 1027
NUMBERWATER = 5 # 1022

def generateCsv(nombre):
    listeBateaux = os.listdir(pathBateaux)
    listeMer = os.listdir(pathMer)
    with open(ABSOLUTE + '/bateaux.csv', 'w') as fichier:
        filewriter = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(nombre):
            filewriter.writerow([pathBateaux + '/' + listeBateaux[i] + ',0'])
        for i in range(nombre):
            filewriter.writerow([pathMer + '/' + listeMer[i] + ',1'])
    return None
