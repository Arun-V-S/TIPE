import csv
import os
from lxml import etree


ABSOLUTE = 'D:/Documents/Prepa/TIPE'

pathBateaux = ABSOLUTE + "/data/MASATI-v2/ship"
pathXml = ABSOLUTE + "/data/MASATI-v2/ship_labels"
pathMer = ABSOLUTE + "/data/MASATI-v2/water"

NUMBERSHIP = 5 # 1027
NUMBERWATER = 5 # 1022

def generateCsv(nombre, total):
    listeBateaux = os.listdir(pathBateaux)
    with open(ABSOLUTE + '/bateauxPos.csv', 'w') as fichier:
        filewriter = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(nombre):
            nom = pathBateaux + '/' + listeBateaux[i]
            #filewriter.writerow([pathBateaux + '/' + listeBateaux[i] + ',0'])
            tree = etree.parse(pathXml + '/' + (listeBateaux[i].replace('.png', '.xml')))

            xmin = tree.xpath('/annotation/object/bndbox/xmin')[0].text
            xmax = tree.xpath('/annotation/object/bndbox/xmax')[0].text
            ymin = tree.xpath('/annotation/object/bndbox/ymin')[0].text
            ymax = tree.xpath('/annotation/object/bndbox/ymax')[0].text
            filewriter.writerow([nom + ',' + xmin + ',' + xmax + ',' + ymin + ',' + ymax])
    with open(ABSOLUTE + '/bateauxPosVal.csv', 'w') as fichier:
        filewriter = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(nombre, total):
            nom = pathBateaux + '/' + listeBateaux[i]
            #filewriter.writerow([pathBateaux + '/' + listeBateaux[i] + ',0'])
            tree = etree.parse(pathXml + '/' + (listeBateaux[i].replace('.png', '.xml')))

            xmin = tree.xpath('/annotation/object/bndbox/xmin')[0].text
            xmax = tree.xpath('/annotation/object/bndbox/xmax')[0].text
            ymin = tree.xpath('/annotation/object/bndbox/ymin')[0].text
            ymax = tree.xpath('/annotation/object/bndbox/ymax')[0].text
            filewriter.writerow([nom + ',' + xmin + ',' + xmax + ',' + ymin + ',' + ymax])

    return None
