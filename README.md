**TIPE** : Application des technologies de réseaux de neurones à la reconnaissance et la localisation de bateaux sur des images satellites. 

I Génération d'images, noyaux de convolutions
=============================================

1- Bruit de Perlin
------------------
Le bruit de Perlin est un générateur de bruit cohérent permettant de générer une __heightmap__, pouvant ensuite servir à construire un terrain plus ou moins complexe. 


2- Algorithmes d'interpolation
------------------------------
Certains algorithmes (Diamond-Square, par exemple) permettent de passer d'une petite matrice (par exemple 17 x 17) de valeurs aléatoires à une matrice (1024 x 1024 dans mon exemple) de valeurs respectant une certaine __continuité spatiale__. 


3- Filtres et noyaux de convolutions
------------------------------------
Une première étude et implémentation des noyaux de convolution préparent des outils pour l'analyse d'images à proprement parler. À chaque filtre (par exemple flou gaußien ou filtre de gradient) peut être associé un noyau de convolution. 

II Première implémentation naïve de réseaux
===========================================

1- Perceptron multicouche et reconnaissance de caractères
---------------------------------------------------------

2- Réseau convolutif et reconnaissance d'objets
-----------------------------------------------
Les simples réseaux linéaires entièrement connectés ne suffisent pas à analyser de façon fiable une image. En effet, il ont une __dépendance spatiale__ trop importante. Il faut donc s'en libérer, en utilisant des couches de convolution, reprenant le principe des noyaux de convolution vus précédemment. 

III Utilisation de Pytorch
==========================

1- Réseau de détection
----------------------


2- Réseau de localisation
-------------------------


3- Plus avancé : génération d'images
------------------------------------


IV Réseaux de neurones et évolution génétique 
=============================================
