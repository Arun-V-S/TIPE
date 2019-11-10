**TIPE** : Application des technologies de réseaux de neurones à la reconnaissance et la localisation de bateaux sur des images satellites. 

I Génération d'images, noyaux de convolutions
=============================================
Pour commencer, on veut développer des outils de base pour la suite du travail. En attendant de disposer d'images réelles, la génération d'images le plus réalistes possibles est nécessaire. 

1 Bruit de Perlin
------------------
Le bruit de Perlin est un générateur de bruit cohérent permettant de générer une __heightmap__, pouvant ensuite servir à construire un terrain plus ou moins complexe. 


2 Algorithmes d'interpolation
------------------------------
Certains algorithmes (Diamond-Square, par exemple) permettent de passer d'une petite matrice (par exemple 17 x 17) de valeurs aléatoires à une matrice (1024 x 1024 dans mon exemple) de valeurs respectant une certaine __continuité spatiale__. 


3 Filtres et noyaux de convolutions
------------------------------------
Une première étude et implémentation des noyaux de convolution préparent des outils pour l'analyse d'images à proprement parler. À chaque filtre (par exemple flou gaußien ou filtre de gradient) peut être associé un noyau de convolution. 

II Première implémentation naïve de réseaux
===========================================
Une première implémentation, peu efficace mais fonctionnelle, permet de se rendre compte des avantages ainsi que des difficultés de cette branche du machine learning. 

1 Perceptron multicouche et reconnaissance de caractères
--------------------------------------------------------
Implémentation d'un premier type de réseaux : les __réseaux linéaires entièrement connectés__, par couches. Test sur des fonctions élémentaires de __logique__ (AND, XOR), ainsi que sur de la __reconnaissance de caractères__ à petite échelle, comme preuve de principe. 

2 Réseau convolutif et reconnaissance d'objets
-----------------------------------------------
Les simples réseaux linéaires entièrement connectés ne suffisent pas à analyser de façon fiable une image. En effet, il ont une __dépendance spatiale__ trop importante. Il faut donc s'en libérer, en utilisant des couches de convolution, reprenant le principe des noyaux de convolution vus précédemment. 

III Utilisation de Pytorch
==========================
L'utilisation d'une bibliothèque, Pytorch en l'occurrence, qui se fonde sur des algorithmes en CUDA et C++ pour les fonctions gourmandes en calculs, permet de pouvoir s'intéresser à une tâche plus conséquente : la détection de bateaux sur des images satellites de grande taille (512 x 512).

1 Réseau de détection
----------------------


2 Réseau de localisation
-------------------------


3 Plus avancé : génération d'images
------------------------------------


IV Réseaux de neurones et évolution génétique 
=============================================
