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
Le premier modèle entraîné, composé de 7 couches de convolution et de 2 couches entièrement connectées, est un simple __classifier__ entraîné pour différencier les images satellites comportant un bateau de celles n'en comportant pas. 


2 Réseau de localisation
-------------------------
Le deuxième modèle a une fonction de localisation, ie. il doit renvoyer, lorsqu'une image comportant un bateau lui est soumis, les coordonnées d'une __bounding box__ encadrant ce bateau. 


3 Plus avancé : génération d'images
-----------------------------------
Avec une structure différente, un __GAN__ (Generative Adversarial Network) ou un __Autoencoder__, il est possible de générer, à partir d'exemples réels, des images satellites réalistes. 


IV Réseaux de neurones et évolution génétique 
=============================================
Les algorithmes d'évolution génétiques sont une branche de l'__optimisation stochastique__ de modèles, dans le cas où une démarche déterministe serait trop complexe, voire impossible. Ici, il s'agit non seulement de déterminer un ensemble de poids et biais satisfaisant une condition de précision, mais aussi de déterminer une __architecture__ de réseau qui la satisfasse. 

1 Une première approche
---
Pour commencer, il semble cohérent de laisser à l'algorithme une liberté totale, concernant l'architecture des réseaux de neurones produits et testés. Pourtant, de façon empirique, les algorithmes ne convergent pas, principalement à cause de la présence d'__actions récursives__ dans l'architecture, tandis que notre domaine d'étude se restreint à des problèmes statiques, c'est à dire ne dépendant pas du temps. 

2 Certaines conditions 
---
En ajoutant des __couches virtuelles__ dans l'architecture des réseaux, on peut s'assurer qu'aucune situation récursive ne se présente. Ce faisant, la recherche de solution est restreinte à des __solutions statiques__, ce qui correspond à notre problème. Expérimentalement, cette fois-ci, l'algorithme d'évolution converge vers un réseaux satisfaisant pour notre problème (relations logiques et reconnaissance de caractères). 
