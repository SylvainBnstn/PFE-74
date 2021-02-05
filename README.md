# PFE-2033
Projet de fin d'étude n°33

ECE Paris 2020/2021

Auteurs:
- Arnaud BAZIN DE BEZONS
- Sylvain BENISTAND
- Jonathan BOUTAKHOT
- Loïc CHEONG
- Thomas HU
- Cécile WANG

Mentor : Jae Yun JUN KIM

Liste des projets PFE 2020/2021 :
https://docs.google.com/spreadsheets/d/1ZpePOLN7cyXvn-rRZXs5oedDq6WzBF2Q4LG4erQi3Xk/edit#gid=0



# Objectifs
- Concevoir une stratégie de tarification dynamique (dynamic pricing) qui optimise le porfit pour un vendeur qui vend des produits pour certaines informations données sur le produit, l'état du marché et les clients
- Modéliser le comportement des clients (naïf, stratégique)
- Modéliser l'effet de l'intéraction client-vendeur sur les prix
- Utiliser le DQN pour ajuster automatiquement les prix afin de maximiser le profit

# Problèmatique
Quelle stratégie de dynamic pricing à adopter dans un marché en présence de clients stratégiques avec un produit ayant une maturité fixée (finie ou infinie) ?

# Choix du produit
L'étude a été fait sur les logements d'Airbnb sur les "Entire home/apt" entre $100-$200 de la ville de New-York entre l'année 2015-2020. Les données ont été récoltés sur ce site http://insideairbnb.com/get-the-data.html. Chaque URLs ont sont stockés dans `Code/urls.txt`



# Comment exécuter le code ?
## Télécharger ou mettre à jour les données
Les données brutes sont stockées dans `Code/airbnb.csv`. S'il n'y est pas sur github, c'est parce que le fichier est très lourd (+300Mo). 

Télécharger les données en décommentant les trois dernières lignes de `Code/airbnb_processing.py`. Par défaut, seulement les données de "new-york-city" sont téléchargées. Pour télécharger les données d'une autre ville, remplacer le nom de la ville dans la fonction `get_data(URLS,"airbnb_data.csv",["new-york-city"])` dans `Code/airbnb_processing.py`

Pour mettre à jour de nouveaux jeux de données, copier l'URLs depuis http://insideairbnb.com/get-the-data.html, puis coller dans `Code/urls.txt`

## Lancer le DQN
Exécuter le fichier `Code/Main.py`

# Explication
Tout se trouve dans le dossier **Rapport** 

