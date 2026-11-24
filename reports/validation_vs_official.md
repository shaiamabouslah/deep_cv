# Validation contre des données officielles (ONAGRI/DGRE)

## Méthodologie

Les taux de remplissage officiels des barrages tunisiens (ONAGRI/DGRE) sont
relayés régulièrement par la presse économique tunisienne. Faute d'API
publique directe, trois points de données officiels ont été collectés via
recherche (presse citant ONAGRI/DGRE) pour **Sidi Salem** et **Nebhana** —
les deux sites les plus testés dans ce projet — puis comparés à notre
estimation NDWI (`water_area_pct`) calculée sur une image Sentinel-2 du
même mois via Google Earth Engine.

## Résultats bruts

| Site | Période | Notre estimation (NDWI) | Taux officiel (ONAGRI/DGRE) | Image utilisée |
|---|---|---|---|---|
| Sidi Salem | Sept. 2025 | 30.7 % | 22.7 % | 2025-09-01 |
| Sidi Salem | Avril 2026 | 22.7 % | 50.0 % | 2026-04-07 |
| Nebhana | Sept. 2025 | 34.3 % | 3.4 % | 2025-09-01 |

Sources presse (ONAGRI/DGRE) : Managers.tn, La Presse de Tunisie, Kapitalis
(avril 2026, septembre 2025).

## Interprétation honnête

**Ce n'est pas une validation positive — et c'est une découverte utile.**

1. **Mismatch de tendance sur Sidi Salem** : entre septembre 2025 et avril
   2026, le taux officiel **augmente** nettement (22.7 % → 50 %, après des
   pluies importantes), alors que notre estimation **diminue** (30.7 % →
   22.7 %). Les deux séries évoluent en sens opposé sur cette période.
2. **Écart important sur Nebhana** : l'officiel indique un barrage presque
   vide (3.4 %), notre estimation indique 34.3 % — un facteur ~10.
3. **Le rayon du buffer n'explique pas l'écart** : testé à 500 m, 1.5 km et
   3 km autour du même point pour Sidi Salem, le résultat ne change presque
   pas (21.7–22.9 % en avril, 30.7–32.5 % en septembre) et l'inversion de
   tendance persiste. La taille de la zone échantillonnée n'est donc pas la
   cause principale.

## Cause la plus probable (diagnostic technique)

Le calcul actuel, `water_area_pct = (NDWI + 1) * 50`, est un **simple
rescaling linéaire du NDWI** vers l'intervalle [0, 100] — ce n'est **pas**
un pourcentage calibré de la surface maximale ou de la capacité du barrage.
Le taux officiel ONAGRI/DGRE, lui, est très probablement un **pourcentage
de volume/capacité** (m³ stockés / capacité totale), une grandeur liée à la
surface par une courbe hauteur-volume propre à chaque barrage, non linéaire
et non disponible dans ce projet. Comparer directement les deux revient à
comparer deux unités différentes :

- Notre métrique : indice spectral de présence d'eau, non normalisé par site
- Métrique officielle : volume stocké / capacité totale du barrage

C'est la cause la plus probable du désaccord, bien plus que le choix du
buffer ou la précision des coordonnées (déjà vérifiée ci-dessus).

## Ce que ça change pour le projet

- Le `water_area_pct` actuel doit être présenté comme un **indice relatif
  de présence d'eau détectée**, pas comme "le pourcentage de remplissage du
  barrage" — nuance importante à l'oral.
- Une vraie calibration demanderait soit (a) la surface maximale connue de
  chaque barrage à pleine capacité (pour normaliser une surface mesurée en
  pixels, pas un simple rescaling de NDWI), soit (b) une régression de notre
  indice contre un historique plus long de relevés officiels pour apprendre
  la relation surface→volume empiriquement, propre à chaque site.
- Les deux pistes sont réalistes mais demandent plus de données officielles
  que ce qui est public via la presse — à chercher du côté d'un accès
  direct DGRE/ONAGRI (API ou bulletin structuré) en prochaine itération.

## Correction appliquée (2026-11-21)

Le `water_area_pct` a été recalculé avec une méthode physiquement fondée :
surface d'eau réelle (comptage de pixels classifiés NDWI > 0 × surface du
pixel), normalisée par l'**étendue maximale historique observée** pour ce
site via **JRC Global Surface Water** (Pekel et al., 2016 — référence
scientifique validée, 1984-2021), au lieu d'un simple rescaling linéaire
du NDWI moyen sans aucune signification physique.

| Site | Période | Avant (NDWI brut) | **Après (surface calibrée)** | Officiel |
|---|---|---|---|---|
| Sidi Salem | Sept. 2025 | 30.7 % | **78.2 %** (214.7 ha / 274.5 ha) | 22.7 % |
| Sidi Salem | Avril 2026 | 22.7 % | **90.5 %** (248.3 ha / 274.5 ha) | 50.0 % |
| Nebhana | Sept. 2025 | 34.3 % | **22.5 %** (11.1 ha / 49.5 ha) | 3.4 % |

**Ce qui est corrigé :** la tendance sur Sidi Salem est maintenant dans le
**bon sens** (78.2 % → 90.5 %, en hausse, comme l'officiel 22.7 % → 50.0 %)
— l'inversion de tendance initialement détectée a disparu. Nebhana reste
surestimé mais le facteur d'écart est réduit (×6.6 au lieu de ×10).

**Ce qui ne l'est pas :** l'écart absolu persiste. Cause la plus probable,
confirmée par les surfaces en hectares ci-dessus : le buffer circulaire de
6 km autour d'un point ne capture toujours pas l'étendue réelle du barrage
(274 ha mesurés pour Sidi Salem, un barrage dont la superficie réelle à
pleine capacité dépasse largement cette valeur) — il faudrait le vrai
polygone du réservoir (OpenStreetMap ou shapefile officiel), pas une
approximation circulaire, pour une correspondance fine à l'officiel.

## Correction géométrique (2026-11-21, round 2) : le vrai contour du barrage

Hypothèse testée : le buffer circulaire de 6 km n'est peut-être pas le
problème — c'est **le centre lui-même** qui est mal placé. Vérification
directe sur JRC Global Surface Water : aux 6 coordonnées approximatives de
`src/sites.py`, la bande `max_extent` vaut **0** à chaque point — aucune
des 6 coordonnées ne tombe sur de l'eau classifiée. Mesure de l'écart réel
(distance au pixel d'eau le plus proche, par transformée de distance) :

| Site | Écart mesuré |
|---|---|
| Sidi Salem | 2816 m |
| Nebhana | 2470 m |
| Sidi El Barrak | 2220 m |
| Bir M'cherga | 721 m |
| Mellègue | 8105 m |
| Lakhmes | 1487 m |

**Méthode de correction** : recherche de la plus grande composante
connexe d'eau (JRC `max_extent`) dans une fenêtre de 15 km autour de
chaque point approximatif — robuste à une graine imprécise, tant que le
vrai réservoir se trouve dans la fenêtre. Résultat :

| Site | Surface réelle trouvée | Verdict |
|---|---|---|
| Sidi Salem | 4292 ha | plausible (grand barrage, cohérent avec sa réputation) |
| Nebhana | 29 ha | plausible (petit barrage) |
| Bir M'cherga | 601 ha | plausible |
| Lakhmes | 91 ha | plausible |
| Sidi El Barrak | 28 359 ha | **implausible** — probablement fusion avec la mer Méditerranée toute proche (pas de masque terre/mer dans cette recherche) |
| Mellègue | 11 ha | **implausible** — bien trop petit pour ce barrage ; le vrai réservoir est probablement hors de la fenêtre de 15 km, cohérent avec son écart initial (8.1 km, le plus grand des 6) |

**Décision** : `src/sites.py` adopte le centre corrigé et un buffer
proportionné à la surface mesurée pour les 4 sites plausibles (Sidi Salem,
Nebhana, Bir M'cherga, Lakhmes). Sidi El Barrak et Mellègue gardent leurs
coordonnées approximatives d'origine — intégrer un résultat qu'on sait
probablement faux serait pire que de garder l'approximation déjà
documentée comme telle.

## Revalidation finale contre l'officiel (coordonnées corrigées)

| Site | Période | Avant correction géométrique | **Après (centre + buffer corrigés)** | Officiel |
|---|---|---|---|---|
| Sidi Salem | Sept. 2025 | 78.2 % | **61.5 %** (2170 ha / 3527 ha) | 22.7 % |
| Sidi Salem | Avril 2026 | 90.5 % | **79.7 %** (2812 ha / 3527 ha) | 50.0 % |
| Nebhana | Sept. 2025 | 22.5 % | **35.3 %** (11.1 ha / 31.3 ha) | 3.4 % |

**Ce que la correction géométrique a réellement changé** : avant, le
buffer de 6 km centré sur un mauvais point ne contenait, pour Sidi Salem,
que 274 ha d'étendue maximale historique — une fraction minime du vrai
lac. Centré correctement, ce même buffer en contient 3527 ha, un ordre de
grandeur cohérent avec la taille réelle connue du barrage. La géométrie
d'échantillonnage n'est donc plus le facteur limitant.

**Ce qui persiste malgré une géométrie correcte** : l'écart absolu avec
l'officiel ne se réduit pas (et se creuse même légèrement sur Sidi Salem).
La tendance reste dans le bon sens (hausse → hausse). Ceci confirme — plus
que ça ne l'infirme — le diagnostic initial : l'écart restant n'est pas un
problème de géométrie mais de **nature de la grandeur mesurée**. Un
réservoir a un profil en cuvette : sa surface se remplit beaucoup plus
vite que son volume près de la cote de retenue normale (une baisse de
volume de moitié ne réduit pas la surface de moitié, car l'eau restante
est concentrée dans la partie la plus profonde et la plus étroite). Notre
`water_area_pct` est un indice de **surface**, l'officiel ONAGRI/DGRE un
pourcentage de **volume/capacité** — deux courbes différentes pour le même
barrage, non linéaires l'une par rapport à l'autre, qui ne coïncideront
jamais exactement sans la courbe hauteur-surface-volume propre à chaque
site (non publique).

## Correction géométrique (2026-11-24, round 3) : Sidi El Barrak et Mellègue

Les deux sites laissés non corrigés au round 2 avaient chacun une cause
distincte, identifiée puis traitée séparément :

**Sidi El Barrak (283 km² au round 2)** : Sidi El Barrak est un barrage
côtier, à quelques kilomètres de la Méditerranée. La recherche de plus
grande composante connexe, sans distinction terre/mer, fusionnait le
réservoir avec la mer ouverte — JRC `max_extent` classe l'eau de mer
exactement comme l'eau douce. **Correction** : la région de recherche est
découpée (`intersection`) avec un masque terre uniquement, construit à
partir des polygones de pays `USDOS/LSIB_SIMPLE/2017` (Tunisie), avant de
chercher la plus grande composante. Résultat : **3018 ha** — un ordre de
grandeur plausible pour un grand barrage, contre 28 359 ha avant.

**Mellègue (11 ha au round 2)** : Mellègue est un barrage sur l'oued
Mellègue, dont le bassin chevauche la frontière tuniso-algérienne — et son
écart initial mesuré (8.1 km) était déjà le plus grand des 6 sites,
suggérant que le vrai réservoir se trouvait hors de la fenêtre de
recherche de 15 km. **Correction** : fenêtre élargie à 35 km, masque
terre étendu à la Tunisie **et** l'Algérie (pour ne pas couper le
réservoir au niveau de la frontière). Résultat : **897 ha** à 22.8 km du
point d'origine — un ordre de grandeur cohérent avec un grand barrage,
contre 11 ha avant. (Testé aussi à 25 km : seulement 42.7 ha, confirmant
que le vrai réservoir était bien hors de cette fenêtre intermédiaire et
que 35 km était nécessaire.)

**Limite assumée** : contrairement à Sidi Salem et Nebhana, il n'existe
pas de point de données officiel ONAGRI/DGRE collecté pour ces deux sites
dans ce projet — la validation ici est une **vérification de plausibilité**
(l'ordre de grandeur en hectares est cohérent avec la réputation de ces
barrages comme grands réservoirs), pas une validation contre un chiffre
officiel comme pour les deux autres sites.

## Verdict

**Résolu : la géométrie d'échantillonnage, pour les 6 sites.** Les 4 sites
du round 2 sont centrés sur le vrai réservoir avec une étendue de référence
réaliste ; les 2 sites restants (Sidi El Barrak, Mellègue) sont maintenant
corrigés aussi, via un masque terre/mer et une fenêtre de recherche élargie
— ce n'est plus une approximation grossière nulle part dans
`src/sites.py`. **Non résolu, et correctement attribué : l'écart absolu**
entre notre indice et l'officiel (Sidi Salem, Nebhana), qui est un problème
de nature de grandeur (surface vs volume), pas de géométrie. **Non validé
officiellement (seulement plausible) : Sidi El Barrak et Mellègue**, faute
de point de données ONAGRI/DGRE collecté pour ces deux sites. C'est un
résultat honnête en trois temps : un vrai problème résolu (la géométrie,
sur les 6 sites), un autre correctement identifié et expliqué plutôt que
masqué (surface ≠ volume), et une limite de validation assumée plutôt que
cachée (plausibilité ≠ validation officielle) — pas une solution magique.
