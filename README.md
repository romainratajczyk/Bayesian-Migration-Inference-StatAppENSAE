
# Prédiction bayésienne des flux migratoires internationaux 
[Dépôt GitHub - Projet Migration](https://github.com/IshaghCheikh/ProjetStat/tree/main)

Ce projet a pour objectif de comprendre la dynamique des flux migratoires internationaux bilatéraux et d'en prédire l'évolution. Face à la complexité de la réalité macroéconomique et statistique des flux, nous déployons une méthodologie progressive : d'un modèle de gravité standard vers des algorithmes de Machine Learning, pour aboutir à une modélisation bayésienne hiérarchique de pointe.

## 🔬 Notre démarche scientifique 

Notre stratégie de modélisation s'articule autour de trois grandes étapes :

1. **Le Benchmark Gravitaire :** Implémentation d'un modèle de gravité log-linéaire classique (OLS) pour capturer les déterminants standards (distance, PIB, liens coloniaux). Cette étape sert de point de référence.
2. **L'Exploration Non-Linéaire (Machine Learning) :** Utilisation de modèles ensemblistes (Random Forest, XGBoost) pour challenger la linéarité du modèle de gravité. Cette étape s'est révélée cruciale pour :
   * Détecter les effets de seuil et les interactions complexes entre variables.
   * Analyser les cartes de résidus (comprendre géographiquement où le modèle se trompe).
   * Extraire les *feature importances*.
3. **L'Inférence Bayésienne Hiérarchique :** Les découvertes issues du Machine Learning sont ensuite injectées dans notre modèle statistique final (ARX Hurdle Bayésien) pour modéliser l'hétéroscédasticité par dyade, informer les priors, disposer des bonnes variables économétriques, et obtenir des prédictions robustes (*Empricial Bayes*).

## 🎯 La finalité : disposer de deux modèles robustes, aux ambitions différentes.

L'objectif in fine est de doter les décideurs publics d'un outil de prévision complet, reposant sur deux modèles complémentaires :

* **Le Pilier "Temps Long" (Modèle Welch & Raftery) :** Une réplication du modèle de référence OutFlow/Allocation. La méthodologie repose sur le calcul d'un taux de départ global par pays d'origine, dont le volume est ensuite réparti dans le monde via une distribution multinomiale. Ce modèle n'utilise aucune variable économétrique, seulement les masses de population. Il gère parfaitement la nature discrète des flux (nombres entiers) et s'avère extrêmement pertinent pour des projections de très longue durée (2050, 2100 et au-delà en théorie).  
* **Le Pilier "Temps Court" (Notre Modèle ARX Hurdle) :** Un modèle bayésien de gravité bilatérale, hautement réactif à l'économétrie et préparé aux chocs macro-démographiques. Pensé pour la précision à court terme (<=5 ans), son objectif est de produire des prévisions extrêmement précises (visant une erreur MAE globale < 1 000 migrants).  

## 📊 Données Utilisées

* **Flux Migratoires :** Estimations pseudo-bayésiennes (Azose & Raftery, 2019) basées sur les stocks mondiaux et l'équilibre démographique.  
* **Covariables Macroéconomiques :** Base de données Gravity (CEPII) enrichie. Intégration de variables géographiques (distance, frontières) et socio-économiques (Population, PIB et ses retards, Mortalité Infantile, Ratio de Soutien Potentiel). 

## 🚀 État d'Avancement et Découvertes Récentes

Nous avons récemment concentré nos efforts sur le modèle **ARX Hurdle Bayésien**, échantillonné via Hamiltonian Monte Carlo (Stan) :

* **Succès de l'architecture "Hurdle" :** Le modèle excelle dans la prédiction de l'ouverture ou de la fermeture des routes migratoires (Accuracy > 96%). Les derniers % restants sont des *cygnes noirs*, imprévisibles. L'idée d'estimer l'inertie *par continent* plutôt que *globalement* a été un succès: par exemple, dans l'espace Schengen, le modèle comprend qu'une route ouverte reste ouverte. Il est en revanche plus souple sur la fermeture éventuelle d'un couloir précédemment ouvert en Afrique ou en Asie.  
* **Gestion de la variance :** La modélisation de l'hétéroscédasticité par continent et les prédictions avec la médiane (minimiseur de la norme L1) ont permis d'empêcher l'explosion mathématique des prédictions sur les dyades instables. Sur un panel de test de 70 à 140 pays, les métriques d'erreur (MAE) et de *coverage* des intervalles de confiance sont très encourageantes.  
* **Le défi des micro-flux :** L'utilisation d'une loi continue (log-normale) se heurte mathématiquement à la nature discrète des micro-flux (couloirs de 1 à 10 personnes), générant un biais de variance. Cependant, ce bruit statistique inhérent aux bases de données n'impacte pas l'utilité du modèle : ces micro-flux ne sont pas pertinents d'un point de vue macroéconomique pour les décideurs publiques. On assume alors ces erreurs, sans vouloir simplement les supprimer *ou* implémenter un modèle ad-hoc destiné à les gérer.   

## ⏭️ Prochaines Étapes immédiates

* **Intégration des Chocs Géopolitiques :** Ajout de données de conflits (ex: base UCDP) pour casser l'inertie auto-régressive du modèle et mieux anticiper les crises migratoires soudaines. Pour le moment, le modèle montre ces limites en prédiction (OOS) sur 2015 à cause du manque d'anticipation des crises ayant lieu entre 2010 et 2015 (crise en Syrie, guerre civile, chute de Kadhafi en Libye...)  
* **Perfection du Hurdle :** Auditer les derniers % de précision (les cygnes noirs) pour tenter de viser les 99% d'Accuracy, les 96% de précision obtenues étant déjà assez spectaculaire sur tant de dyades variées.  
* **Scale-up Mondial :** Lancement de l'inférence HMC sur la matrice mondiale complète (190 pays) via le cluster de calcul Onyxia (GENES). Cette mise à l'échelle devrait mécaniquement écraser notre MAE globale et nous positionner au-delà de l'état de l'art actuel, qui ne dispose pas d'explication économétrique des chocs, et est davantage focalisé sur la prédiction de long-terme.

  
***Auteurs***

Projet réalisé dans le cadre du cours de Statistique Appliquée (ENSAE) par :
Louise, Romain, Ishagh, Varnel


*Dernière mise à jour : 28 Mars 2026*

