***Bayesian Prediction of Migration Flows*** [Dépôt GitHub - Projet Migration](https://github.com/IshaghCheikh/ProjetStat/tree/main)

Ce projet a pour objectif de modéliser et prédire les flux migratoires internationaux bilatéraux. Il s'appuie sur une réplication du modèle de gravité de Welch et Raftery (2022) comme point de référence, pour ensuite explorer des approches plus complexes (modèles bayésiens hiérarchiques (MCMC), Machine Learning (Random Forests, XGBoost) ).

***Objectifs du Projet***

**Réplication** (Benchmark) : Implémenter le modèle de gravité log-linéaire standard pour comprendre les déterminants classiques (distance, population, liens coloniaux, etc.).
**Gestion des Données Manquantes** : Résoudre le problème des flux nuls ($log(0)$) et des discontinuités dans les séries temporelles (ex: PIB, taux de mortalité infantile).
**Comparaison de Modèles**: 
Modèle Gravitaire (OLS). Modèle Poisson Hurdle (pour gérer les zéros). Approche Bayésienne Hiérarchique (Azose & Raftery) pour capturer l'inertie migratoire et améliorer la prédiction. Machine Learning (Random Forest,XGBoost) pour tester la non-linéarité (en cours). 

***Données Utilisées***

**Flux Migratoires ($m_{ijt}$)** : Estimations pseudo-bayésiennes par Azose & Raftery (2019), basées sur les stocks de migrants et l'équation d'équilibre démographique.
**Covariables (Variables explicatives)** : Base de données Gravity du CEPII. Variables géographiques (Distance, Contiguïté). Variables socio-économiques (Population, PIB, Mortalité Infantile - IMR, Ratio de Soutien Potentiel - PSR). 

**Réplication du Modèle Gravitaire** : $R^2$ de 0.49 obtenu sur les données incomplètes, complétion des données en cours (finalisation prévue semaine du 16 Février.)   
**Enrichissement des Données (En cours)** :Intégration robuste du PIB (GDP) et du PIB par habitant (lags inclus) depuis la base CEPII.   Correction des problèmes de chargement de données (récupération des pays manquants comme la Nouvelle-Zélande ou les Pays-Bas, dont l'absence de donnée est absurde, et doit résulter d'un bug technique). Stratégie de "Rectangularisation" pour conserver les flux nuls (zéros) dans le dataset d'entraînement.   
 

Modèles de Machine Learning (RF & XGBoost) : Utilisés pour explorer et capter les effets non-linéaires complexes (ex: effet seuil du PIB, interaction distance/frontière commune) afin d'informer et d'améliorer nos équations économétriques.  

# ***État d'Avancement (Current Status)*** 

## **Modélisation Bayésienne (via Stan (Hamiltonian Monte Carlo):**  

Deux approches complémentaires répondant à deux objectifs différents :  


Le modèle Outflow/Allocation (Réplication Welch & Raftery 2022) : Une approche macro-démographique très inertielle. S'il est moins précis pour capter les chocs de court terme, c'est le modèle taillé pour les projections de très longue durée (2050, 2100 et au-delà en théorie).  


Notre modèle ARX Hurdle (Approche purement dyadique) : Un modèle bilatéral de gravité, pensé pour la précision à court/moyen terme. Il comprend bien les variables économétriques (PIB, géographie) pour expliquer pourquoi les migrants partent vers une destination précise, et est capable d'anticiper des chocs socio-économiques et démographiques.   


Victoires récentes : Succès de la composante "Hurdle" pour prédire l'ouverture/fermeture des routes (Accuracy > 96%), modélisation de l'hétéroscédasticité par continent pour des intervalles de confiance réalistes, et utilisation de la médiane (norme L1) pour empêcher l'explosion mathématique des prédictions sur les dyades instables.  
Erreurs MAE et Coverage très encourageants sur seulement 70 pays, simulations en cours sur 190 pays. 


### Prochaines étapes (Perspectives) :  


Chocs géopolitiques : Intégration de variables de conflits (ex: base UCDP) dans le modèle dyadique pour casser la "paresse" auto-régressive et mieux anticiper les crises soudaines de notre année de test (2015).  


Perfectionnement du Hurdle : Régionaliser l'inertie des routes (un paramètre par continent au lieu d'un global) pour frôler les 98-99% de précision sur la détection des flux non-nuls.  


Scale-up mondial (Onyxia - GENES) : simulation HMC d'envergure sur 190 pays via le cluster Onyxia EN COURS. On devrait mécaniquement tirer notre erreur absolue (MAE) vers zéro pour venir battre l'état de l'art.  


***Auteurs***

Projet réalisé dans le cadre du cours de Statistique Appliquée (ENSAE) par :
Louise, Romain, Ishagh, Varnel


*Dernière mise à jour : Mars 2026*
