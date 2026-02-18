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

***État d'Avancement (Current Status)***

**Réplication du Modèle Gravitaire** : $R^2$ de 0.49 obtenu sur les données incomplètes, complétion des données en cours (finalisation prévue semaine du 16 Février.) 
**Enrichissement des Données (En cours)** :Intégration robuste du PIB (GDP) et du PIB par habitant (lags inclus) depuis la base CEPII. Correction des problèmes de chargement de données (récupération des pays manquants comme la Nouvelle-Zélande ou les Pays-Bas, dont l'absence de donnée est absurde, et doit résulter d'un bug technique). Stratégie de "Rectangularisation" pour conserver les flux nuls (zéros) dans le dataset d'entraînement. 
**Prochaines étapes** : Entraînement du Random Forest, du XGBoost et du modèle Bayésien sur la base complète, dans le but de battre l'erreur relative de prédiction des auteurs.  

***Auteurs***

Projet réalisé dans le cadre du cours de Statistique Appliquée (ENSAE) par :
Louise, Romain, Ishagh, Varnel


*Dernière mise à jour : Février 2026*
