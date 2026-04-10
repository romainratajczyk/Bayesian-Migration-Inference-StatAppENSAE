
# Prédiction bayésienne des flux migratoires internationaux 
[Dépôt GitHub - Projet Migration](https://github.com/IshaghCheikh/ProjetStat/tree/main)

Ce projet vise à comprendre la dynamique des flux migratoires internationaux et d'en prédire l'évolution. Face à la complexité de la réalité macroéconomique et statistique des flux, nous déployons une méthodologie progressive : d'un modèle de gravité standard vers des algorithmes de Machine Learning, pour aboutir à une modélisation bayésienne hiérarchique de pointe. L'enjeu est de capturer l'inertie temporelle, l'hétéroscédasticité, et les chocs macro-démographiques et géopolitiques. 

## 🔬 Notre démarche scientifique 


1. **Benchmark Modèle de Gravité :** Modèle de gravité log-linéaire (OLS) pour capturer les déterminants standards (distance, PIB, liens coloniaux). Biais de spécification majeur (la réalité n'est pas du tout linéaire, ce que les prochains modèles vont démontrer); ce modèle n'est pas optimal.
2. **Exploration Non-Linéaire (ML) :** Méthodes ensemblistes (Random Forest, XGBoost) pour challenger la linéarité du modèle de gravité. 
   * Détection d'effets de seuil et d'interactions complexes entre variables. (e.g., seuil sur le PIB, interaction distance*frontière_commune)
   * Analyse des cartes de résidus (comprendre géographiquement où le modèle se trompe).
   * Extraire les *feature importances*.
3. **Inférence Bayésienne Hiérarchique (Stan/HMC) :** L'analyse des cartes de résidus des modèles ML suggèrent une forte hétéroscédasticité géographique, ce que notre modèle hiérarchique sait très bien gérer. Les effets de seuil découverts par le ML injectés dans une équation de gravité dictant la valeur de la moyenne d'un AR(1) propre à chaque paire de pays (i,j). Le but est d'améliorer les prédictions Out-of-Sample et les métriques d'erreur (MAE,MAPE).

## 🎯 Deux approches pour deux horizons

L'objectif in fine est de doter les décideurs publics d'un outil de prévision complet, reposant sur deux modèles complémentaires :

* **Baseline Long Terme (Modèle de Welch & Raftery, voir /articles) :**  Réplication du modèle de référence OutFlow/Allocation. La méthodologie repose sur le calcul d'un taux de départ global par pays d'origine, dont le volume est ensuite réparti dans le monde via une distribution multinomiale. Ce modèle n'utilise aucune variable économétrique, seulement les masses de population. Cela lui permet des projections de très longue durée (2050, 2100 et au-delà en théorie).  
* ** Notre Modèle sur-mesure ARX Hurdle, prévisions court terme  :** Modèle bayésien à plusieurs composantes hiérarchiques, hautement réactif à l'économétrie et préparé aux chocs macro-démographiques et géopolitiques. Pensé pour la précision à court terme (<=5 ans), son objectif est de produire des prévisions extrêmement précises, surpassant l'état de l'art actuel pour les prévisions de long terme (Welch & Raftery). Il a déjà démontré une erreur MAE (norme L1) meilleure que celle de Welch & Raftery (980<1200) et est en cours d'amélioration, nos pistes sont très encourageantes. (voir Annexe Technique)

## 📊 Données Utilisées

* **Flux Migratoires :** Estimations bayésiennes (JAGS) (Azose & Raftery, 2019) basées sur les stocks mondiaux et l'équilibre démographique.  
* **Covariables Macroéconomiques :** Base de données Gravity (CEPII) enrichie. Intégration de variables géographiques (distance, frontières), socio-économiques (Population, PIB et ses retards, Mortalité Infantile, Labour Force), géopolitiques. 

## 🚀 État d'Avancement et Découvertes Récentes

Nous avons récemment concentré nos efforts sur nos deux modèles **bayésiens**, échantillonnés via Hamiltonian Monte Carlo (Stan) :

* La réplication fidèle du modèle Outflow/Allocation de Welch & Raftery est sur la bonne voie, il nous reste à bien définir les priors pour répliquer leurs métriques d'erreur (eux utilisent de l'Empirical Bayes pour la définition des priors);

* ### Pour le modèle sur-mesure de court terme
* Le précédent modèle répliqué de celui de Welch & Raftery nous sert de baseline: il est simple, respecte le Rasoir d'Ockham, et prédit correctement les flux. On cherche alors à complexifier intelligement le modèle, pour faire mieux (au court terme). 
* **Succès de l'architecture "Hurdle" :** Le modèle excelle dans la prédiction de l'ouverture ou de la fermeture des routes migratoires (avec un Logit, décision dure ajustée avec un seuil ROC, Accuracy > 96%). Les derniers % restants sont des *cygnes noirs*, imprévisibles. L'idée d'estimer l'inertie *par continent* plutôt que *globalement* a été un succès: par exemple, dans l'espace Schengen, le modèle comprend qu'une route ouverte reste ouverte. Il est en revanche plus souple sur la fermeture éventuelle d'un couloir précédemment ouvert en Afrique ou en Asie.  
* **Excellentes premières métriques Out-of-Sample :** La modélisation de l'hétéroscédasticité par régions M49 de l'ONU (subdivisions M49: Europe du Nord, Europe de l'Est, etc.) est un succès, le modèle comprends bien la différence de stabilité dans le monde et ajuste ses inteervalles de crédibilité en conséquence. Les prédictions avec la médiane (minimiseur de la norme L1, adapté pour la MAE) ont démontré, sur un panel de 171 pays (199 états à suivre), des métriques d'erreur (MAE) et un coverage battant déjà les métriques de Welch & Raftery (notre MAE globale est pour l'instant de 980 et devrait encore mécaniquement se rétrécir avec l'ajout des derniers états, et notre coverage est de 97% pour des intervalles de crédibilité à 95%, coverage qu'on cherchera à maintenir sur l'échantillon complet).
* **Le défi des micro-flux :** Notre première version, qui utilisait une loi continue (log-normale), se heurte mathématiquement à la nature discrète des micro-flux (couloirs de 1 à 10 personnes). Cependant, ce bruit statistique inhérent aux bases de données n'impacte pas l'utilité du modèle : ces micro-flux ne sont pas pertinents d'un point de vue macroéconomique pour les décideurs publiques. Nous n'avons pas envisagé d'implémenter un modèle ad-hoc destiné à les gérer spécifiquement.
* **Avancée la plus récente:** En revanche, nous avons substitué la loi log-normale par une binomiale négative tronquée à zéro (très bien adaptée a priori, car son paramètre de dispersion modélise bien l'hétéroscédasticité via la variance; et une loi de Poisson imposerait Var(Y)=E(Y) ce qui est largement faux pour nos flux).  Nous avons aussi changer de paradigme pour les paramètres dyadiques (effet d'émission + attraction + effets de gravités dyadiques pour éviter l'overfitting du modèle qui était sur-paramétré, et absorber mathématiquement les variables omises dans les coefficients propres à chaque pays). Ajout d'hyper-regressions économétriques pour remplacer des priors faiblement informatifs qui généraient des prédictions aberrantes sur des flux manquant de données (car leurs paramètres étaient shrinkés vers la moyenne continentale, insuffisant). 

## ⏭️ Prochaines Étapes immédiates

* **Intégration des Chocs Géopolitiques :** Ajout de données de conflits (ex: base UCDP) pour casser l'inertie auto-régressive du modèle et mieux anticiper les crises migratoires soudaines. Pour le moment, le modèle montre ces limites en prédiction (OOS) sur 2015 à cause du manque d'anticipation des crises ayant lieu entre 2010 et 2015 (crise en Syrie, guerre civile, chute de Kadhafi en Libye...)  
* **Perfection du Hurdle :** Auditer les derniers % de précision (les cygnes noirs) pour tenter de viser >96% d'Accuracy, les 96% de précision obtenues étant déjà excellentes sur tant de dyades variées.  
* **Scale-up Mondial :** Lancement de l'inférence HMC sur la matrice mondiale complète (199 pays) via le cluster de calcul Onyxia (GENES). Cette mise à l'échelle devrait mécaniquement écraser notre MAE globale et nous positionner sur toutes les métriques au-delà de l'état de l'art actuel, qui ne dispose pas d'explication économétrique des chocs, et est davantage focalisé sur la prédiction de long-terme.
* Comparaison des modèles avant/après l'ajout de nouvelles pistes uniquement avec les critères MAE & MAPE. En effet, les critères BIC/AIC ne sont pas adaptés à la haute dimension de notre espace bayésien, et pénalisent injustement le nombre de nos paramètres (qui sont hiérarchiques! leur poids effectif n'est pas de 1). Aussi, le test du critère PSIS-LOO de Gelman et al. (2024) s'est avéré inadapté: la structure AR(1) et l'interdépendance mondiale implique que $P(Y | \theta) \neq P(y_i | \theta) P(Y_{-i} | \theta)$, hypothèse cruciale du critère PSIS-LOO, et ce dernier tend aussi à favoriser des modèles sur-paramétrés. Il nous avait alors premièrement induit en erreur sur un modèle qui présentait un subtil overfitting. 
  
# Annexe technique : Bayesian Hierarchical ARX Hurdle Model (notre modèle de prédiction court-terme)

Cette section détaille l'architecture mathématique et les choix d'inférence de notre modèle bayésien. Pour ceux qui souhaitent comprendre le moteur interne de notre code Stan et la méthodologie de prédiction.

### 1. Architecture en deux étapes (Hurdle-Volume)

Le modèle traite la migration bilatérale en deux étapes séquentielles pour contourner la double difficulté des flux nuls (49% du dataframe) et de la forte variance des grands couloirs.

#### A. Composante Hurdle (Proba d'Ouverture de la route)
Régression logistique (Bernoulli) estimant la probabilité qu'un flux migratoire strictement positif existe entre les pays $i$ et $j$.

$$\text{logit}(P(\text{flow} > 0)) = \alpha_{d} + X_{h} \beta_{h} + \beta_{\text{lag}} \text{is\\_mig\\_lag}$$

Où $X_{h}$ inclut les variables les plus importantes et pertinentes pour le Hurdle (notamment les features les plus importantes indiquées par un Random Forest entraîné) : frontière commune, $\log(\text{distance})$, PIB/tête à la date $t-1$, populations... Sans pour autant répliquer complètement le modèle de gravité (le but est l'*existence ou non* d'une route, pas son *volume*). Si le modèle prédit une fermeture, le flux prédit est 0 net. S'il prédit une ouverture, on passe à la composante Volume.

#### B. Composante Volume (Processus ARX Log-Normal)
AR "X" pour "eXogenous variables", les variables économétriques du modèle de gravité pour $$\mu$$.   
Pour les dyades actives, le volume est modélisé par un processus auto-régressif conditionnel à la dyade :

$$\log(\text{flow}) \sim \mathcal{N}(\mu_{d,t} + \phi_{d} (\text{lag} - \mu_{d,t-1}), \sigma_{d})$$

L'espérance de base $\mu_{d,t}$ intègre les variables gravitaires classiques et les variables non-linéaires découvertes lors de la phase d'exploration par Machine Learning :

$$\mu_{d,t} = \alpha_{V,d} + X \beta_{\text{grav}} + \beta_{\text{gdp}} \log(\text{gdpcap\\_o}) + \beta_{\text{rich}} \text{is\\_rich\\_o}$$

*(Note : `is_rich_o` encode un effet de seuil détecté par Random Forest autour de 18 000 $ de PIB/habitant à partir duquel l'émigration augmente brusquement pour le pays d'origine).*

### 2. Inférence par Hamiltonian Monte Carlo (HMC) avec Stan

Contrairement aux approches par échantillonnage de Gibbs (JAGS) ou marche aléatoire aveugle (Metropolis), l'utilisation de Stan (HMC) est cruciale ici pour explorer un espace de paramètres de très haute dimension (~90 000 dimensions) sans rester piégé.

**Le paysage énergétique et la mécanique hamiltonienne**
L'espace des postérieurs bayésiens est analogue à un paysage énergétique en physique où la log-vraisemblance définit l'énergie potentielle (les "puits" sont les zones de forte probabilité ici). À chaque itération $s$ :
1. L'algorithme reçoit une impulsion cinétique aléatoire.

2. Il simule une trajectoire déterministe le long du gradient de probabilité via les équations de Hamilton. Au moment d'une micro-étape (itération $s$), le moteur Stan fait concrètement ceci :
   * Il prend les valeurs `raw` tirées du bruit (des priors qui peuvent être faiblements informatifs, ou légèrement calibrés par Empirical Bayes) et les multiplie par les $\tau$ pour construire l'état de chaque couloir : $\alpha_{V,d}^{(s)}$, $\phi_{d}^{(s)}$ et $\sigma_{d}^{(s)}$.
   * Il assemble tout ça avec les variables géoéconomiques ($X$, PIB, etc.) pour calculer le $\mu_{d,t}^{(s)}$.
   * Puis, il utilise cette valeur pour évaluer la distance par rapport aux vrais flux via la loi Volume :

   $$\log(\text{flow}) \sim \mathcal{N}(\mu_{d,t}^{(s)} + \phi_{d}^{(s)} (\text{lag} - \mu_{d,t-1}^{(s)}), \sigma_{d}^{(s)})$$

3. À la position d'arrivée, Stan évalue l'acceptation via Metropolis-Hastings en vérifiant la conservation de l'énergie totale ($H$) :

   $$P(\text{acceptation}) = \min(1, \exp(-\Delta H))$$

Si la position est cohérente ($\Delta H \approx 0$), les paramètres sont acceptés et inscrits dans les chaînes de Markov.

**Stabilité géométrique (Non-centered parameterization)**
Pour éviter les géométries en entonnoir qui font diverger/bloquent les chaînes de Markov, le modèle hiérarchique est codé via une paramétrisation décentrée (*transformed parameters*). Stan ne tire pas directement dans la loi normale de la dyade, il tire un bruit pur (`raw`) qu'il multiplie par la variance du cluster ($\tau$) :
* **Intercept dyadique :**

  $$\alpha_{V,d} = \mu_{\text{intercept}} + \tau_{\mu} \times \mu_{\text{raw}}[d]$$

* **Inertie AR1 :**

  $$\phi_{d} = \tanh(\phi_{\text{global} \_ \text{raw}} + \tau_{\phi} \times \phi_{\text{raw}}[d])$$

* **Variance hétéroscédastique :**

  $$\sigma_{d} = \sigma_{\text{cluster}} \times \exp(\tau_{\sigma} \times \sigma_{\text{raw}}[d])$$
  
### 3. Méthode de prédiction

Une fois l'inférence terminée, les matrices de paramètres (ex: 1200 itérations conservées, entraînement sur 1990-2010) sont extraites. NumPy prend la relève pour vectoriser les équations sur les données hors-échantillon (ex: test sur 2015).

**Le choix de la Médiane vs l'Espérance**
Dans un modèle log-normal, l'espérance mathématique est $\exp(\mu + \sigma^2 / 2)$. Sur des couloirs instables (comme MEX-USA), il a été observé un grand $\sigma_{d}$ amplifié par l'inflation auto-régressive $(1+\phi^2)$ ce qui a propulsé les prédictions à des valeurs absurdes (ex: 25 millions de migrants) en tentant de minimiser la *Mean Squared Error* (MSE).

Or, l'objectif macroéconomique et décisionnel est de minimiser l'erreur absolue en nombre d'humains, pas en humains au carré. Ainsi nous extrayons la médiane $\exp(\mu)$ de nos matrices de prédiction, qui est le minimiseur naturel de la norme L1 (MAE).

### 4. Choix méthodologiques et Discussion

* **Synergie ML $\rightarrow$ Bayésien :** Le modèle bayésien n'est pas construit à l'aveugle. Il intègre directement les enseignements de nos modèles XGBoost et Random Forest : effets de seuils sur le PIB, interactions spatiales validées par PDP ($\log(\text{Distance}) \times \text{Frontière}$), et hétéroscédasticité géographique modélisée au niveau continental pour absorber les résidus systématiques détectés en Afrique et en Asie (sur des cartes de résidus mondiales, cf `challenge_gravity_ML.ipynb`).
* **Le problème des zéros :** L'approche Hurdle a été préférée à la transformation $\log(x+1)$ (qui est scientifiquement instable). Forcer une loi normale continue à gérer un pic massif à zéro provoque une divergence de la variance temporelle. Le Hurdle isole le problème structurellement.
* **Évaluation (OOS) :** Entraîné sur la période 1990-2010 et testé sur 2015. Nous utilisons le WMAPE, et la MAPE modifiée de Welch & Raftery (divisée par $y+1$) pour un benchmark fidèle face à la littérature (Welch & Raftery). La couverture spatiale des intervalles de crédibilité (IC) bénéficie beaucoup de l'hétéroscédasticité : étroits en Europe (+/- 30%), ils s'élargissent logiquement sur les couloirs volatiles d'Asie et d'Afrique (+/-150%).
* **La limite des micro-flux :** Le modèle présente un biais théorique inhérent à la loi log-normale sur les flux continus de 1 à 10 migrants. Si un modèle de comptage (ex: Negative Binomial) traiterait mieux ces micro-flux, l'ajout d'un modèle pour les flux intermédiaires nous semble trop *ad-hoc* et perturberait certainement la stabilité de nos simulations. Surtout, ces micro-flux sont macro-économiquement non pertinents et résultent de bruit statistique : on assume alors que notre modèle n'est pas adapté à la prédiction sur les micro-flux.

### 5. Dimensions de l'espace des paramètres  

L'inférence simultanée repose sur une très-haute-dimension (pour 190 pays) :
* **Partie Hurdle ($D_{h}$) :** $\sim 35\ 000$ dimensions ($\alpha_{\text{raw}}$ par dyade).
* **Partie Volume ($D_{v}$) :** Environ 50% des dyades sont actives. Chacune requiert un $\mu_{\text{raw}}$, un $\phi_{\text{raw}}$ et un $\sigma_{\text{raw}}$, soit $\sim 53\ 000$ dimensions.
* **Paramètres globaux & Clusters :** Vecteurs $\beta_{h}$ (3 variables), $\beta_{\text{grav}}$ (~20 variables), variances par continent (6 dimensions), et hyper-paramètres globaux ($\mu$, $\tau$).

**Total : $\sim 90\ 000$ dimensions explorées simultanément par Hamiltonian Monte Carlo.**
*Estimation RAM : 50-64 Go pour être très confortable et robuste aux pics et aux "Silent Kills" du cluster Onyxia-GENES. Plus de 128 Go nécessaires pour extraire TOUTES les variables samplées par Stan (pour 190 pays), le code actuel ne retire que celles importantes pour les prédictions.*    


***Auteurs***

Projet réalisé dans le cadre du cours de Statistique Appliquée (ENSAE) par :
Louise, Romain, Ishagh, Varnel


*Dernière mise à jour : 28 Mars 2026*

