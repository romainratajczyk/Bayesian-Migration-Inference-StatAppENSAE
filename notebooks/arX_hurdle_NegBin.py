#!/usr/bin/env python
# coding: utf-8

# 
# # Bayesian Hierarchical ARX Hurdle Model for Gravity Migration
# 
# #### Brouillon (daté du 26 mars) pour présenter la méthode générale, les paramètres, la hiérarchie, l'hétéroscédasticité, quelques résultats. 
# ### Une version propre sera disponible la semaine prochaine, qui s'intègrera au rapport.
# 
# 
# 
# 
# **A. Hurdle (Logit)** : 
# $\text{logit}(P(\text{flow}>0)) = \alpha_d + X_h \beta_h + \beta_{lag} \text{is\_mig\_lag}$
# 
# **X_h**= (frontière_commune_ij, log(distance_ij) ).  
# **B. Volume (ARX)** : 
# $$\log(\text{flow}) \sim \mathcal{N}(\mu_{d,t} + \phi_d (\text{lag} - \mu_{d,t-1}), \sigma_d)$$
# $$\mu_{d,t} = \alpha_{V,d} + X \beta_{\text{grav}} + \beta_{\text{gdp}} \log(\text{gdpcap\_o}) + \beta_{\text{rich}} \text{is\_rich\_o}$$
# 
# **X**: toutes les variables du modèle de gravité de Welch&raftery.  
# **is_rich_o:** est ce que le pays de départ dépasse le seuil de 18,000$ de PIB/tête (seuil détecter par Random Forest)  
# 
# 
# # Hamiltonian Monte Carlo: 
# (absolument crucial, descendre une pente est bien plus rapide à converger qu'un Metropolis aveugle)
# 
# - Le paysage énergétique est un espace de paramètres postériors, chaque position est un vecteur de paramètres (de dimension 90 000 environ, voir partie Paramètres ci-dessous). Si on travaille en -log-vraisemblance: les puits sont les zones de fortes probabilités. 
# 
# - À chaque itération $s$ (de 1 à iter_sampling): Impulsion aléatoire, puis la fin du mouvement régit par les équations de Hamilton jusqu'à la position x. 
# 
# - A la position x, Stan possède un set de paramètres. Il calcule alors mécaniquement :$$\mu_{d,t}^{(s)} = \alpha_{V,d}^{(s)} + X \beta_{\text{grav}}^{(s)} + \beta_{\text{gdp}}^{(s)} \log(\text{gdpcap\_o}) + \beta_{\text{rich}}^{(s)} \text{is\_rich\_o} $$
# 
# 
# Modèle hiérarchique hétérosced: (dans *transformed parameters*)
# 
# $\alpha_{V,d} = \mu_{intercept} + \tau_{\mu} \times \mu_{raw}[d]$ *(intercept: moyenne sur le meme cluster)*
# 
# $\phi_d = \tanh(\phi_{global\_raw} + \tau_{\phi} \times \phi_{raw}[d])$ *(raw: chaque couloir possède son raw unique, son ADN, générée d'un prior)*  
# Rq: ne pas laisser Stan tirer de mu_d ~ normal(mu_intercept, tau_mu) car il resterait bloqué si tau_mu proche de zéro )
# 
# $\sigma_d = \sigma_{cluster}[continent] \times \exp(\tau_{\sigma} \times \sigma_{raw}[d])$
# 
# Stan prend les raw tirés du bruit et les multiplie par les $\tau$ pour construire l'état de chaque couloir : $\alpha_{V,d}^{(s)}$, $\phi_d^{(s)}$ et $\sigma_d^{(s)}$.
# Il assemble tout ça avec les variables géoécononomiques ($X$, PIB, etc.) pour calculer le $\mu_{d,t}^{(s)}$.
# 
# 
# Puis, il utilise cette valeur pour évaluer la distance par rapport aux vrais flux via la loi Volume:
# $$\log(\text{flow}) \sim \mathcal{N}(\mu_{d,t}^{(s)} + \phi_d^{(s)} (\text{lag} - \mu_{d,t-1}^{(s)}), \sigma_d^{(s)}) $$
# 
# L'acceptation (Metropolis-Hastings) à la fin du mouvement: Stan vérifie si l'énergie totale a été conservée. Il applique la règle d'acceptation :
# $$P(\text{acceptation}) = \min(1, \exp(-\Delta H))$$
# (isomorphisme entre conservation de l'énergie et maximisation de la proba a posteriori plutôt, càd vraisemblance + priors. C'est un compromis entre ce que disent les données et les priors)
# 
# Si la nouvelle position est cohérente avec les données ($\Delta H =0$) , il l'accepte et inscrit les paramètres dans des matrices.  
# Sinon, il rejette la proposition et reste sur la valeur précédente.
# 
# **paramètres globaux:** vecteurs de 1200 composantes à la fin du sampling
# - mu : matrice qui contient les log(flow), 1200* nombre de couloirs
# - sigma_cluster : dimension 1200x6
# - beta_grav: 1200x20 (20 variables explicatives)
# - effets dyadiques: matrices de 1200*nombres de couloirs
# 
# 
# 
# **C. Variance (Geo)** : 
# $\sigma_d \sim \text{HalfNormal}(\sigma_{\text{cluster}}[\text{continent\_origine}[d]])$ *(alternative à InverseGamma)*
# 
# 
# # Prédiction : 
# 
# une fois stocké toutes les matrices de paramètres, numpy prend la relève et calcule bêtement toutes les formules pour chaque itération
# (par ex $$\mu_{d,t}^{(1)} = \alpha_d^{(1)} + X \beta^{(1)}$$ pour l'itération s=1). Il fait ça pour les 1200 itérations, pour chaque couloir. 
# 
# **On a donc chains * iter / thin * dyades prédictions.**  
# **On prend la médiane de ces prédictions pour chaque couloir, pour minimiser l'erreur MAE.**
# 
# # MÉTHODOLOGIE 
# *(pour rapport ou annexe)*
# 
# 1) Couplage entre bayésien & Machine Learning (Partie ARX et Variance Géo).  
# Ce modèle bayésien intègre les découvertes faites par le Random Forest :
# - Saut brutal de migration autour de 18 000 $ de PIB/hab. 
#   Encodé par la variable indicatrice 'is_rich_o' 
# - Interaction 'log_D_ij * LB_ij' (distance * frontière commune) 
#   dont l'importance a été découverte par un PDP 2D du Random forest, et prouvée par régression linéaire 
# - Correction des résidus : La cartographie des erreurs des XGBoost & RF montrait une incertitude 
#   systématique (sous/sur-estimation) en Afrique, et un peu en Asie/Amerique latine. L'hétéroscédasticité 
#   géographique modélise cette variance propre à chaque continent (à affiner par zone géo plus précise?)
# 
# 
# 2) Gestion des zéros (partie Hurdle). 
# Le problème: il y a beaucoup de flux nuls, et on ne peut ni les enlever de l'analyse, ni faire log(x+1) (scientifiquement mauvais)
# Forcer un pic à zéro pour loi Normale (qui ne sait faire que une cloche, et pas une cloche + un pic à zéro) fait diverger 
# la variance et les chaines de Markov. 
# Le modèle Hurdle: regression logistique (Bernoulli); si et seulement si le couloir est ouvert (>0) => équation de gravité ARX. 
# Si non (flux=0) STAN s'arrête là et prédit 0 migrant (dans la phase de prédiction)
# 
# 
# 
# 3) intuition physique de STAN (Hamiltonian Monte Carlo). 
# Contrairement aux auteurs qui utilisaient le Gibbs sampling via JAGS, Stan utilise HMC. 
#  HMC utilise la mécanique hamiltonienne pour explorer le paysage des posteriors bayésiens, (trajectoire guidée par lmes équations de Hamilton)
# avec une étape d'acceptation Metropolis-Hastings à la fin selon $$P(\text{acceptation}) = \min(1, \exp(-\Delta H))$$ 
# pour corriger les erreurs numériques sur la conservation de l'énergie ($$\Delta H =0$$) liées à la discrétisation de temporelle. 
# 
# 
# Une exploration entière par Metropolis (marche aléatoire) aurait été inefficace et incroyablement lente pour autant de paramètres
# 
# 4) Stabilité géométrique.  
# Pour éviter que l'algorithme ne se coince (entonnoir), au lieu d'échantillonner 
# directement α_d ~ N(μ, τ), on échantillonne un bruit pur ε ~ N(0,1), puis on calcule 
#  α_d = μ + τ·ε. Cela détruit les corrélations pathologiques durant le HMC 
# 
# 5) Approche dyadique.  
# Mon modèle est purement "Dyadique" contrairement à celui de Ishagh (Inflow/Outflow). Ce code modélise chaque couloir de migration.  
# On pourra comparer les deux approches in fine. 
# 
# 6) Évaluation Out-Of-Sample.  
# Le modèle est entraîné sur la période 1990-2010 et testé en prédiction pure sur 2015. 
# Pour évaluer la qualité de la prédiction, on retient la MAE (Erreur absolue en nombre d'humains réels) et le MAPE comme Welch&raftery pour pouvoir comparer nos résultats   
# (**attention:** Welch&raftery divisent par y+1 leur erreur MAPE pour éviter la division par zéro, ce qu'on fait donc aussi)
# 
# # Commentaires de résultats
# **Médiane vs Espérance (Le problème des 25M) :**  
# Le modèle est évalué en MAE. L'espérance $exp(\mu + \sigma^2/2)$ minimise la MSE mais donne des prédictions délirantes quand la variance explose. (Stan gonfle la variance future avec l'inflation $1+\phi^2$ car il y a l'incertitude passée PLUS(+) l'incertitude nouvelle à considérer).  
# Un gros sigma donne vite une prédiction max absurde à 25 millions de migrants pour la route MEX-USA par ex. On utilise donc la médiane $exp(\mu)$ comme minimiseur naturel de la norme L1 (MAE). 
# 
# #### De toute façon, le choix le plus "économétrique (pour la décision publique)" pour des flux migratoires, c'est de s'intéresser à l'erreur en nombre de migrants (pas en carré de migrants).
# 
# 
# **Métrique ROC :**  
# Pour le seuil d'ouverture Hurdle, on utilise la courbe ROC plutôt que l'Accuracy pure. Le choix est arbitraire et les deux cas reviennent au même à 0,03% près de précision: en effet il n'y a pas de classe majoritaire dans nos données (49% de zéros). 
# 
# **Coverage & IC :**    
# L'hétéroscédasticité marche super bien ici. Pour un couloir européen stable, le modèle coupe les 2.5% extrêmes et donne un IC étroit (+/- 30%). Pour un couloir asiatique instable, ça s'écarte beaucoup plus (jusqu'à +150% de largeur). Le but ultime c'est que la vraie valeur tombe dans l'IC dans 95% des cas.
# 
# **Comparaison Welch & Raftery :**    
# En plus de la MAE et du Log-MAE (parfait pour les ordres de grandeurs), on suit le WMAPE et le "MAPE+1" (Eq 4 du papier de Welch) pour pouvoir faire un vrai benchmark face à eux sans que la division par zéro des petits couloirs ne fasse crasher le calcul.
# 
# 
# 
# # simulation 140 pays 27 mars: erreur MAE à 1300 (on a diviser l'erreur MAE sur 70 pays par 7 !)
# ## Gros problème de capacité prédictive des flux entre 1 et 10 (visible sur nuage de point). 
# Idée: 
# 
# $Y = 0$ $\rightarrow$ Bernoulli
# $Y \in [1, 10]$ $\rightarrow$ Modèle B
# $Y > 10$ $\rightarrow$ Log-Normal
# 
# ou alors trop *ad-hoc* ?
# 
# 
# # Paramètres: 
# 
# Partie Hurdle ($D_h$) : Un paramètre alpha_raw par dyade. Cela fait 190 * 189 dimensions. 
# Partie Volume ($D_v$) : mu_raw (l'intercepte du volume), phi_raw (l'inertie AR1 propre au couloir) et sigma_raw par dyade. Environ 50% des dyades ont du volume, donc 0.5 * 190 * 189 * 3 dimensions environ. 
# 
# - Gravité & Hurdle : Les vecteurs $\beta_{h}$ (3) et $\beta_{grav}$ (~20).
# - Hyper-paramètres : Les moyennes et variances globales (mu_intercept, sigma_global, phi_global, tau_alpha, tau_mu, tau_sigma, tau_phi).
# - Clusters : Les variances par continent sigma_cluster (6 dimensions).
# 
# 
# # Davantages de commentaires des résultats et de la méthode au fil du notebook, et en commentaire dans les cellules de code. 

# # changement de paradigme, réunion 26 Mar
# 
# - toujours mu_ij propre à chaque dyade, simplement on calcule maintenant alpha_i propre à chaque pays, beta_j propre à chaque pays, et on additionne mu_ij=alpha_i + beta_j (coeff émission + coeff attraction). On passe de 190*189 paramètres inconnus à 2*190.
# Pour une dyade vide, le modèle n'inventera plus un mu_ij absurde 
# 
# (modifier dans Stan les vecteurs de taille D (dyades) par des vecteurs de taille N_pays, modifier le bloc parameters, définir des priors, modifier l'equation dans model pour intégrer ces effets.)
# Dans python, modifier stan_data, au lieu de fournir un dyad_id il faudra un orig_id et dest_id, et Stan additionnera dans model 
# 
# - évaluer et comparer les modèles. AIC/BIC surestimeraient la complexité du modèle ? DIC trop simpliste ? PSIS-LOO est le standard moderne ?(Leave One Out cross validation, cf cours ML & Econometrics 1)
# Coût CPU: negligeable. Coût RAM/Disque colossal.
# Mettre la generation de log_lik de Stan avec interrupteur ==1 à mettre à 0 pour la production de figures et prédictions, et 1 pour la comparaison de modèles (lourds à simuler)

# In[10]:


# Installation des bibliothèques non classiqus
#!pip install pycountry_convert arviz cmdstanpy

# compilation de Stan
import cmdstanpy
cmdstanpy.install_cmdstan()


# # stratégie à faire le 27 mars: 
# 
# ### enrichissement du Hurdle en variables;
#  rechercher les "cygnes noirs" (les derniers 3,8% de precision du Hurdle).  
# 
# variables retenues: (le but n'est pas de mettre TOUTES les variables de gravité. Le Hurdle s'intéresse à l'*existence* du couloir, pas à son *volume*. Les variables les plus pertinentes: OL_ij et COL_ij (passé historique colonial et langue officielle commune) ; log_pop_d et log_pop_o (si les deux pays sont massifs, alors il y a certainement un flux) ) ; log_gdp_d et IMR (indice de richesse du pays d'arrivée).   
# 
# ### beta_lag_global à passer en continental; 
# 
# et montrer que chaque continent a un coeff très différent pour valider l'approche. 
# 
# 

# In[11]:


import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pycountry_convert as pc
from cmdstanpy import CmdStanModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve  
#from sklearn.preprocessing import StandardScaler
#import plotly.express as px

warnings.filterwarnings('ignore')
np.random.seed(42)


# In[12]:


# Chargement & filtrage pays




DATA_PATH = "../data/data_bayesian_arx_hurdle_runB.csv"

df_main = pd.read_csv(DATA_PATH)



# Sélecteur : 1 (30 pays), 2 (70 pays), 3 (110 pays), 4 (140 pays), 5 (199 pays / Full avant suppression des NaN)
CHOIX_ECHANTILLON = 5

# listes pré-définies 
L_30 = [
    'USA', 'FRA', 'DEU', 'GBR', 'CHN', 'IND', 'BRA', 'RUS', # Pôles gravitationnels
    'SYR', 'AFG', 'LBY', 'UKR', 'VEN', 'EGY', 'TUR', 'SDN', # Chocs UCDP/V-Dem critiques
    'YEM', 'COD', 'MLI', 'IRQ', 'MMR', 'SSD', 'SOM', 'BGD', 
    'PAK', 'ZAF', 'MAR', 'MEX', 'NGA', 'IDN'
]


L_70 = L_30 + [
    'JPN', 'CAN', 'THA', 'PHL', 'HUN', 'DZA', 'ZWE', 'KEN', 
    'CIV', 'SAU', 'ARE', 'ARG', 'CHL', 'ESP', 'ITA', 
    'ISR', 'IRN', 'MYS', 'KOR', 'AUS', 'CHE', 'SWE', 'NLD', 
    'BEL', 'POL', 'GRC', 'SEN', 'GHA', 'TZA', 'UGA', 'PER', 
    'RWA', 'ETH', 'LBN', 'COL', 'AGO', 'CMR', 'MOZ', 'ECU', 'URY'
]


L_110 = L_70 + [
    'DNK', 'FIN', 'IRL', 'CZE', 'ROU', 'BGR', 'HRV', 'SRB', 
    'ZMB', 'TCD', 'BFA', 'GIN', 'MDG', 'MWI', 'BDI', 'TGO', 
    'HTI', 'SLV', 'GTM', 'HND', 'PRY', 'NIC', 'PAN', 'UZB', 
    'JOR', 'LKA', 'NPL', 'KHM', 'OMN', 'CUB', 'DOM', 'CRI', 
    'BOL', 'SGP', 'KAZ', 'TUN', 'KWT', 'QAT', 'BEN', 'BWA'
]
# L_140 = L_110 + ['SVK', 'SVN', 'EST', 'LVA', 'LTU', 'ISL', 'CYP', 'LUX', 'ALB', 'BLR', 'BEN', 'SLE', 'LBR', 'MRT', 'CAF', 'COG', 'GAB', 'NAM', 'NER', 'JAM', 'TTO', 'BHS', 'BRB', 'BLZ', 'QAT', 'LAO', 'KWT', 'MNG', 'TJK', 'KGZ']  

if CHOIX_ECHANTILLON == 5:
    # Option 5 : Utilisation de la totalité du Df
    df = df_main[df_main['orig'] != df_main['dest']].copy()
else:

    map_listes = {1: L_30, 2: L_70, 3: L_110}#, 4: L_140}
    cible = map_listes[CHOIX_ECHANTILLON]

    df = df_main[
        df_main['orig'].isin(cible) & 
        df_main['dest'].isin(cible) & 
        (df_main['orig'] != df_main['dest'])
    ].copy()

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)
N_pays = df['orig'].nunique()
print(f"Extraction et simulation sur : {N_pays} pays.")




# In[13]:


# ── RUN B : exclusion des pays structurellement absents du train 1990-2010 ──
# Décommenter ce bloc pour Run B, commenter pour run full 200 pays.
# Pays exclus : n'existaient pas comme entités souveraines sur 1990-2010,
# ou sont des territoires sans données de flux cohérentes.
# Pays conservés par rapport au run 188 pays : ROU, SRB, COD, PSE.

PAYS_EXCLURE_RUN_B = {
    'SSD',  # Indépendance juillet 2011 — aucun flux modélisable en train
    'MNE',  # Indépendance juin 2006 — une seule période disponible (2010)
    'TLS',  # Indépendance 2002 — choc discontinu post-indépendance
    'CUW',  # Autonomie octobre 2010 — une seule période, insuffisant
    'GUM',  # Territoire américain, absent des flux Abel & Cohen
    'MYT',  # Territoire français, absent des flux Abel & Cohen
    'VIR',  # Territoire américain, absent des flux Abel & Cohen
    'CLI',  # Île Christmas, territoire australien minuscule
}

df = df[
    ~df['orig'].isin(PAYS_EXCLURE_RUN_B) &
    ~df['dest'].isin(PAYS_EXCLURE_RUN_B)
].copy()

pays_apres_exclusion = df['orig'].nunique()
print(f"Run B — après exclusion des pays instables : {pays_apres_exclusion} pays")
print(f"Pays exclus : {PAYS_EXCLURE_RUN_B}")

# Vérification PSE et COD effectivement présents
for pays in ['PSE', 'COD', 'ROU', 'SRB']:
    present = pays in df['orig'].unique()
    n_obs   = (df['orig'] == pays).sum()
    print(f"  {pays} : {'présent' if present else 'ABSENT'} — {n_obs} obs comme origine")

    # Vérification couverture GDP pour les pays récupérés
for pays in ['PSE', 'COD', 'ROU', 'SRB']:
    subset = df[df['orig'] == pays][['year', 'log_gdpcap_o_lag']].dropna()
    annees = sorted(subset['year'].unique().tolist())
    print(f"  {pays} — GDP lag non-NaN sur : {annees}")


# In[14]:


# Clustering géographique (EXOGENE au modèle et PUBLI: ISO-3166 alpha-3. Inattaquable)

# à réfléchir: clustering plus précis (sub-divisions ONU là encore public type Asie de l'Est, Asie du Sud...). très intéressant, et cite une source onusienne.
# Attention tout de même : beaucoup de sous-régions ONU, s'assurer que chaque sous région possède assez de dyade pour ne pas laisser le prior laissé à lui même. si pas assez e dyades, les fusionner en une plos grosse région, facilement défendable. 
# OU ALORS: laisser le modèle clusteriser par lui même (plus original)

"""

def get_continent_id(iso3_code):
    try:
        iso2 = pc.country_alpha3_to_country_alpha2(iso3_code)
        continent = pc.country_alpha2_to_continent_code(iso2)
        return {'EU': 1, 'NA': 2, 'AF': 3, 'SA': 4, 'AS': 5, 'OC': 6}.get(continent, 7)
    except Exception:
        return 7

df['continent_orig'] = df['orig'].apply(get_continent_id)
K_clusters = 6

"""



# In[15]:


# Clustering géographique: Sous-régions ONU (norme M49)

ISO3_TO_M49_SUBREGION = {
    #  Europe 
    'DNK': 11, 'EST': 11, 'FIN': 11, 'ISL': 11, 'IRL': 11, 'LVA': 11, 'LTU': 11, 'NOR': 11, 'SWE': 11, 'GBR': 11,
    'ALB': 12, 'AND': 12, 'BIH': 12, 'HRV': 12, 'GRC': 12, 'ITA': 12, 'MLT': 12, 'MNE': 12, 'MKD': 12, 'PRT': 12, 'SRB': 12, 'SVN': 12, 'ESP': 12,
    'AUT': 13, 'BEL': 13, 'FRA': 13, 'DEU': 13, 'LIE': 13, 'LUX': 13, 'MCO': 13, 'NLD': 13, 'CHE': 13,
    'BLR': 14, 'BGR': 14, 'CZE': 14, 'HUN': 14, 'POL': 14, 'MDA': 14, 'ROU': 14, 'RUS': 14, 'SVK': 14, 'UKR': 14,
    #  Afrique 
    'DZA': 15, 'EGY': 15, 'LBY': 15, 'MAR': 15, 'SDN': 15, 'TUN': 15, 'ESH': 15,
    'BEN': 16, 'BFA': 16, 'CPV': 16, 'CIV': 16, 'GMB': 16, 'GHA': 16, 'GIN': 16, 'GNB': 16, 'LBR': 16, 'MLI': 16, 'MRT': 16, 'NER': 16, 'NGA': 16, 'SEN': 16, 'SLE': 16, 'TGO': 16,
    'BDI': 17, 'COM': 17, 'DJI': 17, 'ERI': 17, 'ETH': 17, 'KEN': 17, 'MDG': 17, 'MWI': 17, 'MUS': 17, 'MOZ': 17, 'RWA': 17, 'SYC': 17, 'SOM': 17, 'SSD': 17, 'TZA': 17, 'UGA': 17, 'ZMB': 17, 'ZWE': 17,
    'AGO': 18, 'CMR': 18, 'CAF': 18, 'TCD': 18, 'COD': 18, 'COG': 18, 'GNQ': 18, 'GAB': 18, 'STP': 18,
    'BWA': 19, 'LSO': 19, 'NAM': 19, 'ZAF': 19, 'SWZ': 19,
    #  Amériques 
    'CAN': 21, 'MEX': 21, 'USA': 21,
    'BLZ': 22, 'CRI': 22, 'SLV': 22, 'GTM': 22, 'HND': 22, 'NIC': 22, 'PAN': 22,
    'ATG': 23, 'BHS': 23, 'BRB': 23, 'CUB': 23, 'DMA': 23, 'DOM': 23, 'GRD': 23, 'HTI': 23, 'JAM': 23, 'KNA': 23, 'LCA': 23, 'VCT': 23, 'TTO': 23, 'ABW': 23, 'PRI': 23,
    'ARG': 24, 'BOL': 24, 'BRA': 24, 'CHL': 24, 'COL': 24, 'ECU': 24, 'GUY': 24, 'PRY': 24, 'PER': 24, 'SUR': 24, 'URY': 24, 'VEN': 24,
    #  Asie 
    'CHN': 30, 'HKG': 30, 'JPN': 30, 'KOR': 30, 'MAC': 30, 'MNG': 30, 'PRK': 30,
    'AFG': 34, 'BGD': 34, 'BTN': 34, 'IND': 34, 'IRN': 34, 'MDV': 34, 'NPL': 34, 'PAK': 34, 'LKA': 34,
    'BRN': 35, 'KHM': 35, 'IDN': 35, 'LAO': 35, 'MYS': 35, 'MMR': 35, 'PHL': 35, 'SGP': 35, 'THA': 35, 'TLS': 35, 'VNM': 35,
    'ARM': 145, 'AZE': 145, 'BHR': 145, 'CYP': 145, 'GEO': 145, 'IRQ': 145, 'ISR': 145, 'JOR': 145, 'KWT': 145, 'LBN': 145, 'OMN': 145, 'QAT': 145, 'SAU': 145, 'PSE': 145, 'SYR': 145, 'TUR': 145, 'ARE': 145, 'YEM': 145,
    'KAZ': 143, 'KGZ': 143, 'TJK': 143, 'TKM': 143, 'UZB': 143,
    #  Océanie 
    'AUS': 53, 'FJI': 53, 'NZL': 53, 'PNG': 53, 'SLB': 53, 'VUT': 53, 'WSM': 53, 'TON': 53, 'KIR': 53, 'FSM': 53, 'GUM': 53, 'NCL': 53, 'PYF': 53,
}

SUBREGION_LABELS = {
    11: 'Europe du Nord', 12: 'Europe du Sud', 13: "Europe de l'Ouest", 14: "Europe de l'Est",
    15: 'Afrique du Nord', 16: "Afrique de l'Ouest", 17: "Afrique de l'Est", 18: 'Afrique Centrale', 19: 'Afrique Australe',
    21: 'Amerique du Nord', 22: 'Amerique Centrale', 23: 'Caraibes', 24: 'Amerique du Sud',
    30: "Asie de l'Est", 34: 'Asie du Sud', 35: 'Asie du Sud-Est',
    143: 'Asie Centrale', 145: "Asie de l'Ouest", 53: 'Oceanie', 99: 'Non classifie'
}

# Mapping M49 
df['m49_brut'] = df['orig'].map(lambda x: ISO3_TO_M49_SUBREGION.get(str(x).upper(), 99))
_UNIQUE_M49_PRESENT = sorted(df['m49_brut'].unique())
_M49_TO_STAN = {m49: i + 1 for i, m49 in enumerate(_UNIQUE_M49_PRESENT)}
stan_to_m49 = {v: k for k, v in _M49_TO_STAN.items()}

#  Application au DF (maintien du nommage 'continent_orig' pour compatibilité a l'ancien code)
df['continent_orig'] = df['m49_brut'].map(_M49_TO_STAN)
K_clusters = len(_M49_TO_STAN)

print(f"{K_clusters} clusters détectés")
#  Vérification des dyadiques (Uniquement sur couloirs ouverts)
SEUIL_FUSION = 30
df_actifs = df[df['flow'] > 0].copy()

# Création vectorielle temporaire pour le décompte (la vraie variable 'dyad' sera créée à la cellule suivante)
df_actifs['temp_dyad'] = df_actifs['orig'] + "_" + df_actifs['dest']

# Comptage des couloirs uniques
dyad_counts = df_actifs.groupby('continent_orig')['temp_dyad'].nunique().reset_index(name='n_dyades')
dyad_counts['label'] = dyad_counts['continent_orig'].apply(lambda i: SUBREGION_LABELS.get(stan_to_m49.get(i, 99), 'Inconnu'))
dyad_counts = dyad_counts.sort_values('n_dyades')

print("\nRépartition des dyades par cluster (K) :")
print(dyad_counts[['label', 'continent_orig', 'n_dyades']].to_string(index=False))

problematic = dyad_counts[dyad_counts['n_dyades'] < SEUIL_FUSION]
if not problematic.empty:
    print(f"\n[ALERTE] Clusters sous le seuil critique ({SEUIL_FUSION} dyades) :")
    print(problematic[['label', 'n_dyades']].to_string(index=False))


# à faire: renommer beta_lag_continental en beta_lag_m49 et revoir l'approche bayésienne hiéarachique sur beta_lag (beta_lag_raw et tau_beta_lag etc)
# 
# Régularisation rigide vers le prior si volume de dyades modérés, overfitting si trop peu de dyades. 
# La hiérarchie permet d'apprendre à partir de TOUTES les dyades. 

# In[16]:


# Features, lags et split train/test

df['is_migration'] = (df['flow'] > 0).astype(int)
df['log_flow']     = np.where(df['flow'] > 0, np.log(df['flow']), np.nan)
df['log_flow_lag'] = df.groupby(['orig', 'dest'])['log_flow'].shift(1)
SEUIL_LOG_GDP       = 2.9
df['is_rich_o']     = (df['log_gdpcap_o_lag'] > SEUIL_LOG_GDP).astype(float)
df['is_mig_lag']   = df.groupby(['orig', 'dest'])['is_migration'].shift(1)
df['log_D_ij']      = np.log(df['D_ij'].replace(0, np.nan))
df['logD_times_LB'] = df['log_D_ij'] * df['LB_ij']

df['dyad']          = df['orig'] + "_" + df['dest']

df = df.dropna(subset=['is_mig_lag']).reset_index(drop=True)
df['log_D_ij_sq'] = df['log_D_ij'] ** 2
HURDLE_VARS = [
    'log_D_ij',       # 1. Distance
    'log_D_ij_sq',
    'LB_ij',          # 2. Frontière commune
    'logD_times_LB',  # 3. Interaction
    'COL_ij',         # 4. Colonie
    'OL_ij', 'v2x_polyarchy_o_lag', 'v2x_clphy_o_lag', 'intensity_level_o_lag', 'type_of_conflict_o_lag',
    'v2x_polyarchy_d_lag', 'v2x_clphy_d_lag', 'intensity_level_d_lag', 'type_of_conflict_d_lag']#,          # 5. Langue officielle
    #'log_P_it',       # 6. Population Origine
    #'log_P_jt',       # 7. Population Destination
    #'log_gdpcap_d_lag'# 8. PIB Destination
#]

ML_VARS          = ['log_gdpcap_o_lag', 'is_rich_o'] # ne sert à rien pour l'instant, possible colinéarité, mais seuil réel détecté par random forest. A explorer plus tard. 
# pour que la boucle for génère les log_P_it pour le Hurdle
GRAVITY_VARS_RAW = ['P_it', 'P_jt', 'PSR_i', 'PSR_j', 'IMR_it', 'IMR_jt',
                    'urban_it', 'urban_jt', 'LA_i', 'LA_j']

# retrait de LL_i, LL_j, etc. pour préserver de la multicolinéarité du modèle ém+at
GRAVITY_VARS_BIN = ['LB_ij', 'OL_ij', 'COL_ij', 't_2000', 't_2000_sq'] 

for raw in GRAVITY_VARS_RAW:
    df[f'log_{raw}'] = np.log(df[raw].replace(0, np.nan))

# PURGE STRICTE Des effets monoadiqyes du  VOLUME (sauf si dimension temporelle, dans ce cas orthogonalité, pas de colinéarité parfaite possible)
X_VOL_COLS = [
    'log_D_ij', 'LB_ij', 'OL_ij', 'COL_ij', 't_2000', 't_2000_sq',
    'v2x_polyarchy_o_lag', 'v2x_clphy_o_lag', 'intensity_level_o_lag', 'type_of_conflict_o_lag',
    'v2x_polyarchy_d_lag', 'v2x_clphy_d_lag', 'intensity_level_d_lag', 'type_of_conflict_d_lag'
] 

K_grav, K_h = len(X_VOL_COLS), len(HURDLE_VARS)

df_train = df[df['year'] <= 2010].copy()
df_test  = df[df['year'] == 2015].copy()
df       = df_train 




# In[17]:


# Séparation hurdle / volume

HURDLE_REQUIRED = HURDLE_VARS + ['is_mig_lag', 'is_migration', 'dyad', 'continent_orig']
df_hurdle = df.dropna(subset=HURDLE_REQUIRED).copy().reset_index(drop=True)

# Remplacement de 'log_flow' par 'flow' pour la vraisemblance ZTNB
VOLUME_REQUIRED = X_VOL_COLS + ['flow', 'log_flow_lag', 'dyad', 'continent_orig']
df_volume = df[df['flow'] > 0].dropna(subset=VOLUME_REQUIRED).copy().reset_index(drop=True)

N_h, N_v = len(df_hurdle), len(df_volume)
print(f"Hurdle : {N_h:,} obs | Volume : {N_v:,} obs")


# In[ ]:


# Nettoyage exclusif de la covariable inertielle brute (sans centrage). Penalité AR1 dans metriques OOS prediction, le modèle ne voit jamais de couloirs fermés en t-1 en train
df_test['log_flow_lag_clean'] = (
    df_test['log_flow_lag']
    .fillna(0.0)
    .replace([np.inf, -np.inf], 0.0)
)


# In[19]:


# Encodage dyades et standardisation




dyades_h  = sorted(df_hurdle['dyad'].unique())
dyad_to_h = {d: i+1 for i, d in enumerate(dyades_h)}
df_hurdle['dyad_id_h'] = df_hurdle['dyad'].map(dyad_to_h)
D_h = len(dyades_h)
cluster_h = (df_hurdle.groupby('dyad')['continent_orig'].first()
             .reindex([k for k, v in sorted(dyad_to_h.items(), key=lambda x: x[1])])
             .values.astype(int))

dyades_v  = sorted(df_volume['dyad'].unique())
dyad_to_v = {d: i+1 for i, d in enumerate(dyades_v)}
df_volume['dyad_id_v'] = df_volume['dyad'].map(dyad_to_v)
D_v = len(dyades_v)
cluster_v = (df_volume.groupby('dyad')['continent_orig'].first()
             .reindex([k for k, v in sorted(dyad_to_v.items(), key=lambda x: x[1])])
             .values.astype(int))

BINARY_COLS_VOL = ['LB_ij', 'OL_ij', 'COL_ij'] 
# ('is_rich_o', 'LL_i', 'LL_j' ont été purgés car monadiques)
BINARY_COLS_HUR = ['LB_ij', 'COL_ij', 'OL_ij']

def standardize_matrix(X, col_names, binary_cols, fit_stats=None):
    X_std, stats = X.copy().astype(float), {}
    for j, col in enumerate(col_names):
        if col not in binary_cols:
            mu = X[:, j].mean() if fit_stats is None else fit_stats[col]['mean']
            sd = X[:, j].std()  if fit_stats is None else fit_stats[col]['std']
            sd = max(sd, 1e-8)
            X_std[:, j] = (X[:, j] - mu) / sd
            stats[col] = {'mean': mu, 'std': sd}
        else:
            stats[col] = {'mean': 0.0, 'std': 1.0}
    return X_std, stats

X_vol_std, stats_vol = standardize_matrix(df_volume[X_VOL_COLS].values, X_VOL_COLS, BINARY_COLS_VOL)
X_h_std,   stats_h   = standardize_matrix(df_hurdle[HURDLE_VARS].values, HURDLE_VARS, BINARY_COLS_HUR)



# In[20]:


# Préparation du jeu de test OOS




df_test['dyad']          = df_test['orig'] + "_" + df_test['dest']
df_test['dyad_id_test']  = df_test['dyad'].map(dyad_to_h)
df_test['dyad_id_test_v']= df_test['dyad'].map(dyad_to_v).fillna(0).astype(int)

df_test = df_test.dropna(subset=['dyad_id_test']).copy().reset_index(drop=True)
df_test = df_test.dropna(subset=['log_gdpcap_d_lag'] + HURDLE_VARS + X_VOL_COLS).copy().reset_index(drop=True)
"""
df_test['continent_orig_fill'] = df_test['orig'].apply(get_continent_id)
df_test['continent_orig_fill'] = df_test['continent_orig_fill'].fillna(7).astype(int)
cluster_test_h = df_test['continent_orig_fill'].values.astype(int)
"""

# Application du dictionnaire M49 sur le jeu de test
df_test['m49_brut'] = df_test['orig'].map(lambda x: ISO3_TO_M49_SUBREGION.get(str(x).upper(), 99))

# Projection sur l'indice Stan. En cas de pays inconnu en test, assignation au dernier cluster (sécurité)
df_test['continent_orig_fill'] = df_test['m49_brut'].map(_M49_TO_STAN).fillna(K_clusters).astype(int)
cluster_test_h = df_test['continent_orig_fill'].values

log_flow_lag_test = df_test['log_flow_lag'].fillna(0.0).values
is_mig_lag_test   = df_test['is_mig_lag'].fillna(0.0).values

X_test_v_std, _ = standardize_matrix(df_test[X_VOL_COLS].values, X_VOL_COLS,
                                     BINARY_COLS_VOL, fit_stats=stats_vol)
X_test_h_std, _ = standardize_matrix(df_test[HURDLE_VARS].values, HURDLE_VARS,
                                     BINARY_COLS_HUR, fit_stats=stats_h)


# In[21]:


# Nettoyage impératif des infinis (flux nuls passés en log)
df_hurdle = df_hurdle.replace([np.inf, -np.inf], np.nan).dropna(subset=HURDLE_REQUIRED)
df_volume = df_volume.replace([np.inf, -np.inf], np.nan).dropna(subset=VOLUME_REQUIRED)

# Vérification  (doit retourner 0)
print(f"Infinis dans Volume : {np.isinf(df_volume[X_VOL_COLS].values).sum()}")


# In[22]:


# réseau initial (avant perte temporelle ou vectorielle)
# On reconstruit la liste cible 
pays_cibles = set(cible) if CHOIX_ECHANTILLON != 5 else set(df_main['orig'].unique())

# réseaux post-filtrage
pays_hurdle_train = set(df_hurdle['orig'].unique()).union(set(df_hurdle['dest'].unique()))
pays_volume_train = set(df_volume['orig'].unique()).union(set(df_volume['dest'].unique()))
pays_test_oos     = set(df_test['orig'].unique()).union(set(df_test['dest'].unique()))


exclus_hurdle = sorted(pays_cibles - pays_hurdle_train)
exclus_volume = sorted(pays_cibles - pays_volume_train)
exclus_test   = sorted(pays_cibles - pays_test_oos)

# Audit
print(f" DIAGNOSTIC EXCLUSIONS SILENCIEUSES (Sur {len(pays_cibles)} pays initiaux)")
print(f"Pays perdus pour le modèle Hurdle (Train) : {len(exclus_hurdle)}")
print(exclus_hurdle)

print(f"\nPays perdus pour le modèle Volume (Train) : {len(exclus_volume)}")
print(exclus_volume)

print(f"\nPays perdus pour le jeu de Test (OOS 2015) : {len(exclus_test)}")
print(exclus_test)


# covariables responsables des NaN
print("\n ANALYSE VALEURS MANQUANTES ")
colonnes_cibles = list(set(HURDLE_VARS + X_VOL_COLS))


nan_counts = df[colonnes_cibles].isna().sum()
valeurs_manquantes = nan_counts[nan_counts > 0].sort_values(ascending=False)

if not valeurs_manquantes.empty:
    print(valeurs_manquantes)
else:
    print("Aucun NaN détecté dans les colonnes ici.")


# In[23]:


tous_les_pays = sorted(list(set(df['orig'].unique()).union(set(df['dest'].unique()))))
pays_to_id = {pays: i+1 for i, pays in enumerate(tous_les_pays)}
N_pays_total = len(tous_les_pays)

df_volume['orig_id_v'] = df_volume['orig'].map(pays_to_id)
df_volume['dest_id_v'] = df_volume['dest'].map(pays_to_id)
df_test['orig_id_test_v'] = df_test['orig'].map(pays_to_id)
df_test['dest_id_test_v'] = df_test['dest'].map(pays_to_id)

df_hurdle['orig_id_h'] = df_hurdle['orig'].map(pays_to_id)
df_hurdle['dest_id_h'] = df_hurdle['dest'].map(pays_to_id)


# In[24]:


# paramètres structurels macroéconomiques par pays 
K_Z = 2 # Nombre de variables d'hyper-régression (log Pop, log GDP, on utilise plus log IMR)
Z_mat = np.zeros((N_pays_total, K_Z))


for pays, pays_id in pays_to_id.items():
    idx = pays_id - 1 

    # données côté origine 
    subset_orig = df_train[df_train['orig'] == pays]
    if not subset_orig.empty:
        pop = subset_orig['log_P_it'].mean()
        gdp = subset_orig['log_gdpcap_o_lag'].mean()
        #imr = subset_orig['log_IMR_it'].mean()
    else:
        # Fallback côté destination si le pays n'a jamais été émetteur en train
        subset_dest = df_train[df_train['dest'] == pays]
        pop = subset_dest['log_P_jt'].mean()
        gdp = subset_dest['log_gdpcap_d_lag'].mean()
        #imr = subset_dest['log_IMR_jt'].mean()

    Z_mat[idx, 0] = pop
    Z_mat[idx, 1] = gdp
    #Z_mat[idx, 2] = imr # colinéarité GDP IMR ? 

# Imputation des éventuels NaN par la moyenne globale et Standardisation
for j in range(K_Z):
    col_mean = np.nanmean(Z_mat[:, j])
    Z_mat[np.isnan(Z_mat[:, j]), j] = col_mean
    # Standardisation stricte pour le HMC
    Z_mat[:, j] = (Z_mat[:, j] - np.mean(Z_mat[:, j])) / np.std(Z_mat[:, j])

# Puisque les fondamentaux (Pop, GDP, IMR) sont les mêmes pour un pays qu'il soit émetteur ou récepteur
Z_em = Z_mat
Z_at = Z_mat



# Total Faux Négatifs (Couloirs manqués) : 1295
# Total Faux Positifs (Couloirs inventés) : 1279
# 
# rajouter Z_em_h hyper-regressions hurdle idem Z_at_h. Car dans HURDLE_VARS figure encore des variables monoadiques? 
# Les hyper-regressions sont les mêmes pour hurdle/volume? 
# encore des prédictions aberrantes (>2Millions pour un flux réel de zéro!) 
# 
# corrélation positive entre erreur et amplitude: normal ? oui. NegBin: incertitude croit linéairement avec la moyenne. 
# 
# graphe en violon: distribution posteriors pas assez étroites (normal, on a que T=5 périodes d'entrainement), néanmoins les pays stable ont bien une dispersion inverse (phi) haute
# Relire description de l'article des auteurs: ils parlent de prise en compte de changement de population non du aux migrations; ils considèrent age et sexe. (?) 
# 
# 
# Pondérer le seuil de décision, anciennement ROC (argument MAPE comme métrique, arguments macroéconomiques).
# 
# 
# 

# In[25]:


stan_data = {
    'N_pays': N_pays_total,

    'N_h': int(N_h),
    'D_h': int(D_h),
    'K_h': int(K_h),
    'dyad_id_h': df_hurdle['dyad_id_h'].astype(int).tolist(),
    'is_mig': df_hurdle['is_migration'].astype(int).tolist(),
    'is_mig_lag': df_hurdle['is_mig_lag'].astype(float).tolist(),
    'X_h': X_h_std.tolist(),
    'cluster_h': cluster_h.tolist(),
    'K_Z': int(K_Z),
    'Z_em': Z_em.tolist(),
    'Z_at': Z_at.tolist(),
    'N_v': int(N_v),
    'D_v': int(D_v),
    'K_v': int(K_grav),
    'dyad_id_v': df_volume['dyad_id_v'].astype(int).tolist(),

    'orig_id_v': df_volume['orig_id_v'].astype(int).tolist(),
    'dest_id_v': df_volume['dest_id_v'].astype(int).tolist(),

    'flow': df_volume['flow'].astype(int).tolist(),
    'log_flow_lag': df_volume['log_flow_lag'].astype(float).tolist(),
    'X_v': X_vol_std.tolist(),
    'cluster_v': cluster_v.tolist(),

    'K_clusters': int(K_clusters),
    'do_ppc': 0, # a revoir 
    'do_loo': 0,  # =1 : on active la génération des log-likelihoods pour comparer les critères Stan des modèles

    'N_test': int(len(df_test)),
    'dyad_id_test_h': df_test['dyad_id_test'].astype(int).tolist(),
    'dyad_id_test_v': df_test['dyad_id_test_v'].astype(int).tolist(),

    'orig_id_test_v': df_test['orig_id_test_v'].astype(int).tolist(),
    'dest_id_test_v': df_test['dest_id_test_v'].astype(int).tolist(),
    'orig_id_h': df_hurdle['orig_id_h'].astype(int).tolist(), # NOUVEAU
    'dest_id_h': df_hurdle['dest_id_h'].astype(int).tolist(), # NOUVEAU
    'X_h_test': X_test_h_std.tolist(),
    'X_v_test': X_test_v_std.tolist(),
    'is_mig_lag_test': is_mig_lag_test.tolist(),
    'log_flow_lag_test': df_test['log_flow_lag_clean'].tolist(),
    'cluster_test_h': cluster_test_h.tolist(),
}


# In[26]:


# cellule d'audit vibe-codé, pas très grave car fonctionnelle, vérifiée. 


# AUDIT D'INTÉGRITÉ DIMENSIONNELLE ET MATRICIELLE PRE-STAN


# Bilan de la déperdition de l'information (Base Origine != Dest)
mask_base = (df_main['orig'] != df_main['dest'])
N_total = mask_base.sum()

# Séparation des dimensions temporelles théoriques
N_train_theorique = (mask_base & (df_main['year'] <= 2010)).sum()
N_test_theorique  = (mask_base & (df_main['year'] == 2015)).sum()

# Calcul des pertes dues aux dropna (indépendant de la coupure 2015)
perte_train = N_train_theorique - len(df_hurdle)
perte_test  = N_test_theorique - len(df_test)

print("─── DÉCOMPOSITION DE L'ÉCHANTILLON ──────────────────────────────")
print(f"Espace tensoriel brut (Orig != Dest) : {N_total:,} obs")
print(f"  ├─ Troncature Train (<=2010)       : {N_train_theorique:,} obs")
print(f"  │  └─ Après dropna (df_hurdle)     : {len(df_hurdle):,} obs")
print(f"  │  └─ Perte intra-Train nette      : -{perte_train:,} obs ({(perte_train/N_train_theorique)*100:.2f}%)")
print(f"  │")
print(f"  └─ Troncature Test OOS (2015)      : {N_test_theorique:,} obs")
print(f"     └─ Après dropna (df_test)       : {len(df_test):,} obs")
print(f"     └─ Perte intra-Test nette       : -{perte_test:,} obs ({(perte_test/N_test_theorique)*100:.2f}%)")
print("─────────────────────────────────────────────────────────────────\n")

# Validation de l'intégrité numérique du dictionnaire stan_data
print("─── SCAN DE CORRUPTION STAN_DATA ────────────────────────────────")
anomalies_fatales = 0

for key, val in stan_data.items():
    if isinstance(val, (list, np.ndarray)):
        arr = np.array(val)
        if np.issubdtype(arr.dtype, np.number):
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"[ERREUR] Variable '{key}' -> {n_nan} NaN | {n_inf} Inf")
                anomalies_fatales += 1

if anomalies_fatales == 0:
    print("[CLINIQUE] 0 NaN, 0 Inf détectés dans stan_data. "
          "Vecteurs purs.\nTransmission au C++ autorisée.")
else:
    raise ValueError("INTERRUPTION : Corruption détectée dans stan_data. Le HMC crashera.")


# In[27]:


# Sampling Stan parameters

N_CHAINS = 4
PARALLEL_CHAINS = 4
ITER_SAMPLING = 800
THIN = 2

N_DRAWS = ITER_SAMPLING // THIN


# In[28]:


# Sampling Stan


STAN_FILE = "../STAN/HMC_ARX_NegBinomial.stan" 



binary = STAN_FILE.replace('.stan', '')
if os.path.exists(binary):
    os.remove(binary)
    print(f"Binaire supprimé : {binary}")

os.makedirs("./stan_outputs_tmux", exist_ok=True)

model = CmdStanModel(stan_file=STAN_FILE)

# # 1. On cible UNIQUEMENT les échelles (tau) et paramètres globaux
# def custom_inits():
#     return {
#         'tau_alpha': 0.5,
#         'tau_mu': 0.5,
#         'tau_phi': 0.5,
#         'tau_sigma': 0.5,
#         'sigma_global': 0.5,
#         'phi_global_raw': 0.5
#     }

# # 2. On crée un dictionnaire par chaîne
# inits_dict = [custom_inits() for _ in range(N_CHAINS)]

fit = model.sample(
    data             = stan_data,
    chains           = N_CHAINS,
    parallel_chains  = PARALLEL_CHAINS,       
    iter_warmup      = 750,
    iter_sampling    = ITER_SAMPLING,
    save_warmup      = False,
    seed             = 42,
    inits            = 0.1,
    thin             = THIN,       
    adapt_delta      = 0.95,
    max_treedepth    = 10,
    show_progress    = True,
    sig_figs = 4,
    output_dir       = "./stan_outputs_tmux"
)




# nomenclature dynamique

custom_prefix = f"ARX_{N_pays}pays_{N_CHAINS}c_{ITER_SAMPLING}it"
renamed_csvs = []


for i, old_path in enumerate(fit.runset.csv_files):
    new_path = os.path.join("./stan_outputs_tmux", f"{custom_prefix}_chain{i+1}.csv")
    os.replace(old_path, new_path)  # os.replace  écraseme les runs précédents identiques
    renamed_csvs.append(new_path)


csv_files = renamed_csvs
print(f"Outputs sécurisés sous : {custom_prefix}_chain*.csv")




# Stan galère au début de la chaîne à 0%, puis le temps (en seconde/itération) diminue exponentiellement jusqu'à se stabiliser et rester constant vers 50% de la simulation.

# ## Fausse élégance de l'ancien modèle: 
# biais de variable omise, l'attractivité d'une destination dépend de toutes les autres destinations possibles. L'ancien modèle purement dyadique donnait l'illusion d'une compréhension fine de l'économétrie des flux, c'est faux. *Multilateral Resistance Term*. 
# L'ancien modèle n'était pas si fin du tout: ok on a le PIB, la population,.. Qu'en est il de la qualité des institutions, du climat politique, de la fiscalité interne ? c'est un moyen simple et obligatoire d'absorber toutes les variables omises. 
# Aussi, on ne demande pas au modèle d'expliquer l'économétrie "pourquoi un pays émet ou reçoit". On lui demande de distribuer correctement les flux, c'est tout. Le modèle nouveau est élégant à sa manière: il sépare orthogonalement ce qui appartient à l'Etat (alpha_i) et ce qui appartient à la géographie, les données physiques immuables (X_v)   
# 
# 
# C'est en fait la seule et unqiue solution devant autant de variables omises: le modèle n'essaye pas de deviner l'incommensurable, il l'absorbe mathématiquement. les alpha_i et beta_j sont des véritables trous noirs de variable omise. La qualité du modèle nous dira simplement à quel point ce trou noir est fort gravitationnellement, à quel point le modèle a réussit à capter les variables omises. 
# Variables omises: peuvent etre mesurables mais oubliées, peuvent être difficilement quanitfiables, voir impossible à quantifier: optimisme d'une génération, dynamique culturelle,... Avant, tous les beta_grav étaient largement biaisés. Maintenant, ils sont purs. (à quel point purs?) 
# 
# ## On gagne en robustesse de prédiction; on perd en capacité à expliquer, attribuer la cause de l'émission à un facteur précis. Ce n'est de toute façon pas notre but. 
# 
# Multicolinéarité parfaite: il faut strictement supprimer les variables monoadiques de X_v dans l'équation mu_ij=alpha_i + gamma_j + X_v*beta_grav. 

# In[ ]:


# si perte de connexion à la cellule précédente: 
# Ctrl + K + C/U pour commenter/décommenter
csv_files=csv_files = [
    "/home/onyxia/work/ProjetStat/notebooks/stan_outputs_tmux/ARX_200pays_4c_800it_chain1.csv",
    "/home/onyxia/work/ProjetStat/notebooks/stan_outputs_tmux/ARX_200pays_4c_800it_chain2.csv",
    "/home/onyxia/work/ProjetStat/notebooks/stan_outputs_tmux/ARX_200pays_4c_800it_chain3.csv",
    "/home/onyxia/work/ProjetStat/notebooks/stan_outputs_tmux/ARX_200pays_4c_800it_chain4.csv"
 ]
print(f"Fichiers ciblés : {len(csv_files)}")

# Lecture de l'en-tête
with open(csv_files[0], 'r') as f:
    for line in f:
        if not line.startswith('#'):
            all_cols = line.strip().split(',')
            break


vars_to_keep_main = [
    'prob_mig_test', 'mu_dt_test', 'phi_test',
    'beta_grav', 'phi_disp_cluster', 'alpha_global',
    'tau_alpha', 'beta_h', 'beta_lag_m49', 'mu_em', 'mu_at', 
    'rho_global_monitor', 'phi_disp_global', 'divergent__'
]
vars_to_keep_loo = ['log_lik_h', 'log_lik_v']

cols_main = [c for c in all_cols if any(c.startswith(v) for v in vars_to_keep_main)]
cols_loo = [c for c in all_cols if any(c.startswith(v) for v in vars_to_keep_loo)]

print(f"Extraction : {len(cols_main)} colonnes paramètres, {len(cols_loo)} colonnes log-vraisemblance.")

# Lecture RAM-efficient des deux blocs
dfs_main = []
dfs_loo = []

for file in csv_files:
    print(f"Lecture de {file}...")
    # On lit uniquement les colonnes requises
    df_chain = pd.read_csv(file, comment='#', usecols=cols_main + cols_loo, engine='c')


    dfs_main.append(df_chain[cols_main])
    if cols_loo: # Si do_loo était à 1
        dfs_loo.append(df_chain[cols_loo])

    del df_chain # Libération RAM

# paramètres classiques 
df_final = pd.concat(dfs_main, ignore_index=True)
print(f"Succès Paramètres. Empreinte RAM : {df_final.memory_usage().sum() / 1024**2:.2f} Mo")

# Export des Log-likelihoods. Sera ignoré silencieusement si interrupteur do_loo=0 pour la production
if cols_loo:
    df_loo_final = pd.concat(dfs_loo, ignore_index=True)


    log_lik_h_tensor = df_loo_final.filter(like='log_lik_h').values.reshape(N_CHAINS, N_DRAWS, -1)
    log_lik_v_tensor = df_loo_final.filter(like='log_lik_v').values.reshape(N_CHAINS, N_DRAWS, -1)

    # Sauvegarde sur disque en format compressé numpy 
    export_path = f"./stan_outputs/log_lik_{N_pays}pays_EmissionAttractionNegBin.npz" # titre à adapter manuellement selon le modèle, retester modèle Log-normal
    np.savez_compressed(
        export_path, 
        log_lik_h=log_lik_h_tensor, 
        log_lik_v=log_lik_v_tensor
    )
    print(f"Log-likelihoods exportées et compressées vers : {export_path}")

    # Purge  de la RAM
    del dfs_loo
    del df_loo_final
    del log_lik_h_tensor
    del log_lik_v_tensor
else:
    print("Aucune log-vraisemblance détectée (do_loo = 0, interrupteur fermé?)")


prob_mig = df_final.filter(like='prob_mig_test').values
mu_test = df_final.filter(like='mu_dt_test').values
phi_t = df_final.filter(like='phi_test').values
beta_grav = df_final.filter(like='beta_grav').values
beta_h = df_final.filter(like='beta_h').values
phi_disp_cluster = df_final.filter(like='phi_disp_cluster').values

print(f"Shape de mu_test : {mu_test.shape}")


# In[ ]:


# Chargement ArviZ optimisé RAM-efficient


#params_watch = [
#    'alpha_global', 'tau_alpha', 'beta_lag_continent',
#    'mu_intercept', 'phi_global_monitor', 'sigma_global'
#]

#idata = az.from_cmdstanpy(
#    posterior = fit,
    #log_likelihood = {
    #    'hurdle' : 'log_lik_h',
    #    'volume' : 'log_lik_v',
    #},
    #posterior_predictive = {
    #    'is_mig_hat'      : 'is_mig_hat',       
    #    'flow_hat_jensen' : 'flow_hat_jensen',  
    #},
#)


#print(f"az summary of simulation for ({N_pays} countries)")
#print(az.summary(idata, var_names=params_watch))


# # Prédictions en Numpy (plus rapide) 
# 
# Avec médiane (minimiseur norme L1)

# clip: pour ne pas laisser le processeur essayer de calculer des flottants incalculabkes (exp(709) max)
# Si le Hurdle s'est trompé et considère ouvert un couloir fermé, son esperance tend vers zéro, la boucle while va tirer des zéros à l'infini et les rejeter, boucle sans fin. Fixer alors flux=1 (pas flux=0, ça reviendrait à annuler l'architecture Hurdle, bien que le flux soit probablement nul en réalité. Il faudrait résoudre le problème à la source: améliorer le Hurdle.)
# 
# Support ZTNB: N^*
# 
# Esperance de ZNTB: doit être définie sur R+, d'ou la prise de l'exponentielle puis inversion. Problème d'exponentiation sur les FP. 

# In[ ]:


#  Purge des tirages asymétriques (NaN générés par pd.concat)
valid_draws = ~(np.isnan(mu_test).any(axis=1) | np.isnan(phi_t).any(axis=1) | np.isnan(prob_mig).any(axis=1))

mu_clean = mu_test[valid_draws]
phi_clean = phi_t[valid_draws]
prob_clean = prob_mig[valid_draws]

print(f"Nettoyage de {mu_test.shape[0] - valid_draws.sum()} tirages incomplets")

#  Protections numériques (Norme IEEE 754, maximum à exp(709), numpy renverrait inf au delà)
# Borne basse -50 : Laisse lambda tendre vers 0 sans erreur underflow.
# Borne haute 50 : Autorise des flux titanesques tout en empêchant le crash 'inf' de np.exp()
# inattaquable scientifiquement, simplement une limite physique absolue. 
eta_safe = np.clip(mu_clean, -50.0, 50.0)
phi_safe = np.clip(phi_clean, 1e-8, 1e8)

lam = np.exp(eta_safe)
n_sp = phi_safe
p_sp = np.clip(phi_safe / (phi_safe + lam), 1e-10, 1.0 - 1e-10)

# Simulation Sto exacte (distrib ZTNB)
flow_cond_sim = np.random.negative_binomial(n_sp, p_sp)
zeros_mask = (flow_cond_sim == 0)

# Limite de sécurité à 30 itérations pour le Rejection Sampling
max_retries = 30
retries = 0

while zeros_mask.any() and retries < max_retries:
    flow_cond_sim[zeros_mask] = np.random.negative_binomial(n_sp[zeros_mask], p_sp[zeros_mask])
    zeros_mask = (flow_cond_sim == 0)
    retries += 1

# Application de l'asymptote mathématique : si lambda -> 0, ZTNB(lambda) -> 1 (ne pas corrompre l'architecture du Hurlde, même si elle s'est trompée)
if zeros_mask.any():
    flow_cond_sim[zeros_mask] = 1

# Extractiondes médianes
flow_cond_med_final = np.median(flow_cond_sim, axis=0)
prob_med = np.median(prob_clean, axis=0)  

# anciennement: pure Receiver Operating Characteristic (ROC) sur le Hurdle. Mais objectif MAPE, donc à pondérer. 

y_true = df_test['flow'].values
y_true_bin = (y_true > 0).astype(int)

# Pondération de la fonction de perte (Asymétrie MAPE)
# W_FP > 1 force l'algorithme à exiger une probabilité beaucoup plus élevée avant d'ouvrir un couloir.
W_FP = 5.0



fpr, tpr, thresholds = roc_curve(y_true_bin, prob_med)

# Maximisation du gain sous pénalité asymétrique
asymmetric_score = tpr - (W_FP * fpr)
optimal_idx = np.argmax(asymmetric_score)
optimal_threshold = thresholds[optimal_idx]

print(f"Seuil ROC optimal trouvé pour ({N_pays} pays) : {optimal_threshold:.3f}")

# Décision dure : application du processus Hurdle
y_pred = np.where(prob_med > optimal_threshold, flow_cond_med_final, 0.0)

# Intervalles de confiance
is_mig_sim = np.random.binomial(1, np.clip(prob_clean, 0, 1))
flow_all = is_mig_sim * flow_cond_sim               

#  quantiles
y_pred_q05 = np.percentile(flow_all, 2.5, axis=0) 
y_pred_q95 = np.percentile(flow_all, 97.5, axis=0) 

print(f"Prédictions OOS reconstruites ({N_pays} pays) : {y_pred.shape[0]} observations")
print(f"  Médiane prédite ({N_pays} pays) : {np.median(y_pred):,.0f} migrants")
print(f"  Max prédit      ({N_pays} pays) : {y_pred.max():,.0f} migrants")


# # A FAIRE : code qui minimise la MAPE avec le seuil ROC. 

# ## Médiane attendue: proche de 1 (car 49% de zéros dans la base complète de 190 pays)
# ## Max attendu: à vérifier

# anciens résultats à garder (avec flow_cond= esperance, plutot que  flow_cond = mediane maintenant)
# 
# Seuil ROC optimal trouvé : 0.792. 
# 
# Prédictions OOS reconstruites : 4692 observations. 
# 
#   Médiane prédite : 511 migrants. 
# 
#   Max prédit      : 25,083,907 migrants. 
# 

# 
# # Explication des prédictions délirantes, choix en fonction de minimisation MSE ou MAE: 
# 
# - Stan calcule sigma_oos en gonflant la variance passée (avec l'inflation $(1 + \phi^2)$). ( L'inflation AR(1) )
# 
# - Certains des couloirs très instables ont pu voir leur $\sigma$ grimper à 1.5 ou 2.0 (plafonné à 2.0 avec np.clip).
# 
# - Si $\sigma = 2.0$, alors $\sigma^2/2 = 2$. Et $\exp(2) \approx \mathbf{7.4}$.
# Imaginons le couloir Mexique $\rightarrow$ USA. $\mu$ prédit par exemple 3,2 millions de migrants (valeur max de df_main['flow']). Si ce couloir subit la pénalité de volatilité maximale de Stan : $3.2 \text{ millions} \times 7.4 \approx \textbf{23,6 millions}$.
# 
# D'où la prédiction max de 25 M ! 

# In[ ]:


# Métriques OOS

# Évaluation du Hurdle avec le seuil donné par ROC. On vise >96.5% d'accuracy 
y_pred_bin = (prob_med > optimal_threshold).astype(int)
acc = accuracy_score(y_true_bin, y_pred_bin)

# Erreurs Absolues (norme L1) 
mask = y_true > 0
cond_mae   = np.mean(np.abs(y_true[mask] - y_pred[mask]))
global_mae = np.mean(np.abs(y_true - y_pred))

# Erreurs Relatives (%)
# A. WMAPE (Weighted MAPE) : Donne du poids aux gros couloirs
wmape = np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100

# B. MAPE modifiée de Welch & Raftery (Eq 4 page 7 de leur papier)
# Formula: 100/F * sum(|y - y_hat| / (y + 1)) pour remédier à la division par zéro 

mape_wr = np.mean(np.abs(y_true - y_pred) / (y_true + 1.0)) * 100

# Log-MAE et Coverage
log_mae  = np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))
coverage = np.mean((y_true >= y_pred_q05) & (y_true <= y_pred_q95))


print(f"PERFORMANCES du modèle ({N_pays} countries) :")
print(f"Hurdle Accuracy (open/close) : {acc*100:.1f}%")
print(f"IC 95% Coverage              : {coverage*100:.1f}%")
print(f"Conditional MAE (flow > 0)   : {cond_mae:,.0f} migrants")
print(f"Log-MAE                      : {log_mae:.4f}")

print(f"\n COMPARAISON DES MODÈLES")
print(f"{'Modèle':<40} | {'MAE (Migrants)':<15} | {'MAPE (+1)':<15}")
print("-" * 75)
print(f"{'Welch & Raftery 2022 (Bayésien Global)':<40} | {'~ 1,200':<15} | {'~ 76.0 %':<15}")
print(f"{'Random Forest (Notre base, ML)':<40} | {'~ 1,792':<15} | {'640 % sans le +1':<15}")
print("-" * 75)
print(f"{f'Notre Modèle (ARX Hurdle Bayésien ({N_pays} countries) )':<40} | {global_mae:<15,.0f} | {f'{mape_wr:.1f} %':<15}")
print("-" * 75)




# 
# # Commentaires Hurdle
# 
# Proba(ouvert) = 1/ (1+exp(-score) ). Score = alpha+beta*X . Si beta_lag est fort: + 0xbeta_lag si fermé hier, +1*beta_lag si ouvert hier. La proba bondit exponentiellement. beta>6 : Proba(ouvert demain | ouvert hier)>95% environ. (à calculer avec tableau de données)
# 
# Avec Hurdle de 77% avec l'esperance pour la prediction comme minimiseur L^2 (ancienne version! on est toujours au dessus de 92% avec le nouveau Hurdle même sur 190 pays, et avec médiane comme minimseur L1 pour les prédictions).
# 
# Hurdle Accuracy (open/close) : 77.5%
# 
# Conditional MAE (flow > 0) : 21,657 migrants
# 
# Global MAE : 17,780 migrants
# 
# Global WMAPE : 177.6%
# 
# Log-MAE : 2.036
# 
# IC 90% Coverage : 15.8% (anceisn resultats, nouveaux sont à 74%)
# 
# 
# Les erreurs MAE sont énormes contrairement à la littérature et au RF. Certainement parce qu'on travaille pour l'instant en 
# sous-échantillon pour notre modèle bayésien! 
# 
# # MAPE 
# erreur de 45 000% : imaginons que le Hurdle se trompe, on prédit 100 migrants au lieu de 0, ça fait une erreur MAPE de 10 000 % déjà ! 
# Nouveau: sur 190 pays, erreur MAPE de 268%. MAE de 946 migrants (Welch&raftery: MAE 1200 et MAPE 76%)
# 
# # IC (Intervalle de Confiance) et Coverage 
# Si le modèle sort un intervalle de confiance à 95%, dans un monde parfait on veut que les prédictions tombent dedans 95% du temps! 
# IC 90% Coverage de 74.4%<90% : le modèle est trop confiant, les intervalles encore trop étroits. 
# **Calcul du coverage:** couloir par couloir, Python vérifie si la valeur réelle tombe dans l'IC (1) ou non (0) et on fait la proportion de (1). 
# 
# **L'hétéroscédasticité :** 
# - pour un couloir européen, on jette les 5% valeurs les plus extremes de manière bilatérale, on obtient par exemple [700,1300] pour une vraie valeur 1000. ça donne une largeur de 30% (faible volatilité)
# - Pour un couloir asiatique, on jette les valeurs extremes pareils, mais on aura peut etre [250,4 500] pour une vraie valeur de 1 000, soit +150% de largeur! 
# - Enfin, l'hétéroscédasticité permet de ne pas trop toucher aux beta si la prédiction est instable sur les clusters instables, grâce au controle par sigma dans $\log(\text{flow}) \sim \mathcal{N}(\text{ar\_pred}, \sigma_d)$.
# # en bref: le couloir est maigre en migrant pour une sigma basse, et très volumineux pour une sigma haute. 
# 
# 
# **WMAPE: pénalise fortement les gros couloirs, moins les petits couloirs.**
# 
# 
# 
# **Log-MAE : (non comparable avec W&R car ils ne l'utilisent pas)**
# 
# Si le vrai flux est de 10 et qu'on prédit 20 :erreur de 10 en MAE.
# 
# Si le vrai flux est de 1 000 000 et qu'on prédit 1 000 010 : erreur de 10 en MAE aussi. 
# 
# En Log-MAE, la première erreur est grave: la prédiction est 2 fois plus grosse que la réalité. LogMAE Bien pour des données de plusieurs ordres de grandeur. 

# # Le modèle Hurdle (avec décision dure) atteint déjà 96.21% d'Accuracy (96.18% en ROC)
# rien qu'avec les données géographiques! 
# Reste à enrichir le vecteur X_h pour espérer encore améliorer le modèle, et passer beta_lag en continental plutôt que global (viser >98% ?).  
# 
# D'ailleurs, comparer les beta_lag par continents est intéressant: surement un beta_lag très fort pour l'Europe (routes européennes ne ferment pas une fois ouverte grâce à Schengen)
# 
# ### Nouveau: le Hurdle a été enrichi, mais les derniers % à attraper sont des cygnes noirs, qu'on aura probablement jamais. 

# In[ ]:


import plotly.express as px

#  Isolement des erreurs conditionnelles
df_test['y_true_bin'] = y_true_bin
df_test['y_pred_bin'] = y_pred_bin

# Faux Négatifs (FN): Modèle dit fermé (0), Réalité ouverte (1). Cygne noir 
df_test['FN'] = ((df_test['y_true_bin'] == 1) & (df_test['y_pred_bin'] == 0)).astype(int)

# Faux Positifs (FP): Modèle dit ouvert (1), Réalité fermée (0). Fantôme 
df_test['FP'] = ((df_test['y_true_bin'] == 0) & (df_test['y_pred_bin'] == 1)).astype(int)

# Agrégation spatiale par Etat émetteur (origine)
error_map = df_test.groupby('orig')[['FN', 'FP']].sum().reset_index()

print(f"Total Faux Négatifs (Couloirs manqués) : {df_test['FN'].sum()}")
print(f"Total Faux Positifs (Couloirs inventés) : {df_test['FP'].sum()}")

# Carte (Faux Négatifs en ROUGE)
fig_fn = px.choropleth(
    error_map, 
    locations="orig", 
    color="FN",
    hover_name="orig",
    color_continuous_scale="Reds",
    title="Cartographie des Faux Négatifs (FN) par pays d'origine ",
    labels={'FN': 'Nombre de FN'}
)
fig_fn.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
fig_fn.show()

# Carte (Faux Positifs en BLEU)
fig_fp = px.choropleth(
    error_map, 
    locations="orig", 
    color="FP",
    hover_name="orig",
    color_continuous_scale="Blues",
    title="Cartographie des Faux Positifs (FP) par pays d'origine",
    labels={'FP': 'Nombre de FP'}
)
fig_fp.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
fig_fp.show()


# # Problème actuel du modèle pour gérer les Faux Positifs (principaux responsables de l'explosion MAPE):
# 
# le modèle postule implicitement qu'en l'absence d'information sur un micro-état/état instable, son coefficient d'attraction est équivalent à un pays structurelleemnt moyen 
# 
# $P(\gamma_j | Y)$ converge donc vers la distribution a priori $P(\gamma_j)$.
# 
# $P(\gamma_j | Y) = \frac{P(Y | \gamma_j) P(\gamma_j)}{\int P(Y | \gamma_j) P(\gamma_j) d\gamma_j}$ 
# Si les observations Y_j sont quasi-vides (n_j = 0), la vraisemblance $P(Y_j | \gamma_j) = c$ pour tout $\gamma_j$.  $P(\gamma_j | Y) = \frac{c P(\gamma_j)}{c \int P(\gamma_j) d\gamma_j} = P(\gamma_j)$
#   
# 
# Face à un émetteur massif ($\alpha_i \gg 0$), l'équation $\mu_{ij} = \alpha_i + \mu_{at} - \text{Gravité}$ génère une log-espérance fortement positive, qui conduit à une prédiction aberrante après passage à l'exponentiel
# 
# 
# Solution: priors latents $P(\gamma, \theta | Y) \propto P(Y | \gamma) P(\gamma | Z, \theta) P(\theta)$
# où Z sont des données macro-démographiques
# Stan observe les 190 pays. Il voit que globalement, les pays avec une forte population ont une forte attraction observée dans Y. Le HMC ajuste donc le gradient de $\theta_{population}$ vers une valeur positive. 
# Le prior d'un micro-état k (sans données de flux) se translate : son $\mu_k$ devient fortement négatif car $\theta_{population}$ est positif mais $\log(P_k)$ est très faible. Shrinkage de ce micro-état / ou pays instablevers un nouveau plancher propre, et non plus vers la moyenne mondiale. L'algo apprend les lois macroéconomiques sur les pays denses pour punir/contraindre l'ignorance sur les pays vides/insables, et ne plus reproduire les erreurs du précédent modèle (décrites ci dessus dans ce markdown)

# In[ ]:


# explorer effet du seuil ROC 

# Tester une liste de seuils arbitraires instantanément
seuils_a_tester = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, optimal_threshold]

print(f"Exploration des Seuils Hurdle, {N_pays} pays")
for s in seuils_a_tester:
    # 1. On applique la décision dure avec le seuil 's' (instantané en NumPy)
    pred_test = (prob_med > s).astype(int)

    # 2. On calcule l'accuracy
    acc_test = accuracy_score(y_true_bin, pred_test)

    # 3. Affichage
    if s == optimal_threshold:
        print(f"Seuil ROC Optimal ({s:.3f}) : Accuracy = {acc_test*100:.2f}%  <<< (Celui du modèle)")
    else:
        print(f"Seuil manuel à {s:.1f}   : Accuracy = {acc_test*100:.2f}%")


# In[ ]:


# Visualisation de la courbe ROC
fig, ax = plt.subplots(figsize=(8, 6))

# Tracer la courbe ROC
ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'Courbe ROC (Seuil Opt = {optimal_threshold:.3f})')

# Tracer la ligne de hasard (random guess)
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Hasard (50/50)')

# Placer le point optimal
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='#F44336', s=100, zorder=5, 
           label='Seuil Optimal', marker='*')

# Annotations
ax.annotate(f'  Seuil: {optimal_threshold:.3f}', 
            (fpr[optimal_idx], tpr[optimal_idx]), 
            xytext=(10, -10), textcoords='offset points', fontsize=10, color='#F44336', weight='bold')

ax.set_xlabel('Taux de Faux Positifs')
ax.set_ylabel('Taux de Vrais Positifs')
ax.set_title(f"Analyse ROC pour le Modèle Hurdle, {N_pays} pays")
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"roc_curve_hurdle_{N_pays}_c.pdf", bbox_inches='tight')
plt.show()


# In[ ]:


# Visualisations. Retrouver les graphes de la LogNormale écrasés par NegBin (oubli de renommer les savefig)

"""
CONTINENT_NAMES = {1: 'Europe', 2:'Am. Nord', 3:'Afrique', 
                   4:'Am.Sud', 5:'Asie', 6: 'Océanie'}

fig, ax = plt.subplots(figsize=(10, 5))

for k in range(1, K_clusters + 1):
    draws_k = phi_disp_cluster[:, k-1].flatten()
    ax.violinplot(draws_k, positions=[k], widths=0.6, showmedians=True)

ax.set_xticks(range(1, K_clusters + 1))
ax.set_xticklabels([CONTINENT_NAMES.get(k, f'C{k}') for k in range(1, K_clusters + 1)])
ax.set_ylabel("phi_disp_cluster")
ax.set_title(f"Hétéroscédasticité Géographique — Dispersion inverse par Continent, pour {N_pays} pays")
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"NegBin_dispersion_cluster_{N_pays}_c.pdf", bbox_inches='tight')
plt.show()
"""

fig, ax = plt.subplots(figsize=(12, 6))

for k in range(1, K_clusters + 1):
    # Remplacement de sigma_cluster par phi_disp_cluster
    draws_k = phi_disp_cluster[:, k-1].flatten()
    ax.violinplot(draws_k, positions=[k], widths=0.6, showmedians=True)

ax.set_xticks(range(1, K_clusters + 1))

# Extraction dynamique des noms de sous-régions
x_labels = [SUBREGION_LABELS.get(stan_to_m49.get(k, 99), f'Cluster {k}') for k in range(1, K_clusters + 1)]
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

ax.set_ylabel("phi_disp_cluster (Dispersion inverse)")
ax.set_title(f"Hétéroscédasticité Géographique (M49) pour {N_pays} pays\n(Un \u03c6 bas indique une forte variance)")
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"NegBin_dispersion_cluster_M49_{N_pays}.pdf", bbox_inches='tight')
plt.show()


beta_means = beta_grav.mean(axis=0)
beta_q05, beta_q95 = np.percentile(beta_grav, [5, 95], axis=0)

order = np.argsort(beta_means)

fig, ax = plt.subplots(figsize=(10, max(6, K_grav * 0.4)))

colors_coef = ['#F44336' if beta_q05[i] > 0 or beta_q95[i] < 0 else '#90A4AE' for i in order]

ax.barh(range(K_grav), beta_means[order], 
        xerr=[beta_means[order] - beta_q05[order], beta_q95[order] - beta_means[order]], 
        color=colors_coef, alpha=0.8, capsize=3)

ax.set_yticks(range(K_grav))
ax.set_yticklabels([X_VOL_COLS[i] for i in order], fontsize=9)
ax.axvline(0, color='black', lw=1, ls='--')
ax.set_title(f"Coefficients Gravité pour {N_pays} pays - IC 90%\nRouge = significatif (IC exclut 0)")

plt.tight_layout()
plt.savefig(f"NegBin_gravity_coefficients_{N_pays}_c.pdf", bbox_inches='tight')
plt.show()     

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.scatter(y_true, y_pred, alpha=0.4, s=12, color='#1565C0', edgecolors='none')
lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
ax.plot(lim, lim, 'r--', lw=1.5, label='Prédiction parfaite')
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set_xlabel("Flux Réel 2015")
ax.set_ylabel("Flux Prédit")
ax.set_title(f"OOS 2015 — Observé vs Prédit pour {N_pays} pays (MAE = {global_mae:,.0f})")
ax.legend()

ax2 = axes[1]
order_err = np.argsort(y_true)
ax2.scatter(range(len(y_true)), np.abs(y_true[order_err] - y_pred[order_err]),
            alpha=0.3, s=8, color='#F44336')
ax2.set_xlabel("Dyades triées par flux réel croissant")
ax2.set_ylabel("|Erreur|")
ax2.set_yscale('log')
ax2.set_title(f"Distribution des erreurs absolues pour {N_pays} pays")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"NegBin_prediction_scatter_{N_pays}_c.pdf", bbox_inches='tight')
plt.show()


# * graphes en violon: affiche la masse de proba a posteriori entière. L'épaisseur du violon en un point y est directement proportionnel à la probabilité a posteriori que le paramètre vaille y. Cible: valeur haute, et distribution étroite. 
# 
# 
# * Intervalles de crédibilité: sachant les données, X% de certitude probabiliste qu'il contienne la vraie valeur du paramètre
# 
# * pente et accélération avec t, interpréter

# Graphe de distribution des erreurs par dyades flux croissant: 
# devrait ressembler à une bande horizontale diffuse. Tendance croissante pour les gros flux: l'erreur est positivement corrélée à la taille du flux, c'est problématique. 

# In[ ]:


# import numpy as np
# import pandas as pd
# import arviz as az

# # 1. Définition des paramètres de base (nomenclature ZTNB)
# base_params = [
#     'alpha_global', 'tau_alpha', 'beta_lag_m49',
#     'mu_intercept', 'tau_mu', 'rho_global_monitor',
#     'phi_disp_global', 'tau_phi_disp', 'tau_rho',
#     'phi_disp_cluster'
# ]

# posterior_dict = {}

# print(f"Formatage ArviZ : {N_CHAINS} chaînes de {N_DRAWS} itérations détectées.")

# # 2. Extraction stricte, transtypage et restructuration tridimensionnelle
# for param in base_params:
#     # Capture exhaustive indépendante de la syntaxe du compilateur Stan ('.' ou '[')
#     cols = [c for c in df_final.columns if c == param or c.startswith(f"{param}.") or c.startswith(f"{param}[")]

#     if not cols:
#         continue

#     # Transtypage forcé au niveau Pandas AVANT l'extraction NumPy pour garantir un float64 strict
#     data_matrix = df_final[cols].astype(float).values

#     # Redimensionnement dynamique
#     if len(cols) == 1:
#         posterior_dict[param] = data_matrix.reshape(N_CHAINS, N_DRAWS)
#     else:
#         posterior_dict[param] = data_matrix.reshape(N_CHAINS, N_DRAWS, len(cols))

# # 3. Instanciation de l'objet Inférence ArviZ
# idata = az.from_dict({"posterior": posterior_dict})

# # 4. Génération du résumé avec Intervalle de Densité Maximale (HDI) à 90%
# summary = az.summary(idata, hdi_prob=0.90)

# # Filtrage pour correspondre aux colonnes demandées (nomenclature exacte ArviZ)
# columns_to_display = ['mean', 'sd', 'hdi_5%', 'hdi_95%', 'r_hat', 'ess_bulk']
# print(summary[columns_to_display])

# # 5. Extraction des scalaires de diagnostic
# r_hat_val = summary['r_hat'].max()
# ess_bulk_val = summary['ess_bulk'].min()

# print(f"\nR_hat max    : {r_hat_val:.4f}")
# print(f"ESS_bulk min : {ess_bulk_val:.0f}")

# # 6. Décompte des divergences
# if 'divergent__' in df_final.columns:
#     div_total = pd.to_numeric(df_final['divergent__'], errors='coerce').sum()
#     print(f"Total divergences : {div_total:.0f}")


# In[ ]:


import numpy as np

# HURDLE (Probabilité d'ouverture) 
beta_h_means = beta_h.mean(axis=0)
beta_h_q05   = np.percentile(beta_h, 5, axis=0)
beta_h_q95   = np.percentile(beta_h, 95, axis=0)
K_h_sim = beta_h_means.shape[0]

print(f"\n[ HURDLE (Logit) | Simul {N_pays} pays ]")
print(f"{'Variable':<25} {'Moyenne':>10} {'IC 5%':>10} {'IC 95%':>10}  {'Significatif?':>14}")
print("-" * 75)
for j in range(K_h_sim):
    col = HURDLE_VARS[j] if j < len(HURDLE_VARS) else f"beta_h[{j+1}]"
    sig = "✓ OUI" if (beta_h_q05[j] > 0 or beta_h_q95[j] < 0) else "  non"
    print(f"{col:<25} {beta_h_means[j]:>10.3f} {beta_h_q05[j]:>10.3f} {beta_h_q95[j]:>10.3f}  {sig:>14}")

# VOLUME (flux>0) 
beta_means = beta_grav.mean(axis=0)
beta_q05   = np.percentile(beta_grav, 5, axis=0)
beta_q95   = np.percentile(beta_grav, 95, axis=0)
K_v_sim = beta_means.shape[0]

print(f"\n[ VOLUME (ZTNB) | Simul {N_pays} pays ]")
print(f"{'Variable':<25} {'Moyenne':>10} {'IC 5%':>10} {'IC 95%':>10}  {'Significatif?':>14}")
print("-" * 75)
for j in range(K_v_sim):
    col = X_VOL_COLS[j] if j < len(X_VOL_COLS) else f"beta_grav[{j+1}]"
    sig = "✓ OUI" if (beta_q05[j] > 0 or beta_q95[j] < 0) else "  non"
    print(f"{col:<25} {beta_means[j]:>10.3f} {beta_q05[j]:>10.3f} {beta_q95[j]:>10.3f}  {sig:>14}")


# Figure 1: le Graal, ce serait des formes étalées horizontalement, basses sur l'axe des Y (modèle sûr de lui + volatilité basse). 
# Figure 2: 
