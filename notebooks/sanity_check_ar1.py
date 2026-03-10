"""
=============================================================================
SANITY CHECK v2 — Modèle Hurdle AR(1) Hiérarchique Bayésien
=============================================================================

LEÇON DU SANITY CHECK v1 :
  R_hat=4.39, ESS=4 → la loi Normale est fondamentalement inadaptée.
  La matrice migratoire est CREUSE (~30% de vrais zéros sur 11 pays).
  Forcer log1p(0)=0 dans une loi continue crée une masse de probabilité
  artificielle en 0 que le modèle Normal ne peut pas gérer.

SOLUTION — MODÈLE HURDLE EN DEUX ÉTAPES :
  Étape 1 — L'Obstacle (Bernoulli logistique, hiérarchique) :
    P(flow > 0 | dyade d, année t) = logistic(alpha_d + beta_lag * is_mig_lag)
    
  Étape 2 — Le Volume (Log-Normale sur flux > 0 uniquement) :
    log(flow) | flow > 0, dyade d ~ Normal(mu_d + phi_d*(log_flow_lag - mu_d),
                                            sigma_d)
  
  is_migration (colonne déjà présente dans le dataset) est exactement
  l'indicateur binaire de l'Étape 1. Le dataset nous a dit ce qu'il voulait.

POURQUOI CE MODÈLE CONVERGE LÀ OÙ L'AUTRE ÉCHOUE :
  - Pas de zéros dans la vraisemblance log-normale (on conditionne sur flow>0)
  - Le problème topologique (masse de Dirac en 0) est modélisé explicitement
  - Deux sous-problèmes bien posés < un seul mal posé
=============================================================================
"""

import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. CHARGEMENT ET FILTRAGE
# =============================================================================

DATA_PATH = "/Users/romain/Desktop/Projets DS/ProjetStat/data/data_final/DF_GRAVITY_sans_NaN.csv"
df_main = pd.read_csv(DATA_PATH)

PAYS_STABLES    = ['FRA', 'USA', 'ESP', 'CAN', 'MEX']
PAYS_CHAOTIQUES = ['DZA', 'MMR', 'RWA', 'HTI', 'ZAF', 'NER']
PAYS_TEST       = PAYS_STABLES + PAYS_CHAOTIQUES

df = df_main[
    df_main['orig'].isin(PAYS_TEST) &
    df_main['dest'].isin(PAYS_TEST) &
    (df_main['orig'] != df_main['dest'])
].copy()

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)

print(f"Lignes après filtrage : {len(df)}")
print(f"% flux nuls : {(df['flow'] == 0).mean():.1%}  ← justifie le modèle Hurdle")


# =============================================================================
# 2. CONSTRUCTION DES DYADES ET DES LAGS
# =============================================================================

df['dyad'] = df['orig'] + "_" + df['dest']
dyades_uniques = sorted(df['dyad'].unique())
dyad_to_id = {d: i + 1 for i, d in enumerate(dyades_uniques)}
df['dyad_id'] = df['dyad'].map(dyad_to_id)
D = len(dyades_uniques)

# --- Lags ---
# is_migration : indicateur binaire (1 si flow > 0), déjà dans le dataset
# On reconstruit proprement si la colonne s'appelle différemment
if 'is_migration' not in df.columns:
    df['is_migration'] = (df['flow'] > 0).astype(int)

# Lag de is_migration (pour l'Étape 1 du Hurdle)
df['is_mig_lag'] = df.groupby('dyad')['is_migration'].shift(1)

# Lag de log(flow) — uniquement utile pour l'Étape 2 (flux > 0)
# On prend le log du flux précédent SI il était positif, sinon NaN
df['log_flow'] = np.where(df['flow'] > 0, np.log(df['flow']), np.nan)
df['log_flow_lag'] = df.groupby('dyad')['log_flow'].shift(1)

# Suppression des premières années (pas de lag disponible)
df_clean = df.dropna(subset=['is_mig_lag']).copy().reset_index(drop=True)

print(f"\nAprès suppression des premières années : {len(df_clean)} obs")
print(f"Dyades uniques : {df_clean['dyad'].nunique()}")


# =============================================================================
# 3. SÉPARATION DES DEUX SOUS-PROBLÈMES
# =============================================================================

# --- Étape 1 : Toutes les observations (prédire is_migration) ---
df_hurdle = df_clean.copy()
N_total = len(df_hurdle)

# --- Étape 2 : Uniquement les flux > 0 (prédire le volume) ---
# IMPORTANT : on conditionne sur flow > 0 ET log_flow_lag non-nul
# Si log_flow_lag est NaN (t-1 avait flow=0), on peut soit :
#   (a) imputer par la moyenne de la dyade
#   (b) exclure ces observations
# On choisit (b) pour le sanity check (plus propre statistiquement)
df_volume = df_clean[
    (df_clean['flow'] > 0) &
    df_clean['log_flow_lag'].notna()
].copy().reset_index(drop=True)

N_vol = len(df_volume)

print(f"\nÉtape 1 (Hurdle/Bernoulli) : {N_total} observations")
print(f"Étape 2 (Volume/Log-Normale) : {N_vol} observations (flux > 0 avec lag > 0)")
print(f"Proportion retenue pour le volume : {N_vol/N_total:.1%}")


# =============================================================================
# 4. DICTIONNAIRE STAN
# =============================================================================

# Re-encodage des dyades pour le sous-ensemble volume
# (certaines dyades n'ont peut-être JAMAIS de flux > 0 avec lag > 0)
dyades_vol = sorted(df_volume['dyad'].unique())
dyad_to_id_vol = {d: i + 1 for i, d in enumerate(dyades_vol)}
id_to_dyad_vol = {v: k for k, v in dyad_to_id_vol.items()}
df_volume['dyad_id_vol'] = df_volume['dyad'].map(dyad_to_id_vol)
D_vol = len(dyades_vol)

# Encodage pour l'étape hurdle (toutes dyades)
id_to_dyad_all = {v: k for k, v in dyad_to_id.items()}

stan_data = {
    # --- Étape 1 : Hurdle ---
    'N_h'        : N_total,
    'D_h'        : D,
    'dyad_id_h'  : df_hurdle['dyad_id'].astype(int).tolist(),
    'is_mig'     : df_hurdle['is_migration'].astype(int).tolist(),
    'is_mig_lag' : df_hurdle['is_mig_lag'].astype(float).tolist(),

    # --- Étape 2 : Volume ---
    'N_v'          : N_vol,
    'D_v'          : D_vol,
    'dyad_id_v'    : df_volume['dyad_id_vol'].astype(int).tolist(),
    'log_flow'     : df_volume['log_flow'].tolist(),
    'log_flow_lag' : df_volume['log_flow_lag'].tolist(),
}

print(f"\nDictionnaire Stan construit.")
print(f"  N_h={N_total}, D_h={D}, N_v={N_vol}, D_v={D_vol}")

# Vérifications de sanité
assert not np.any(np.isnan(stan_data['log_flow'])), "NaN dans log_flow !"
assert not np.any(np.isinf(stan_data['log_flow'])), "Inf dans log_flow !"
assert not np.any(np.isnan(stan_data['log_flow_lag'])), "NaN dans log_flow_lag !"
assert all(v in [0,1] for v in stan_data['is_mig']), "is_mig non-binaire !"
print("  ✓ Toutes les assertions passées (pas de NaN/Inf)")


# =============================================================================
# 5. COMPILATION ET SAMPLING
# =============================================================================

STAN_FILE = "/Users/romain/Desktop/Projets DS/ProjetStat/STAN/MCMC_AR1.stan"

print(f"\nCompilation du modèle : {STAN_FILE}")
model = CmdStanModel(stan_file=STAN_FILE)
print("Compilation OK.")

print("\nLancement MCMC...")
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    seed=42,
    adapt_delta=0.95,
    max_treedepth=12,
    show_progress=True,
)


# =============================================================================
# 6. DIAGNOSTICS
# =============================================================================

print("\n" + "="*60)
print("  DIAGNOSTICS DE CONVERGENCE")
print("="*60)

resume = fit.summary()

params_globaux = [
    'mu_global', 'tau_mu', 'phi_global', 'tau_phi', 'sigma_global',  # Volume
    'alpha_global', 'tau_alpha', 'beta_lag_global'                    # Hurdle
]

print(f"\n{'Paramètre':<25} {'Moyenne':>10} {'StdDev':>10} {'R_hat':>8} {'ESS_bulk':>10}")
print("-" * 65)
for p in params_globaux:
    if p in resume.index:
        row = resume.loc[p]
        flag = " ✓" if row['R_hat'] < 1.01 else " ⚠️"
        print(f"{p:<25} {row['Mean']:>10.4f} {row['StdDev']:>10.4f} "
              f"{row['R_hat']:>8.4f}{flag} {row['ESS_bulk']:>10.0f}")

rhat_max = resume['R_hat'].max()
ess_min  = resume['ESS_bulk'].min()
print(f"\n  R_hat max    : {rhat_max:.4f}  {'✓' if rhat_max < 1.01 else '⚠️'}")
print(f"  ESS_bulk min : {ess_min:.0f}   {'✓' if ess_min > 400 else '⚠️'}")
print(fit.diagnose())


# =============================================================================
# 7. VISUALISATIONS
# =============================================================================

idata = az.from_cmdstanpy(
    posterior=fit,
    log_likelihood={'hurdle': 'log_lik_h', 'volume': 'log_lik_v'},
    posterior_predictive={'is_mig_hat': 'is_mig_hat', 'log_flow_hat': 'log_flow_hat'}
)

# --- Traceplots ---
fig, axes = plt.subplots(len(params_globaux), 2, figsize=(14, 3 * len(params_globaux)))
fig.suptitle("Traceplots — Hyperparamètres Globaux (Modèle Hurdle AR1)", 
             fontsize=13, fontweight='bold', y=1.01)
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

for i, param in enumerate(params_globaux):
    if param not in idata.posterior:
        axes[i, 0].set_visible(False)
        axes[i, 1].set_visible(False)
        continue
    chains_data = idata.posterior[param].values
    ax_t, ax_h = axes[i, 0], axes[i, 1]
    for c in range(chains_data.shape[0]):
        ax_t.plot(chains_data[c], alpha=0.7, lw=0.5, color=colors[c])
    ax_t.set_title(f'Trace : {param}', fontsize=9)
    all_d = chains_data.flatten()
    ax_h.hist(all_d, bins=60, color='#1565C0', alpha=0.7, density=True)
    ax_h.axvline(np.mean(all_d), color='red', lw=1.5, linestyle='--',
                 label=f'μ={np.mean(all_d):.3f}')
    ax_h.set_title(f'Posterior : {param}', fontsize=9)
    ax_h.legend(fontsize=8)

plt.tight_layout()

plt.close()



# --- Posterior Predictive Check : Étape 1 (Hurdle) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PPC — Étape 1 : Hurdle (Bernoulli)\nLe modèle reproduit-il la fréquence des zéros ?",
             fontsize=12, fontweight='bold')

is_mig_obs  = np.array(stan_data['is_mig'])
is_mig_hat  = idata.posterior_predictive['is_mig_hat'].values.reshape(-1, N_total)

obs_rate  = is_mig_obs.mean()
pred_rates = is_mig_hat.mean(axis=1)

axes[0].hist(pred_rates, bins=40, color='#2196F3', alpha=0.7, density=True, label='Réplications')
axes[0].axvline(obs_rate, color='red', lw=2, label=f'Observé = {obs_rate:.3f}')
axes[0].set_xlabel("Proportion de flux > 0")
axes[0].set_title("Distribution de la proportion prédite de couloirs actifs")
axes[0].legend()

# Calibration par dyade
dyade_obs_rate  = df_hurdle.groupby('dyad')['is_migration'].mean().values
dyade_pred_rate = np.array([
    is_mig_hat[:, df_hurdle['dyad'] == d].mean()
    for d in dyades_uniques if d in df_hurdle['dyad'].values
])
axes[1].scatter(dyade_obs_rate, dyade_pred_rate, alpha=0.6, s=40, color='#1565C0')
lim = [0, 1]
axes[1].plot(lim, lim, 'r--', lw=1.5)
axes[1].set_xlabel("Proportion observée (par dyade)")
axes[1].set_ylabel("Proportion prédite (médiane)")
axes[1].set_title("Calibration par dyade")

plt.tight_layout()

plt.close()



# --- Posterior Predictive Check : Étape 2 (Volume) ---
log_flow_hat = idata.posterior_predictive['log_flow_hat'].values.reshape(-1, N_vol)
log_flow_obs = np.array(stan_data['log_flow'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PPC — Étape 2 : Volume (Log-Normale)\nConditionnellement à flow > 0",
             fontsize=12, fontweight='bold')

axes[0].hist(log_flow_obs, bins=40, color='black', alpha=0.6, density=True, 
             label='Observé', zorder=3)
for i in range(min(150, log_flow_hat.shape[0])):
    axes[0].hist(log_flow_hat[i], bins=40, alpha=0.02, density=True, color='#2196F3')
axes[0].set_xlabel("log(flow) | flow > 0")
axes[0].set_title("Distribution du volume prédit vs observé")
axes[0].legend()

y_pred_med = np.median(log_flow_hat, axis=0)
axes[1].scatter(log_flow_obs, y_pred_med, alpha=0.4, s=15, color='#1565C0')
lim2 = [log_flow_obs.min(), log_flow_obs.max()]
axes[1].plot(lim2, lim2, 'r--', lw=1.5, label='Prédiction parfaite')
resid  = log_flow_obs - y_pred_med
mae_v  = np.mean(np.abs(resid))
r2_v   = 1 - np.sum(resid**2) / np.sum((log_flow_obs - log_flow_obs.mean())**2)
axes[1].text(0.05, 0.95, f"MAE(log) = {mae_v:.3f}\nR²       = {r2_v:.3f}",
             transform=axes[1].transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[1].set_xlabel("log(flow) observé")
axes[1].set_ylabel("log(flow) prédit (médiane)")
axes[1].set_title("Observé vs Prédit (Volume uniquement)")
axes[1].legend()

plt.tight_layout()

plt.close()


print("\n✓ Pipeline complet terminé.")
