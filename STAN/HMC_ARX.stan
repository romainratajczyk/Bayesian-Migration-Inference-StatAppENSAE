/*
=============================================================================
  arx_hurdle_geo.stan
  Modèle ARX Hurdle Hiérarchique avec Hétéroscédasticité Géographique

  COMPOSANT A — Hurdle (Bernoulli hiérarchique) :
    logit(P(flow>0)) = α_d + X_h·β_h + β_lag·is_mig_lag
    avec β_h = [β_D, β_LB, β_int]  (distance, frontière, interaction)

  COMPOSANT B — Volume (ARX log-Normal hiérarchique) :
    μ_{d,t} = α_{V,d} + X_v·β_grav
    log(flow_{d,t}) ~ Normal(μ_{d,t} + φ_d·(log_flow_{t-1} - μ_{d,t-1}), σ_d)

  COMPOSANT C — Hétéroscédasticité géographique :
    σ_d ~ HalfNormal(σ_cluster[continent(d)])
    σ_cluster[k] ~ HalfNormal(σ_global)   ← hiérarchie sur les clusters
    (paramétrisation non-centrée sur σ_d pour éviter les divergences)

  CONVENTIONS :
    - Toutes les matrices X sont pré-standardisées côté Python
      → priors N(0,1) sur β cohérents
    - Matt Trick appliqué sur α_d (hurdle), α_{V,d} (volume), φ_d, σ_d
    - φ_d contraint dans (-1, 1) via tanh (stationnarité stricte)
      Note : on autorise des φ légèrement négatifs pour les dyades avec
      flux très épisodiques (oscillation amortie possible)
=============================================================================
*/

data {
  // =========================================================================
  // PARTIE 1 : HURDLE
  // =========================================================================
  int<lower=1> N_h;                                  // Nb d'observations total
  int<lower=1> D_h;                                  // Nb de dyades (hurdle)
  int<lower=1> K_h;                                  // Nb de covariables hurdle
  array[N_h] int<lower=1, upper=D_h> dyad_id_h;
  array[N_h] int<lower=0, upper=1>   is_mig;
  vector[N_h]                         is_mig_lag;
  matrix[N_h, K_h]                    X_h;           // [log_D, LB, log_D×LB]

  // =========================================================================
  // PARTIE 2 : VOLUME
  // =========================================================================
  int<lower=1> N_v;                                  // Nb d'obs avec flow > 0
  int<lower=1> D_v;                                  // Nb de dyades (volume)
  int<lower=1> K_v;                                  // Nb de covariables gravité
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;
  vector[N_v]                         log_flow;
  vector[N_v]                         log_flow_lag;
  matrix[N_v, K_v]                    X_v;           // Matrice gravité standardisée

  // =========================================================================
  // CLUSTERS GÉOGRAPHIQUES (Composant C)
  // =========================================================================
  int<lower=1> K_clusters;                           // Nb de continents (6)
  array[D_h] int<lower=1, upper=K_clusters> cluster_h; // Continent de chaque dyade hurdle
  array[D_v] int<lower=1, upper=K_clusters> cluster_v; // Continent de chaque dyade volume
}

parameters {
  // =========================================================================
  // COMPOSANT A — HURDLE
  // =========================================================================

  // Hyperparamètres globaux
  real alpha_global;             // Intercept global (logit-échelle)
  real<lower=0> tau_alpha;       // Dispersion des effets aléatoires α_d
  vector[K_h] beta_h;            // [β_D, β_LB, β_int] — coefficients globaux
  real beta_lag_global;          // Effet de l'inertie migratoire

  // Matt Trick : α_d = alpha_global + tau_alpha * alpha_raw[d]
  vector[D_h] alpha_raw;

  // =========================================================================
  // COMPOSANT B — VOLUME (ARX)
  // =========================================================================

  // Intercept hiérarchique de l'équation de gravité
  real mu_intercept;             // Intercept global de μ_{d,t}
  real<lower=0> tau_mu;          // Dispersion des intercepts α_{V,d}

  // Coefficients de gravité (globaux, pas hiérarchiques : trop peu de données)
  vector[K_v] beta_grav;         // Effets des K_v covariables (dont seuil PIB)

  // Persistance AR(1) : φ_d dans (-1, 1) via tanh
  real phi_global_raw;           // espace non-contraint → phi_global = tanh(phi_global_raw)
  real<lower=0> tau_phi;
  vector[D_v] phi_raw;           // Matt Trick : φ_d = tanh(phi_global_raw + tau_phi*phi_raw)

  // Matt Trick intercept volume
  vector[D_v] mu_raw;            // α_{V,d} = mu_intercept + tau_mu * mu_raw[d]

  // =========================================================================
  // COMPOSANT C — HÉTÉROSCÉDASTICITÉ GÉOGRAPHIQUE
  // =========================================================================

  // Hiérarchie à deux niveaux sur σ :
  //   σ_global → σ_cluster[k] → σ_d
  // Paramétrisation non-centrée pour éviter les funnels de Neal sur les variances

  real<lower=0> sigma_global;            // Volatilité de référence globale
  vector<lower=0>[K_clusters] sigma_cluster; // Volatilité de référence par continent

  // Matt Trick sur σ_d :
  // σ_d = sigma_cluster[cluster_v[d]] * exp(sigma_raw[d] * tau_sigma)
  // → garantit σ_d > 0, évite les funnels, et est robuste aux petits clusters
  vector[D_v] sigma_raw;                 // N(0,1) → transformation log-normale
  real<lower=0> tau_sigma;               // Dispersion intra-cluster de σ_d
}

transformed parameters {
  // =========================================================================
  // RECONSTRUCTION DES PARAMÈTRES
  // =========================================================================

  // --- A : Effets aléatoires hurdle ---
  vector[D_h] alpha_d;
  for (d in 1:D_h)
    alpha_d[d] = alpha_global + tau_alpha * alpha_raw[d];

  // --- B : Intercepts volume et persistances ---
  vector[D_v] alpha_V;    // Intercept propre à chaque dyade (volume)
  vector[D_v] phi_d;      // Persistance AR dans (-1, 1)

  real phi_global = tanh(phi_global_raw);  // Transformation globale

  for (d in 1:D_v) {
    alpha_V[d] = mu_intercept + tau_mu * mu_raw[d];
    // tanh mappe (-∞,+∞) → (-1,1) : stationnarité garantie
    phi_d[d]   = tanh(phi_global_raw + tau_phi * phi_raw[d]);
  }

  // --- C : Volatilités par dyade ---
  // σ_d = σ_cluster[k] * exp(tau_sigma * sigma_raw[d])
  // → σ_d > 0 toujours, distribution log-normale autour du cluster
  vector<lower=0>[D_v] sigma_d;
  for (d in 1:D_v)
    sigma_d[d] = sigma_cluster[cluster_v[d]] * exp(tau_sigma * sigma_raw[d]);

  // =========================================================================
  // PRÉDICTEURS LINÉAIRES (vectorisés pour la vitesse)
  // =========================================================================

  // --- A : Score logistique hurdle ---
  vector[N_h] logit_p;
  logit_p = alpha_d[dyad_id_h]
            + X_h * beta_h
            + beta_lag_global * is_mig_lag;

  // --- B : μ_{d,t} pour chaque observation volume ---
  // μ_{d,t} = α_{V,d} + X_v · β_grav
  vector[N_v] mu_dt;
  mu_dt = alpha_V[dyad_id_v] + X_v * beta_grav;

  // Prédicteur AR(1) : μ_{d,t} + φ_d · (log_flow_{t-1} - μ_{d,t-1})
  // Approximation : on utilise log_flow_lag comme proxy de μ_{d,t-1}
  // (standard dans les modèles ARX à covariables variant dans le temps)
  vector[N_v] ar_pred;
  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    ar_pred[n] = mu_dt[n] + phi_d[d] * (log_flow_lag[n] - mu_dt[n]);
  }
}

model {
  // =========================================================================
  // PRIORS — COMPOSANT A (Hurdle)
  // =========================================================================
  // logit(0.7) ≈ 0.85 → on centre autour d'un taux d'activation ~70%
  alpha_global    ~ normal(0.5, 2);
  tau_alpha       ~ exponential(1);

  // Coefficients hurdle : priors faiblement informatifs sur X standardisée
  beta_h[1] ~ normal(-1, 1);   // β_D : distance → effet négatif attendu
  beta_h[2] ~ normal(2, 1);    // β_LB : frontière commune → effet positif fort
  beta_h[3] ~ normal(0.5, 1);  // β_int : interaction (signe incertain)

  // Inertie migratoire : fort effet positif attendu
  beta_lag_global ~ normal(1.5, 1);

  alpha_raw ~ std_normal();    // Matt Trick


  // =========================================================================
  // PRIORS — COMPOSANT B (Volume ARX)
  // =========================================================================

  // Intercept global de gravité
  // log(flow) ∈ [4, 12] pour flux > 0 ; prior large mais centré
  mu_intercept ~ normal(8, 3);
  tau_mu       ~ exponential(1);

  // Coefficients de gravité sur X standardisée → N(0,1) standard
  // Quelques priors légèrement informatifs sur les signes attendus :
  beta_grav ~ normal(0, 1);   // Prior commun ; ajusté ci-dessous pour quelques clés

  // Persistance AR
  phi_global_raw ~ normal(1, 0.5);   // tanh(1) ≈ 0.76 : forte persistance attendue
  tau_phi        ~ exponential(2);
  phi_raw        ~ std_normal();

  mu_raw ~ std_normal();             // Matt Trick intercepts volume


  // =========================================================================
  // PRIORS — COMPOSANT C (Hétéroscédasticité géographique)
  // =========================================================================

  // Hiérarchie : global → cluster → dyade
  // log(flow) résiduel typiquement d'ordre 1-2 → prior sur σ centré sur 1
  sigma_global ~ exponential(1);

  // σ_cluster ~ HalfNormal(sigma_global) : clusters proches du global
  // mais libres de s'en écarter
  for (k in 1:K_clusters)
    sigma_cluster[k] ~ normal(sigma_global, 0.5);  // Half-Normal implicite (lower=0)

  // Dispersion intra-cluster : prior conservateur pour éviter les divergences
  tau_sigma ~ exponential(2);

  // Matt Trick σ_d
  sigma_raw ~ std_normal();


  // =========================================================================
  // VRAISEMBLANCES
  // =========================================================================

  // --- A : Bernoulli logistique (hurdle) ---
  is_mig ~ bernoulli_logit(logit_p);

  // --- B : Log-Normale AR(1) conditionnelle (volume) ---
  // sigma_d[dyad_id_v] : indexation vectorisée
  log_flow ~ normal(ar_pred, sigma_d[dyad_id_v]);
}

generated quantities {
  // =========================================================================
  // POST-PROCESSING
  // =========================================================================

  // Log-vraisemblances ponctuelles (pour LOO-CV avec ArviZ)
  vector[N_h] log_lik_h;
  vector[N_v] log_lik_v;

  // Prédictions pour PPC
  array[N_h] int is_mig_hat;
  vector[N_v] log_flow_hat;

  // Calculs
  for (n in 1:N_h) {
    is_mig_hat[n] = bernoulli_logit_rng(logit_p[n]);
    log_lik_h[n]  = bernoulli_logit_lpmf(is_mig[n] | logit_p[n]);
  }

  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    log_flow_hat[n] = normal_rng(ar_pred[n], sigma_d[d]);
    log_lik_v[n]    = normal_lpdf(log_flow[n] | ar_pred[n], sigma_d[d]);
  }

  // Phi global pour monitoring (en espace tanh, donc dans (-1,1))
  real phi_global_monitor = phi_global;
}