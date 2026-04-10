// hurdle_ar1.stan
// Modèle Hurdle AR(1) Hiérarchique pour Flux Migratoires Bilatéraux

data {
  // PARTIE 1 : HURDLE (Bernoulli)
  int<lower=1> N_h;                              // Nb total d'observations
  int<lower=1> D_h;                              // Nb de dyades (total)
  array[N_h] int<lower=1, upper=D_h> dyad_id_h;  // Dyade pour chaque obs
  array[N_h] int<lower=0, upper=1> is_mig;       // 1 si flow > 0, 0 sinon
  vector[N_h] is_mig_lag;                        // is_migration à t-1

  // PARTIE 2 : VOLUME (Log-Normale, conditionnelle sur flow > 0)
  int<lower=1> N_v;                              // Nb d'obs avec flow > 0
  int<lower=1> D_v;                              // Nb de dyades avec volume
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;  // Dyade pour chaque obs volume
  vector[N_v] log_flow;                          // log(flow) | flow > 0
  vector[N_v] log_flow_lag;                      // log(flow) à t-1 | flow_lag > 0
}

parameters {
  // PARTIE 1 : HURDLE : Hyperparamètres globaux
  real alpha_global;           // Intercept global de la logistique
  real<lower=0> tau_alpha;     // Dispersion des intercepts entre dyades
  real beta_lag_global;        // Effet global du lag binaire sur P(flow > 0)

  // Effets aléatoires non-centrés 
  vector[D_h] alpha_raw;       // alpha_d = alpha_global + tau_alpha * alpha_raw

  // PARTIE 2 : VOLUME : Hyperparamètres globaux
  real mu_global;                    // Niveau d'équilibre global (log-échelle)
  real<lower=0> tau_mu;              // Dispersion des équilibres entre dyades

  real<lower=0, upper=1> phi_global; // Persistance AR globale dans (0,1)
  real<lower=0> tau_phi;             // Dispersion des persistances

  real<lower=0> sigma_global;        // Volatilité globale de référence

  // Effets aléatoires non-centrés
  vector[D_v] mu_raw;                // mu_d  = mu_global  + tau_mu  * mu_raw
  vector[D_v] phi_raw;               // phi_d = f(phi_global, tau_phi, phi_raw)
  vector<lower=0>[D_v] sigma_d;      // Volatilité propre à chaque dyade
}

transformed parameters {
  // RECONSTRUCTION DES PARAMÈTRES AU NIVEAU DYADE

  // Partie 1
  vector[D_h] alpha_d;
  for (d in 1:D_h)
    alpha_d[d] = alpha_global + tau_alpha * alpha_raw[d];

  // Partie 2
  vector[D_v] mu_d;
  vector[D_v] phi_d;

  for (d in 1:D_v) {
    mu_d[d]  = mu_global + tau_mu * mu_raw[d];
    // Transformation logistique pour contraindre phi dans (0,1)
    // Garantit la stationnarité du processus AR(1)
    phi_d[d] = inv_logit(logit(phi_global) + tau_phi * phi_raw[d]);
  }
}

model {
  // PRIORS : Partie 1 (Hurdle)
  // logit(0.5) = 0 → prior centré sur P(migration) = 50% a priori
  // En réalité notre taux est ~70%, mais on reste large (prior faible)
  alpha_global   ~ normal(0, 2);
  tau_alpha      ~ exponential(1);
  beta_lag_global ~ normal(1, 1);  // Lag positif attendu : si migré avant,
                                    // plus probable de migrer encore

  alpha_raw ~ std_normal();         // Matt Trick Partie 1

  // PRIORS : Partie 2 (Volume)
  // log(flow) pour flux > 0 est typiquement dans [4, 12] (e^4 ≈ 55 migrants)
  mu_global    ~ normal(8, 3);
  tau_mu       ~ exponential(1);

  phi_global   ~ beta(4, 2);       // Asymétrique vers 1 : flux très persistants
  tau_phi      ~ exponential(2);

  sigma_global ~ exponential(1);

  mu_raw  ~ std_normal();          // Matt Trick Partie 2
  phi_raw ~ std_normal();
  sigma_d ~ normal(sigma_global, 0.5);

  // VRAISEMBLANCE : Partie 1 : Bernoulli logistique
  // P(is_mig[n] = 1) = logistic(alpha_d + beta_lag * is_mig_lag[n])
  {
    vector[N_h] logit_p;
    for (n in 1:N_h) {
      int d = dyad_id_h[n];
      logit_p[n] = alpha_d[d] + beta_lag_global * is_mig_lag[n];
    }
    is_mig ~ bernoulli_logit(logit_p);
  }

  // VRAISEMBLANCE : Partie 2 : Log-Normale AR(1) conditionnelle sur flow > 0
  // On ne met PAS de zéros ici : l'ensemble d'entraînement est déjà filtré
  // sur flow > 0 ET log_flow_lag non-manquant (conditioning set propre).
  {
    vector[N_v] mu_pred;
    for (n in 1:N_v) {
      int d = dyad_id_v[n];
      // AR(1) avec mean-reversion vers mu_d[d]
      mu_pred[n] = mu_d[d] + phi_d[d] * (log_flow_lag[n] - mu_d[d]);
    }
    log_flow ~ normal(mu_pred, sigma_d[dyad_id_v]);
  }
}

generated quantities {
  // POST-PROCESSING : PPC + Log-vraisemblance pour LOO

  // Partie 1
  array[N_h] int is_mig_hat;
  vector[N_h] log_lik_h;

  for (n in 1:N_h) {
    int d = dyad_id_h[n];
    real logit_p_n = alpha_d[d] + beta_lag_global * is_mig_lag[n];
    is_mig_hat[n] = bernoulli_logit_rng(logit_p_n);
    log_lik_h[n]  = bernoulli_logit_lpmf(is_mig[n] | logit_p_n);
  }

  // Partie 2
  vector[N_v] log_flow_hat;
  vector[N_v] log_lik_v;

  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    real mu_pred_n = mu_d[d] + phi_d[d] * (log_flow_lag[n] - mu_d[d]);
    log_flow_hat[n] = normal_rng(mu_pred_n, sigma_d[d]);
    log_lik_v[n]    = normal_lpdf(log_flow[n] | mu_pred_n, sigma_d[d]);
  }
}
