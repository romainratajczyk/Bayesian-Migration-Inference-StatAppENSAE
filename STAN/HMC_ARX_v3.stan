/*
Bayesian Hierarchical ARX Hurdle Model for Gravity Migration

Component A: Hurdle (Hierarchical Bernoulli)
Component B: Volume (Hierarchical ARX Log-Normal)
Component C: Geographic Heteroscedasticity (Variance clustering)
*/



/*
Voir notes cahier pour les dernières modifications & optimisation
*/

data {
  // 1. Hurdle
  int<lower=1> N_h;
  int<lower=1> D_h;
  int<lower=1> K_h;
  array[N_h] int<lower=1, upper=D_h> dyad_id_h;
  array[N_h] int<lower=0, upper=1>   is_mig;
  vector[N_h]                         is_mig_lag;
  matrix[N_h, K_h]                    X_h;

  // 2. Volume
  int<lower=1> N_v;
  int<lower=1> D_v;
  int<lower=1> K_v;
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;
  vector[N_v]                         log_flow;
  vector[N_v]                         log_flow_lag;
  matrix[N_v, K_v]                    X_v;

  // 3. Clusters géographiques
  int<lower=1>                              K_clusters;
  array[D_h] int<lower=1, upper=K_clusters> cluster_h;
  array[D_v] int<lower=1, upper=K_clusters> cluster_v;

  // 4. Test OOS. SEULEMENT les covariables, PAS de prédictions ici,
  //    sont calculées en Python post-sampling pour éviter les chargements de gros CSV 
  int<lower=0>                              N_test;
  array[N_test] int<lower=1, upper=D_h>     dyad_id_test_h;
  array[N_test] int<lower=0, upper=D_v>     dyad_id_test_v;  // 0 = nouvelle dyade
  matrix[N_test, K_h]                        X_h_test;
  vector[N_test]                             is_mig_lag_test;
  matrix[N_test, K_v]                        X_v_test;
  vector[N_test]                             log_flow_lag_test;
  array[N_test] int<lower=1, upper=K_clusters> cluster_test_h;
}

parameters {
  // A. Hurdle
  real alpha_global;
  real<lower=0> tau_alpha;
  vector[K_h] beta_h;
  real beta_lag_global;
  vector[D_h] alpha_raw;

  // B. Volume ARX
  real mu_intercept;
  real<lower=0> tau_mu;
  vector[K_v] beta_grav;
  real phi_global_raw;
  real<lower=0> tau_phi;
  vector[D_v] phi_raw;
  vector[D_v] mu_raw;

  // C. Hétéroscédasticité géographique
  real<lower=0> sigma_global;
  vector<lower=0>[K_clusters] sigma_cluster;
  vector[D_v] sigma_raw;
  real<lower=0> tau_sigma;
}

transformed parameters {
  // A
  vector[D_h] alpha_d;
  for (d in 1:D_h)
    alpha_d[d] = alpha_global + tau_alpha * alpha_raw[d];

  // B
  real phi_global = tanh(phi_global_raw);
  vector[D_v] alpha_V;
  vector[D_v] phi_d;
  for (d in 1:D_v) {
    alpha_V[d] = mu_intercept + tau_mu * mu_raw[d];
    phi_d[d]   = tanh(phi_global_raw + tau_phi * phi_raw[d]);
  }

  // C — paramétrisation log-normale non-centrée sur sigma_d
  // sigma_d : volatilité in-sample (inchangée, utilisée pour la vraisemblance)
vector<lower=0>[D_v] sigma_d;
for (d in 1:D_v)
  sigma_d[d] = fmax(sigma_cluster[cluster_v[d]] * exp(tau_sigma * sigma_raw[d]), 1e-4); // protection numérique contre les sigma=0, aurait fait planter la chaine. 
                                                                                            // Pas dramatique: La trajectorie aurait été rejetée par Metropolis-Hastings, mais cela peut aider à la convergence et à la santé globale du code. 
// sigma_d_oos : volatilité à horizon h=1 pas (5 ans) pour la prédiction OOS
// Formule AR(1) exacte : Var(t+h|t) = sigma² * (1 - phi^2h) / (1 - phi²)
// Avec h=1 pas de 5 ans, phi^(2*1) = phi² car on a déjà absorbé h dans phi
// En pratique : sigma_oos = sigma_d * sqrt((1 - phi_d²·phi_d²) / (1 - phi_d²))
// Simplification robuste : sigma_oos = sigma_d / sqrt(1 - phi_d²)
// → quand phi→1, sigma_oos → ∞ correctement (très persistant = très incertain)
vector<lower=0>[D_v] sigma_d_oos;
for (d in 1:D_v) {
  real phi2 = square(phi_d[d]);
  // Protection numérique : si phi très proche de 1, on plafonne
  real denom = fmax(1.0 - phi2, 0.01);
  sigma_d_oos[d] = sigma_d[d] / sqrt(denom);
}
  // Prédicteurs linéaires (vectorisés)
  vector[N_h] logit_p =
    alpha_d[dyad_id_h] + X_h * beta_h + beta_lag_global * is_mig_lag;

  vector[N_v] mu_dt = alpha_V[dyad_id_v] + X_v * beta_grav;

  vector[N_v] ar_pred;
  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    ar_pred[n] = mu_dt[n] + phi_d[d] * (log_flow_lag[n] - mu_dt[n]);
  }
}

model {
  // A — Priors hurdle
  alpha_global    ~ normal(0.5, 2);
  tau_alpha       ~ exponential(1);
  beta_h[1]       ~ normal(-1, 1);
  beta_h[2]       ~ normal(2, 1);
  beta_h[3]       ~ normal(0.5, 1);
  beta_lag_global ~ normal(1.5, 1);
  alpha_raw       ~ std_normal();

  // B — Priors volume
  mu_intercept   ~ normal(8, 3);
  tau_mu         ~ exponential(1);
  beta_grav      ~ normal(0, 1);
  phi_global_raw ~ normal(0.5, 0.5); // tanh(0.5) ≈ 0.46 < tanh(1) = 0.76 pour mieux capter l'hétérogénéité.  stationnarité a priori modérée, on ne veut pas Phi(FR>DZA)≈Phi(Thai>MMR) 
  tau_phi        ~ exponential(2);
  phi_raw        ~ std_normal();
  mu_raw         ~ std_normal();

  // C — Priors variance
  sigma_global ~ exponential(1);
  for (k in 1:K_clusters)
    sigma_cluster[k] ~ normal(sigma_global, 0.5);
  tau_sigma ~ exponential(2);
  sigma_raw ~ std_normal();

  // Vraisemblances
  is_mig   ~ bernoulli_logit(logit_p);
  log_flow ~ normal(ar_pred, sigma_d[dyad_id_v]);
}

generated quantities {
  //
  // LOG-VRAISEMBLANCES (pour LOO/WAIC via ArviZ — légères, à garder)
  // 
  vector[N_h] log_lik_h;
  vector[N_v] log_lik_v;

  for (n in 1:N_h)
    log_lik_h[n] = bernoulli_logit_lpmf(is_mig[n] | logit_p[n]);

  for (n in 1:N_v)
    log_lik_v[n] = normal_lpdf(log_flow[n] | ar_pred[n], sigma_d[dyad_id_v[n]]);

  // 
  // PRÉDICTIONS IN-SAMPLE AVEC CORRECTION DE JENSEN 
  //
  // l'esperance est le meilleur prédicteur MSE. Ne pas l'utiliser, c'est se mettre des batons dans les roues.
  // On garde aussi is_mig_hat pour le PPC du hurdle (binaire, pas de Jensen)
  // 
  array[N_h] int is_mig_hat;
  vector[N_v] flow_hat_jensen;   // Prédiction corrigée (échelle réelle, migrants)

  for (n in 1:N_h)
    is_mig_hat[n] = bernoulli_logit_rng(logit_p[n]);

  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    // Correction de Jensen : exp(mu + sigma²/2)
    flow_hat_jensen[n] = exp(ar_pred[n] + 0.5 * square(sigma_d[d]));
  }

  // 
  // PRÉDICTIONS OOS LÉGÈRES (probabilité hurdle + mu_dt uniquement)
  // Le flow final est reconstruit en Python — aucun normal_rng ici
  // Cela évite de stocker N_test * iter * chains valeurs dans les .csv Stan
  //
  vector[N_test] prob_mig_test;    // P(flow > 0) pour chaque obs test
  vector[N_test] mu_dt_test;       // Espérance conditionnelle (log-échelle)
  vector[N_test] sigma_test;       // sigma utilisé pour chaque obs test

  for (n in 1:N_test) {
    int d_h = dyad_id_test_h[n];
    int d_v = dyad_id_test_v[n];

    // Hurdle : P(flow > 0)
    real logit_p_test = alpha_d[d_h]
                        + dot_product(X_h_test[n], beta_h)
                        + beta_lag_global * is_mig_lag_test[n];
    prob_mig_test[n] = inv_logit(logit_p_test);

    // Volume : mu_dt et sigma (paramètres suffisants pour la reconstruction Python)
    real mu_base = mu_intercept + dot_product(X_v_test[n], beta_grav);

    if (d_v > 0) {
      // Dyade existante : effet aléatoire propre + AR(1)
      real mu_full = alpha_V[d_v] + dot_product(X_v_test[n], beta_grav);

      if (is_mig_lag_test[n] > 0) {
        mu_dt_test[n] = mu_full
                        + phi_d[d_v] * (log_flow_lag_test[n] - mu_full);
      } else {
        mu_dt_test[n] = mu_full;
      }
      sigma_test[n] = sigma_d_oos[d_v];

    } else {
  int k = cluster_test_h[n];
  // On utilise la moyenne des alpha_V des dyades du même cluster
  // comme intercept de référence pour cette nouvelle dyade
  real mu_cluster = 0.0;
  int  n_cluster  = 0;
  for (d in 1:D_v) {
    if (cluster_v[d] == k) {
      mu_cluster += alpha_V[d];
      n_cluster  += 1;
    }
  }
  mu_cluster = (n_cluster > 0) ? mu_cluster / n_cluster : mu_intercept;
  mu_dt_test[n] = mu_cluster + dot_product(X_v_test[n], beta_grav);
  sigma_test[n] = sigma_cluster[k];
}
  }

  // Monitoring
  real phi_global_monitor = phi_global;
}
