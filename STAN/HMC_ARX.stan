/*
Bayesian Hierarchical ARX Hurdle Model for Gravity Migration

Component A: Hurdle (Hierarchical Bernoulli)
Component B: Volume (Hierarchical ARX Log-Normal)
Component C: Geographic Heteroscedasticity (Variance clustering)
*/

data {
  // 1. Hurdle Data (All dyads)
  int<lower=1> N_h;
  int<lower=1> D_h;
  int<lower=1> K_h;
  array[N_h] int<lower=1, upper=D_h> dyad_id_h;
  array[N_h] int<lower=0, upper=1> is_mig;
  vector[N_h] is_mig_lag;
  matrix[N_h, K_h] X_h;

  // 2. Volume Data (Dyads with strictly positive flow)
  int<lower=1> N_v;
  int<lower=1> D_v;
  int<lower=1> K_v;
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;
  vector[N_v] log_flow;
  vector[N_v] log_flow_lag;
  matrix[N_v, K_v] X_v;

  // 3. Geographic Clusters
  int<lower=1> K_clusters;
  array[D_h] int<lower=1, upper=K_clusters> cluster_h;
  array[D_v] int<lower=1, upper=K_clusters> cluster_v;

  // 4. Out-of-Sample Test Data
  int<lower=0> N_test;
  array[N_test] int<lower=1, upper=D_h> dyad_id_test;
  matrix[N_test, K_h] X_h_test;
  vector[N_test] is_mig_lag_test;
  matrix[N_test, K_v] X_v_test;
  vector[N_test] log_flow_lag_test;
  array[N_test] int<lower=0, upper=D_v> dyad_id_test_v;
}

parameters {
  // A. Hurdle Parameters
  real alpha_global;
  real<lower=0> tau_alpha;
  vector[K_h] beta_h;
  real beta_lag_global;
  vector[D_h] alpha_raw; // Non-centered parameterization

  // B. Volume Parameters (Gravity + ARX)
  real mu_intercept;
  real<lower=0> tau_mu;
  vector[K_v] beta_grav;
  
  real phi_global_raw;
  real<lower=0> tau_phi;
  vector[D_v] phi_raw;
  vector[D_v] mu_raw;

  // C. Heteroscedasticity Parameters
  real<lower=0> sigma_global;
  vector<lower=0>[K_clusters] sigma_cluster;
  vector[D_v] sigma_raw;
  real<lower=0> tau_sigma;
}

transformed parameters {
  // A. Hurdle Random Effects
  vector[D_h] alpha_d;
  for (d in 1:D_h)
    alpha_d[d] = alpha_global + tau_alpha * alpha_raw[d];

  // B. Volume Intercepts & AR(1) Persistences
  vector[D_v] alpha_V;
  vector[D_v] phi_d;
  real phi_global = tanh(phi_global_raw); // Constrained to (-1, 1) for stationarity

  for (d in 1:D_v) {
    alpha_V[d] = mu_intercept + tau_mu * mu_raw[d];
    phi_d[d]   = tanh(phi_global_raw + tau_phi * phi_raw[d]);
  }

  // C. Volatility by Dyad
  vector<lower=0>[D_v] sigma_d;
  for (d in 1:D_v)
    sigma_d[d] = sigma_cluster[cluster_v[d]] * exp(tau_sigma * sigma_raw[d]);

  // Linear Predictors
  vector[N_h] logit_p = alpha_d[dyad_id_h] + X_h * beta_h + beta_lag_global * is_mig_lag;
  vector[N_v] mu_dt = alpha_V[dyad_id_v] + X_v * beta_grav;
  vector[N_v] ar_pred;
  
  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    ar_pred[n] = mu_dt[n] + phi_d[d] * (log_flow_lag[n] - mu_dt[n]);
  }
}

model {
  // A. Hurdle Priors
  alpha_global ~ normal(0.5, 2);
  tau_alpha ~ exponential(1);
  beta_h[1] ~ normal(-1, 1);
  beta_h[2] ~ normal(2, 1);
  beta_h[3] ~ normal(0.5, 1);
  beta_lag_global ~ normal(1.5, 1);
  alpha_raw ~ std_normal();

  // B. Volume Priors
  mu_intercept ~ normal(8, 3);
  tau_mu ~ exponential(1);
  beta_grav ~ normal(0, 1);
  
  phi_global_raw ~ normal(1, 0.5);
  tau_phi ~ exponential(2);
  phi_raw ~ std_normal();
  mu_raw ~ std_normal();

  // C. Geographic Variance Priors
  sigma_global ~ exponential(1);
  for (k in 1:K_clusters)
    sigma_cluster[k] ~ normal(sigma_global, 0.5);
  tau_sigma ~ exponential(2);
  sigma_raw ~ std_normal();

  // Likelihoods
  is_mig ~ bernoulli_logit(logit_p);
  log_flow ~ normal(ar_pred, sigma_d[dyad_id_v]);
}

generated quantities {
  // 1. In-Sample Predictions & Log-Likelihoods
  vector[N_h] log_lik_h;
  vector[N_v] log_lik_v;
  array[N_h] int is_mig_hat;
  vector[N_v] log_flow_hat;

  for (n in 1:N_h) {
    is_mig_hat[n] = bernoulli_logit_rng(logit_p[n]);
    log_lik_h[n]  = bernoulli_logit_lpmf(is_mig[n] | logit_p[n]);
  }

  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    log_flow_hat[n] = normal_rng(ar_pred[n], sigma_d[d]);
    log_lik_v[n]    = normal_lpdf(log_flow[n] | ar_pred[n], sigma_d[d]);
  }

  real phi_global_monitor = phi_global;

  // 2. Out-of-Sample Predictions
  vector[N_test] prob_mig_test;
  array[N_test] int is_mig_test_hat;
  vector[N_test] log_flow_test_hat;
  vector[N_test] flow_test_hat;

  for (n in 1:N_test) {
    int d_h = dyad_id_test[n];
    int d_v = dyad_id_test_v[n];

    // Hurdle step
    real logit_p_test = alpha_d[d_h] + dot_product(X_h_test[n], beta_h) + beta_lag_global * is_mig_lag_test[n];
    prob_mig_test[n] = inv_logit(logit_p_test);
    is_mig_test_hat[n] = bernoulli_logit_rng(logit_p_test);

    // Volume step (ARX)
    real mu_dt_test;
    real ar_pred_test;
    real sim_sigma;

    if (d_v > 0) {
      // Existing dyad: use local historical parameters
      mu_dt_test = alpha_V[d_v] + dot_product(X_v_test[n], beta_grav);
      sim_sigma = sigma_d[d_v];
      
      if (is_mig_lag_test[n] == 1) {
        ar_pred_test = mu_dt_test + phi_d[d_v] * (log_flow_lag_test[n] - mu_dt_test);
      } else {
        ar_pred_test = mu_dt_test;
      }
    } else {
      // New entrant fallback: use global averages and cluster variance
      mu_dt_test = mu_intercept + dot_product(X_v_test[n], beta_grav);
      ar_pred_test = mu_dt_test;
      sim_sigma = sigma_cluster[cluster_h[d_h]]; 
    }
    
    log_flow_test_hat[n] = normal_rng(ar_pred_test, sim_sigma);

    // Recombination
    if (is_mig_test_hat[n] == 1) {
      flow_test_hat[n] = exp(log_flow_test_hat[n]);
    } else {
      flow_test_hat[n] = 0.0;
    }
  }
}