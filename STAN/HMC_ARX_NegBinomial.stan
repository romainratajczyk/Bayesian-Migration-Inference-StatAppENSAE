// vectoriser avec produit de Hadamard pour gagner en rapidité de calcul (à faire) 





// Point1: but de ce code: dans HMC_ARX_v4 la variance est calculée comme sigma_d[d] = fmax(sigma_cluster[...] * exp(...), 1e-4). Ne dépend pas de la magnitude du flux.
// on pense que c'est pour cela que la distribution des erreurs est positivement corrélée à la magnitude du flux pour les flux > 15 000. 

// Point 2: Aussi, le but premier est de remplacer la loi log-normale par une loi Negative Binomiale, qui gère nativement la nature discrète des micro-flux et converge asymptotiqueemnt vers une allure gaussienne qui ressemble à la log-normale précédente pour les macro-flux. 
// l'hétéroscédasticité est aussi native dans la loi NegBin! le point 1 est donc redondant. 

// Point3: Enfin, traiter de manière hiérarchique (comme les autres paramètres) le coefficient beta_lag du Hurdle. 

/*
Bayesian Hierarchical ARX Hurdle Model for Gravity Migration
Component A: Hurdle (Hierarchical Bernoulli)
Component B: Volume (Hierarchical ARX Zero-Truncated Negative Binomial avec Effets Origine/Destination)
Component C: Geographic Heteroscedasticity (Dispersion clustering)
*/

data {
  // Paramètres dimensionnels 
  int<lower=1> N_pays;
  // Hyper-régression
  int<lower=1> K_Z; 
  matrix[N_pays, K_Z] Z_em;
  matrix[N_pays, K_Z] Z_at;

  // Hurdle (classification binaire avec les variables X_h et is_mig)
  int<lower=1> N_h;
  int<lower=1> D_h;
  int<lower=1> K_h;
  array[N_h] int<lower=1, upper=D_h> dyad_id_h;
  array[N_h] int<lower=1, upper=N_pays> orig_id_h; 
  array[N_h] int<lower=1, upper=N_pays> dest_id_h;
  array[N_h] int<lower=0, upper=1>   is_mig;
  vector[N_h]                        is_mig_lag;
  matrix[N_h, K_h]                   X_h;

  //  Volume ZTNB: restriction à N^*
  int<lower=1> N_v;
  int<lower=1> D_v;
  int<lower=1> K_v;
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;
  array[N_v] int<lower=1, upper=N_pays> orig_id_v; // pour indexer l'effet d'émission
  array[N_v] int<lower=1, upper=N_pays> dest_id_v; // pour indexer l'effet d'attraction
  array[N_v] int<lower=1>            flow;              
  vector[N_v]                        log_flow_lag;      
  matrix[N_v, K_v]                   X_v;

  //  Clusters géographiques (M49 onusiens)
  int<lower=1>                       K_clusters;
  array[D_h] int<lower=1, upper=K_clusters> cluster_h;
  array[D_v] int<lower=1, upper=K_clusters> cluster_v;

  //  Test OOS
  int<lower=0>                       N_test;
  array[N_test] int<lower=1, upper=D_h>     dyad_id_test_h;
  array[N_test] int<lower=0, upper=D_v>     dyad_id_test_v;
  array[N_test] int<lower=1, upper=N_pays>  orig_id_test_v;
  array[N_test] int<lower=1, upper=N_pays>  dest_id_test_v;
  matrix[N_test, K_h]                       X_h_test;
  vector[N_test]                            is_mig_lag_test;
  matrix[N_test, K_v]                       X_v_test;
  vector[N_test]                            log_flow_lag_test;
  array[N_test] int<lower=1, upper=K_clusters> cluster_test_h;

  //  Flags (interrupteurs)
  int<lower=0, upper=1> do_ppc;
  int<lower=0, upper=1> do_loo; // interrupteur log_lik
}

parameters {
  // A. Hurdle

  //real alpha_global;
  //real<lower=0> tau_alpha;

  vector[K_h] beta_h; // variables de X_h du hurdle
  real mu_beta_lag;                     
  real<lower=0> sigma_beta_lag;         
  vector[K_clusters] beta_lag_raw;     // autant de beta_lag que de clusters M49
  // vector[D_h] alpha_raw; // un alpha_raw par dyade (l'ADN d'une dyade, généré d'un prior)

// Hyper-parametres Hurdle (Emission/Attraction)
  real intercept_h_em;
  vector[K_Z] theta_h_em;
  real<lower=0> tau_h_em;
  vector[N_pays] alpha_h_em_raw; 
  
  real intercept_h_at;
  vector[K_Z] theta_h_at;
  real<lower=0> tau_h_at;
  vector[N_pays] gamma_h_at_raw;


  // B. Volume ARX (Effets Emission/Attraction)

  real intercept_em;
  vector[K_Z] theta_em;
  real<lower=0> tau_em;
  vector[N_pays] alpha_em_raw; 
  
  real intercept_at;
  vector[K_Z] theta_at;
  real<lower=0> tau_at;
  vector[N_pays] gamma_at_raw;

  vector[K_v] beta_grav;
  real rho_global_raw;
  real<lower=0> tau_rho;
  vector[D_v] rho_raw;

  // C. Dispersion (Phi remplace Sigma la variance du modèle log-normal précédent)
  real<lower=0> phi_disp_global;
  vector<lower=0>[K_clusters] phi_disp_cluster;
  vector[D_v] phi_disp_raw;
  real<lower=0> tau_phi_disp;           
}

transformed parameters {
  // A. Prédicteurs Hurdle
  
  //vector[D_h] alpha_d;
  //for (d in 1:D_h)
  //  alpha_d[d] = alpha_global + tau_alpha * alpha_raw[d];
  vector[K_clusters] beta_lag_m49 = mu_beta_lag + sigma_beta_lag * beta_lag_raw;

// Calcul des effets pays Hurdle
  vector[N_pays] mu_h_em_vec = intercept_h_em + Z_em * theta_h_em;
  vector[N_pays] mu_h_at_vec = intercept_h_at + Z_at * theta_h_at;
  
  vector[N_pays] alpha_h_em = mu_h_em_vec + tau_h_em * alpha_h_em_raw;
  vector[N_pays] gamma_h_at = mu_h_at_vec + tau_h_at * gamma_h_at_raw;



  // B. Prédicteurs Volume ARX
  real rho_global = tanh(rho_global_raw); // ramener dans l'intervalle (-1, 1) pour la stationnarité de l'AR(1)
  
  // calcul des effets pays
// Hyper-espérances vectorielles
  vector[N_pays] mu_em_vec = intercept_em + Z_em * theta_em;
  vector[N_pays] mu_at_vec = intercept_at + Z_at * theta_at;

  // Calcul des effets pays (Shrinkage vers Z*theta)
  vector[N_pays] alpha_em = mu_em_vec + tau_em * alpha_em_raw;
  vector[N_pays] gamma_at = mu_at_vec + tau_at * gamma_at_raw;

  vector[D_v] rho_d;
  for (d in 1:D_v) {
    rho_d[d] = tanh(rho_global_raw + tau_rho * rho_raw[d]); // injecter l'effet hiérarchie cluster dans la tanh
  }

  // C. Dispersion Hiérarchique
  vector<lower=0>[D_v] phi_disp_d;
  for (d in 1:D_v) {
    phi_disp_d[d] = phi_disp_cluster[cluster_v[d]] * exp(tau_phi_disp * phi_disp_raw[d]); // éviter les valeurs extrêmes de dispersion qui posent problème pour la négative binomiale (et qui sont de toute façon invraisemblables)
  }

  // Construction vectorisée des prédicteurs
  vector[N_h] lag_effect; 
  for (n in 1:N_h) {
    lag_effect[n] = beta_lag_m49[cluster_h[dyad_id_h[n]]] * is_mig_lag[n];
  }
  
  vector[N_h] logit_p = alpha_h_em[orig_id_h] + gamma_h_at[dest_id_h] + X_h * beta_h + lag_effect;

  // Substitution dyadique par l'addition des marginales : alpha_i + gamma_j
  vector[N_v] mu_dt = alpha_em[orig_id_v] + gamma_at[dest_id_v] + X_v * beta_grav;

  vector[N_v] ar_pred; 
  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    ar_pred[n] = mu_dt[n] + rho_d[d] * (log_flow_lag[n] - mu_dt[n]);
  }
}

model {
// A. Priors Hurdle
  
  intercept_h_em ~ normal(-1.0, 1.5);
  theta_h_em     ~ normal(0, 0.5);
  tau_h_em       ~ normal(0, 0.25); 
  alpha_h_em_raw ~ std_normal();
  
  intercept_h_at ~ normal(0, 1.0);
  theta_h_at     ~ normal(0, 0.5);
  tau_h_at       ~ normal(0, 0.25);
  gamma_h_at_raw ~ std_normal();

  beta_h[1]      ~ normal(-0.5, 0.5); // 1. Distance
  beta_h[2]      ~ normal(-0.5, 0.5); // 2. distance^2
  beta_h[3]      ~ normal(0, 2); // 3. Frontière commune
  beta_h[4]      ~ normal(0, 2); // 4. Interaction frontière_commune*distance
  beta_h[5]      ~ normal(0, 2); // 5. Colonie
  beta_h[6]      ~ normal(0, 2); // 6. Langue officielle
  beta_h[7] ~ normal(1.0, 1.0); // Prior positif forcé pour 'logit_rf'. Le ML est présumé prédictif
  beta_h[8] ~ normal(1.0, 1.0); // Prior strictement positif forcé pour log_TC_lag
  // Variables 7 à 9 supprimées de beta_h car transférées dans Z_em / Z_at
  
  mu_beta_lag    ~ normal(2.0, 2.5); // definition du prior à discuter
  sigma_beta_lag ~ exponential(1);
  beta_lag_raw   ~ std_normal();


  // B. Priors Volume (Emission / Attraction)

  // B. Priors Volume (Emission / Attraction via Hyper-régression)
  intercept_em ~ normal(0, 1);
  theta_em     ~ normal(0, 0.5);
  tau_em       ~ normal(0, 0.25); // half normal (tronqué car lower=0 déclaré)
  alpha_em_raw ~ std_normal();  // equivalent strict et optimisé d'une boucle "for (p in 1:N_pays) alpha_em_raw[p] ~ normal(0, 1);""
  
  intercept_at ~ normal(0, 1);
  theta_at     ~ normal(0, 0.5);
  tau_at       ~ normal(0, 0.25); 
  gamma_at_raw ~ std_normal();
  

  beta_grav      ~ normal(0, 1);
  rho_global_raw ~ normal(0.5, 0.5);
  tau_rho        ~ exponential(2);
  rho_raw        ~ std_normal();

  // C. Priors Dispersion
  phi_disp_global ~ exponential(1);
  for (k in 1:K_clusters)
    phi_disp_cluster[k] ~ normal(phi_disp_global, 0.5);
    
  tau_phi_disp ~ exponential(2);
  phi_disp_raw ~ std_normal();

  // Vraisemblances
  is_mig ~ bernoulli_logit(logit_p);

  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    target += neg_binomial_2_log_lpmf(flow[n] | ar_pred[n], phi_disp_d[d]);
    target += -log1m_exp(neg_binomial_2_log_lpmf(0 | ar_pred[n], phi_disp_d[d])); // += - car -= pas géré par Stan
  }
}

generated quantities {
  // Allocation dynamique : Taille 0 si do_loo = 0 pour désactiver l'écriture disque
  vector[do_loo ? N_h : 0] log_lik_h;
  vector[do_loo ? N_v : 0] log_lik_v;

  if (do_loo) {
    for (n in 1:N_h)
      log_lik_h[n] = bernoulli_logit_lpmf(is_mig[n] | logit_p[n]);

    for (n in 1:N_v) {
      int d = dyad_id_v[n];
      log_lik_v[n] = neg_binomial_2_log_lpmf(flow[n] | ar_pred[n], phi_disp_d[d]) 
                     - log1m_exp(neg_binomial_2_log_lpmf(0 | ar_pred[n], phi_disp_d[d]));
    }
  }

  array[do_ppc ? N_h : 0] int is_mig_hat;
  if (do_ppc) {
    for (n in 1:N_h)
      is_mig_hat[n] = bernoulli_logit_rng(logit_p[n]);
  }

  // Prédictions OOS pour les dyades de test
  vector[N_test] prob_mig_test;
  vector[N_test] mu_dt_test;     
  vector[N_test] phi_test;       

  for (n in 1:N_test) {
    int d_h = dyad_id_test_h[n];
    int d_v = dyad_id_test_v[n];
    int k = cluster_test_h[n];

    real logit_p_test = alpha_h_em[orig_id_test_v[n]] + gamma_h_at[dest_id_test_v[n]]
                        + dot_product(X_h_test[n], beta_h)
                        + beta_lag_m49[k] * is_mig_lag_test[n];
    prob_mig_test[n] = inv_logit(logit_p_test);

    // Résolution immédiate des effets pays (valide même pour une dyade jamais vue en Train)
    real mu_full = alpha_em[orig_id_test_v[n]] + gamma_at[dest_id_test_v[n]] + dot_product(X_v_test[n], beta_grav);

    if (d_v > 0) {
      if (is_mig_lag_test[n] > 0) {
        mu_dt_test[n] = mu_full + rho_d[d_v] * (log_flow_lag_test[n] - mu_full);
      } else { 
        mu_dt_test[n] = mu_full; // gravité pure, aucune inertie 
      }
      phi_test[n] = phi_disp_d[d_v];
    } else {
      mu_dt_test[n] = mu_full;
      phi_test[n] = phi_disp_cluster[k];
    }
  }

  real rho_global_monitor = rho_global;
}