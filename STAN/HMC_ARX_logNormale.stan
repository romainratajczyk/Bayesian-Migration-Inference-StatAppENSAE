/*
Bayesian Hierarchical ARX Hurdle Model for Gravity Migration

Component A: Hurdle (Hierarchical Bernoulli)
Component B: Volume (Hierarchical ARX Log-Normal)
Component C: Geographic Heteroscedasticity (Variance clustering)
*/



/*
Voir notes cahier pour les dernières modifications & optimisation
*/

data {  // arguments d'entrée du modèle, préparés en Python: doivent correspondre au dictionnaire stan_data de Py

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

  // 5. flag: si do_ppc=0, Stan ne calculera pas les predictions in-sample: économie de temps et de RAM
  int<lower=0, upper=1> do_ppc;  // 1 = activer les prédictions in-sample (PPC), 0 = production
}

parameters {       // Paramètres à estimer par le modèle (échantillonnage ici) 
  // A. Hurdle
  real alpha_global;
  real<lower=0> tau_alpha;
  vector[K_h] beta_h;
  vector[K_clusters] beta_lag_continent;
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

transformed parameters {     // les paramètres transformés, calculs intermédiaires,  utilisés pour la vraisemblance et les prédictions
          // équations hiérarchiques ici, juste avant de calculer la log-vraisemblance et conclure avec Metropolis  
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
    phi_d[d]   = tanh(phi_global_raw + tau_phi * phi_raw[d]); // garanti prior modéré et autour de 0.5, et stationnaire
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
  // quand phi→1, sigma_oos → ∞ correctement (très persistant = très incertain)

  // inflation modérée et plafonnée
  vector<lower=0>[D_v] sigma_d_oos;
  for (d in 1:D_v) {

    // Formule exacte à h=1 pas AR(1) : sigma * sqrt(1 + phi²) (la valeur de demain dépend d'aujoiurd'hui, mais a EN plus sa propre incertidude)
   
    // Avec phi=0.95 : facteur = sqrt(1 + 0.90) ≈ 1.38 

    // Plafond à 3×sigma_d pour éviter les explosions numériques

    real inflation = sqrt(1.0 + square(phi_d[d]));
    sigma_d_oos[d] = fmin(sigma_d[d] * inflation, 3.0 * sigma_d[d]);
  }

  // Prédicteurs linéaires (vectorisés)
  vector[N_h] lag_effect;
  for (n in 1:N_h) {
  // On va chercher le continent de la dyade, puis on applique l'inertie
    lag_effect[n] = beta_lag_continent[cluster_h[dyad_id_h[n]]] * is_mig_lag[n];
  }
  vector[N_h] logit_p = alpha_d[dyad_id_h] + X_h * beta_h + lag_effect;

  vector[N_v] mu_dt = alpha_V[dyad_id_v] + X_v * beta_grav;

  vector[N_v] ar_pred;
  for (n in 1:N_v) {
    int d = dyad_id_v[n];
    ar_pred[n] = mu_dt[n] + phi_d[d] * (log_flow_lag[n] - mu_dt[n]);
  }
}

model {   // log-vraisemblance et les priors. Le coeur du modèle hiérarchique 
  // A — Priors hurdle
  alpha_global    ~ normal(0.5, 2);
  tau_alpha       ~ exponential(1);
  beta_h[1]       ~ normal(-1, 1);
  beta_h[2]       ~ normal(2, 1);
  beta_h[3]       ~ normal(0.5, 1);
  beta_h[4]       ~ normal(1, 1);    // COL_ij (Colonie : facilitateur)
  beta_h[5]       ~ normal(1, 1);    // OL_ij (Langue commune : facilitateur)
  beta_h[6]       ~ normal(0.5, 1);  // log_pop_o (Masse démo origine)
  beta_h[7]       ~ normal(0.5, 1);  // log_pop_d (Masse démo destination)
  beta_h[8]       ~ normal(0.5, 1);  // log_gdp_d (Attractivité richesse destination)
  beta_lag_continent ~ normal(3.0, 2.5); // OK on a déjà observé des valeurs >5 voir 6, mais attention à l'Empirical Bayes "mal assumé": on utilise pas les posteriors pour informer le prior. 
  alpha_raw       ~ std_normal();

  // B — Priors volume
  mu_intercept   ~ normal(0, 2);
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
  tau_sigma ~ exponential(0.4); // à tester exponential(0.5) passé à exponential(0.2) moins informatif pour coverage moins étroits (esperance = 1/lambda) 
  sigma_raw ~ std_normal();

  // Vraisemblances
  is_mig   ~ bernoulli_logit(logit_p); // entrainement de la décision Hurdle (ajuste les coeffs beta) sur N_h (tous les flux)
  log_flow ~ normal(ar_pred, sigma_d[dyad_id_v]); // entrainement du volume sur N_v (conditionné à flux>0) pour avoir le bon AR(1)
}

generated quantities {    // calculs post-sampling, prédictions in-sample et out-of-sample. On ne génére pas tout, numpy le fera plus vite 
  //
  // LOG-VRAISEMBLANCES (pour LOO/WAIC via ArviZ — légères, à garder)
  //
// LOG-VRAISEMBLANCES (Corrigées par le Jacobien pour comparabilité avec ZTNB)
  vector[N_h] log_lik_h;
  vector[N_v] log_lik_v;

  for (n in 1:N_h)
    log_lik_h[n] = bernoulli_logit_lpmf(is_mig[n] | logit_p[n]);

  for (n in 1:N_v) {
    // Calcul de la densité sur l'espace log-normal, rapportée à l'espace réel Y
    log_lik_v[n] = normal_lpdf(log_flow[n] | ar_pred[n], sigma_d[dyad_id_v[n]]) - log_flow[n];
  }
  //
  // PRÉDICTIONS IN-SAMPLE AVEC CORRECTION DE JENSEN
  //
  // l'esperance est le meilleur prédicteur MSE. Ne pas l'utiliser, c'est se mettre des batons dans les roues.
  // On garde aussi is_mig_hat pour le PPC du hurdle (binaire, pas de Jensen)
  // Désactivées en production (do_ppc=0) pour économiser ~3–5 GB de RAM
  //
  array[do_ppc ? N_h : 0] int is_mig_hat;
  vector[do_ppc ? N_v : 0] flow_hat_jensen;   // Prédiction corrigée (échelle réelle, migrants)

  if (do_ppc) {
    for (n in 1:N_h)
      is_mig_hat[n] = bernoulli_logit_rng(logit_p[n]);

    for (n in 1:N_v) {
      int d = dyad_id_v[n];
      // Correction de Jensen : exp(mu + sigma²/2)
      flow_hat_jensen[n] = exp(ar_pred[n] + 0.5 * square(sigma_d[d]));
    }
  }

  //
  // PRÉDICTIONS OOS LÉGÈRES (probabilité hurdle + mu_dt uniquement)
  // Le flow final est reconstruit en Python — aucun normal_rng ici
  // Cela évite de stocker N_test * iter * chains valeurs dans les .csv Stan
  // mu_cluster_k précalculé dans transformed parameters : la boucle O(N_test × D_v)
  // qui paralysait le sampler à 190 pays est supprimée
  


  // Moyenne de alpha_V par cluster — précalculée ici pour éviter la boucle
  // O(N_test × D_v) dans generated quantities (le vrai goulot d'étranglement à 190 pays)
  vector[K_clusters] mu_cluster_k;
  {
    vector[K_clusters] n_cluster_k = rep_vector(0.0, K_clusters);
    mu_cluster_k = rep_vector(0.0, K_clusters);
    for (d in 1:D_v) {
      mu_cluster_k[cluster_v[d]] += alpha_V[d];
      n_cluster_k[cluster_v[d]]  += 1.0;
    }
    for (k in 1:K_clusters) {
      if (n_cluster_k[k] > 0)
        mu_cluster_k[k] /= n_cluster_k[k];
      else
        mu_cluster_k[k] = mu_intercept;
    }
  }


  vector[N_test] prob_mig_test;    // P(flow > 0) pour chaque obs test
  vector[N_test] mu_dt_test;       // Espérance conditionnelle (log-échelle)
  vector[N_test] sigma_test;       // sigma utilisé pour chaque obs test

  for (n in 1:N_test) {
    int d_h = dyad_id_test_h[n];
    int d_v = dyad_id_test_v[n];

    // Hurdle : P(flow > 0)
    real logit_p_test = alpha_d[d_h]
                        + dot_product(X_h_test[n], beta_h)
                        + beta_lag_continent[cluster_test_h[n]] * is_mig_lag_test[n];
    prob_mig_test[n] = inv_logit(logit_p_test);

    // Volume : mu_dt et sigma (paramètres suffisants pour la reconstruction Python)
    real mu_base = mu_intercept + dot_product(X_v_test[n], beta_grav);

    if (d_v > 0) { // d_v>0 signifie que le couloir existe 
      //effet aléatoire  + AR(1)
      real mu_full = alpha_V[d_v] + dot_product(X_v_test[n], beta_grav);

      if (is_mig_lag_test[n] > 0) {
        mu_dt_test[n] = mu_full
                        + phi_d[d_v] * (log_flow_lag_test[n] - mu_full);
      } else { 
        mu_dt_test[n] = mu_full;
      }
      sigma_test[n] = sigma_d_oos[d_v];

    } else { // si la dyade n'existe pas avant 2015, on se rabat sur les données continentales 
      // intercept = moyenne des alpha_V du même cluster
      // lu directement depuis mu_cluster_k (précalculé dans transformed parameters)
      int k = cluster_test_h[n];
      mu_dt_test[n] = mu_cluster_k[k] + dot_product(X_v_test[n], beta_grav);
      sigma_test[n] = sigma_cluster[k];
    }
  }

  // Monitoring
  real phi_global_monitor = phi_global;
}
