data {
  int<lower=1> N;           // Nombre total d'observations (Dyades * Temps)
  int<lower=1> D;           // Nombre de dyades (ex: 110)
  array[N] int dyad_id;     // ID de la dyade pour chaque observation (de 1 à 110)
  vector[N] log_y;          // Log du flux à l'instant t
  vector[N] log_y_lag;      // Log du flux à l'instant t-1
}

parameters {
  // Paramètres globaux (Hyperparamètres)
  real mu_global;
  real<lower=0> sigma_mu;
  
  // Paramètres spécifiques aux 110 dyades
  vector[D] mu_dyad;                   // Constante d'équilibre de la dyade
  vector<lower=-1, upper=1>[D] phi;    // Inertie AR(1) (entre -1 et 1 pour la stabilité)
  vector<lower=0>[D] sigma_dyad;       // Bruit spécifique à la dyade
}

model {
  // 1. Priors (Lois a priori)
  mu_global ~ normal(0, 5);
  sigma_mu ~ exponential(1);
  
  // Modèle hiérarchique : les mu de chaque dyade viennent d'une distribution globale
  mu_dyad ~ normal(mu_global, sigma_mu);
  
  // Priors pour l'inertie et la variance
  phi ~ uniform(-1, 1);
  sigma_dyad ~ exponential(1);
  
  // 2. Vraisemblance (Likelihood)
  for (n in 1:N) {
    log_y[n] ~ normal(mu_dyad[dyad_id[n]] + phi[dyad_id[n]] * log_y_lag[n], sigma_dyad[dyad_id[n]]);
  }
}



// exemples jags / exemples dans gelman