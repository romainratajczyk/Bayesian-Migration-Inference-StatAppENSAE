// 1) DATA
data {
  int<lower=0> N;                // Nombre d'observations 
  int<lower=0> K;                // Nombre de variables explicatives 
  matrix[N, K] X;                // matrice contenant toutes les variables (Pop, Dist, PIB...)
  vector[N] y;                   
}

// 2) PARAMETERS
parameters {
  real alpha;                    // L'intercept
  vector[K] beta;                // Un vecteur contenant les 20 coeffs (un par variable)
  real<lower=0> sigma;           // L'écart-type de l'erreur
}

// 3) MODEL
model {
  // Priors 
  // On suppose que les effets sont raisonnables (entre -10 et +10 environ)
  beta ~ normal(0, 5);           
  alpha ~ normal(0, 10);
  sigma ~ cauchy(0, 2.5);        

  // Vraisemblance 
  // X * beta fait tout le calcul d'un coup
  y ~ normal(alpha + X * beta, sigma);
}





// exemples jags / exemples dans gelman