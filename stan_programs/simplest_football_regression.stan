// Predicting the number of home goals scored by a team

data {
  int<lower=1> Ng; // Number of games
  int<lower=1> Nht; // Number of home teams
  int<lower=0> goals[Ng]; // home goals -integer real number array of size Ng
  int<lower=1, upper=Nht> hteam_ind[Ng]; // home teams - integer real number array of size Ng
  real alpha_mean;
  real<lower=0> alpha_sd;
  //real<lower=0> phi_mean;
  //real<lower=0> phi_sd;
  real hteam_mean; // prior mean number of goals scored by home teams
  real hteam_sigma; // prior variance of 
  
}

parameters {
  // This block is for specifying the primitives we're using for inference - in this case alpha and beta
  // lower and upper bounds here are hard logical constraints - soft constraits are exactly a prior!
  real alpha;
  real phi;
  vector[Nht] hteam_raw;
}

transformed parameters {
  // could declare 'eta' here if we want to save it
  vector[Nht] hteam = alpha + hteam_mean + hteam_sigma * hteam_raw;
  vector[Ng] mu = hteam[hteam_ind];
}

model {
  // model
  goals ~ neg_binomial_2_log(mu, phi);
  
  // Prioirs
  alpha ~ normal(alpha_mean, alpha_sd);
  //phi ~ normal(phi_mean, phi_sd);
  phi ~ normal(1,1);
  hteam_raw ~ normal(0, 1);
}

generated quantities {
  // this specified which values we're going to predict for. If they are the same as the data in then we're testing how well we can replicate the data.
  // if they aren't the same values as the data in, then they are predicting for a value.
  // here we are predicting the data points to compare the model preditions with the data
  //loop at the moment because random number generators are not vectorised.
  int pred_goals_rep[Ng];
  for (n in 1:Ng) {
    if (mu[n]>20) {
      reject("mu out of bounds, rejected...");
    }
    pred_goals_rep[n] = neg_binomial_2_log_rng(mu[n], phi);
  }
}
