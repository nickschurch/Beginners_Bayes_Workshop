// Predicting the number of home goals scored by a team

data {
  int<lower=1> Ng; // Number of games
  int<lower=1> Nt; // Number of teams
  int<lower=0> hgoals[Ng]; // home goals -integer real number array of size Ng
  int<lower=0> agoals[Ng]; // away goals -integer real number array of size Ng
  int goaldiff[Ng]; // goal differnce -integer real number array of size Ng
  int<lower=1, upper=Nt> hteam_ind[Ng]; // home teams - integer real number array of size Ng
  int<lower=1, upper=Nt> ateam_ind[Ng]; // away teams - integer real number array of size Ng
  real<lower=0> beta_sd;
  real hteam_mean; // prior mean number of goals scored by home teams
  real hteam_sigma; // prior variance of 
  real ateam_mean; // prior mean number of goals scored by away teams
  real ateam_sigma; // prior variance of 
  real a_alpha_mean; // prior mean shoft of goals scored by away teams
  real a_alpha_sd; // prior variance of 
  real h_alpha_mean; // prior mean shoft of goals scored by away teams
  real h_alpha_sd; // prior variance of 
}

parameters {
  // This block is for specifying the primitives we're using for inference - in this case alpha and beta
  // lower and upper bounds here are hard logical constraints - soft constraits are exactly a prior!
  real a_alpha;
  real h_alpha;
  real a_phi;
  real h_phi;
  vector[Nt] ateam_raw;
  vector[Nt] hteam_raw;
}

transformed parameters {
  // could declare 'eta' here if we want to save it
  vector[Nt] hteam = h_alpha + hteam_mean + hteam_sigma * hteam_raw;
  vector[Ng] h_mu = hteam[hteam_ind];
  vector[Nt] ateam = a_alpha + ateam_mean + ateam_sigma * ateam_raw;
  vector[Ng] a_mu = ateam[ateam_ind];
}

model {
  // model
  hgoals ~ neg_binomial_2_log(h_mu, h_phi);
  agoals ~ neg_binomial_2_log(a_mu, a_phi);

  // Prioirs
  h_alpha ~ normal(h_alpha_mean, h_alpha_sd);
  a_alpha ~ normal(a_alpha_mean, a_alpha_sd);
  h_phi ~ normal(5,1);
  a_phi ~ normal(5,1);
  hteam_raw ~ normal(0, 1);
  ateam_raw ~ normal(0, 1);
}

generated quantities {
  // this specified which values we're going to predict for. If they are the same as the data in then we're testing how well we can replicate the data.
  // if they aren't the same values as the data in, then they are predicting for a value.
  // here we are predicting the data points to compare the model preditions with the data
  //loop at the moment because random number generators are not vectorised.
  int pred_goaldiff_rep[Nt*Nt];
  int pred_goaldiff_rep_2d[Nt,Nt];
  int i = 1;
  for (hnt in 1:Nt) {
    for (ant in 1:Nt) {
      if (h_mu[hnt]>20) {
        reject("mu out of bounds, rejected...");
      }
      if (a_mu[ant]>20) {
        reject("mu out of bounds, rejected...");
      }
      pred_goaldiff_rep[i] = hgoals[hnt] - agoals[ant];
      pred_goaldiff_rep_2d[hnt,ant] = hgoals[hnt] - agoals[ant];
      i = i+1;
    }
  }
}
