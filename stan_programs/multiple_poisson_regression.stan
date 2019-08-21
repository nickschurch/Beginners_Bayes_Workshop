data {
  // currently the <> keywords are limited to lower and upper bounds and apply to the values of each element of an array
  int<lower=1> N;
  int<lower=0> complaints[N]; // integer real number array of size N
  vector<lower=0>[N] traps; // vector of real numbers of size N - not restricted to integers! Can add [X] after traps to specify an array of vectors
  real alpha_mean;
  real alpha_sd;
  real beta_mean;
  real beta_sd;
  vector<lower=0>[N] log_sq_foot;
  vector<lower=0, upper=1>[N] live_in_super;
  real beta_super_mean;
  real beta_super_sd;
}

parameters {
  // This block is for specifying the primitives we're using for inference - in this case alpha and beta
  // lower and upper bounds here are hard logical constraints - soft constraits are exactly a prior!
  real alpha; // intercept
  real beta; // slope
  real beta_super; // slope for relation with a live-in super
}

transformed parameters {
  // could declare 'eta' here if we want to save it 
}

model {
  // poisson_log(x) is more efficient and stable alternative to poisson(exp(x))
  // complaints ~ poisson(exp(alpha + beta * traps));
  complaints ~ poisson_log(alpha + beta * traps + log_sq_foot + beta_super * live_in_super); // vectorised equation form - could have written this with loop indecies
  
  // weakly informative priors:
  // we expect negative slope on traps and a positive intercept,
  // but we will allow ourselves to be wrong
  // Taken from the prior check
  alpha ~ normal(alpha_mean, alpha_sd);
  beta ~ normal(beta_mean, beta_sd);
  beta_super ~ normal(beta_super_mean, beta_super_sd);
}

generated quantities {
  // this specified which values we're going to predict for. If they are the same as the data in then we're testing how well we can replicate the data.
  // if they aren't the same values as the data in, then they are predicting for a value.
  // here we are predicting the data points to compare the model preditions with the data
  //loop at the moment because random number generators are not vectorised.
  int pred_complaints_rep[N];
  for (n in 1:N) {
    real eta = alpha + beta * traps[n] + log_sq_foot[n] + beta_super * live_in_super[n];
    if (eta>20) {
      reject("eta out of bounds, rejected...");
    }
    pred_complaints_rep[n] = poisson_log_rng(eta);
  }
}
