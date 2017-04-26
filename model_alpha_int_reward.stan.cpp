
// SIMPLE SOFTMAX: LEARNING RATE ~ REWARD

// Modified from code by Bradley Doll (https://github.com/dollbb/estRLParam)

data {
    int<lower=0> NS; // number of subjects
    int<lower=0> NT; // number of trials
    int<lower=0, upper=1> respond_left[NS,NT]; // vector of LEFT responses (0=right, 1=left)
    int<lower=0, upper=1> reward_left[NS,NT]; // vector of rewards for left bandits (0=wrong answer, 1=correct answer)
    int<lower=0, upper=1> reward_right[NS,NT]; // vector of rewards for right bandits (0=wrong answer, 1=correct answer)
    int<lower=1, upper=4> bandit_left[NS,NT]; // which bandit (1 through 4) was presented on the left?
    int<lower=1, upper=4> bandit_right[NS,NT]; // which bandit (1 through 4) was presented on the right?
}

parameters {
	// Inverse temperature for softmax choice
	real itemp_mean; // group mean
	real<lower=0, upper=pi()/2> itemp_sd_unif; // group SD (pre-transform)
	vector[NS] itemp_raw; // participant values (pre-transform)

	// Logistic intercepts for learning rate
	real beta_int_mean; // group mean
	real<lower=0, upper=pi()/2> beta_int_sd_unif; // group standard deviation (pre-transform)
	vector[NS] beta_int_raw; // participant values (pre-transform)

	real beta_reward_mean; // group mean
	real<lower=0, upper=pi()/2> beta_reward_sd_unif; // group standard deviation (pre-transform)
	vector[NS] beta_reward_raw; // participant values (pre-transform)
}

transformed parameters {
	real<lower=0> itemp_sd;
	real<lower=0> beta_int_sd;
	real<lower=0> beta_reward_sd;
	vector[NS] itemp;
	vector[NS] beta_int;
	vector[NS] beta_reward;

	itemp_sd <- 0 + 5*tan(itemp_sd_unif);
	itemp <- itemp_mean + itemp_sd * itemp_raw;

	beta_int_sd <- 0 + 5*tan(beta_int_sd_unif);
	beta_int <- beta_int_mean + beta_int_sd * beta_int_raw;

	beta_reward_sd <- 0 + 5*tan(beta_reward_sd_unif);
	beta_reward <- beta_reward_mean + beta_reward_sd * beta_reward_raw;
}

model {
	vector[2] alpha; // learning rate holder variable
	vector[4] q; // set up a 4-item array to hold Q values for each bandit
	
	// PRIORS
	// Softmax choice inverse temperature
	itemp_mean ~ normal(0, 100); // prior for group mean
	itemp_raw ~ normal(0, 1); // prior for participant-level
	// Learning rate logistic intercept
	beta_int_mean ~ normal(0, 100); // prior for group mean
	beta_int_raw ~ normal(0, 1); // prior for participant-level

	// Learning rate logistic coefficient for whether option was rewarded
	beta_reward_mean ~ normal(0, 100); // prior for group mean
	beta_reward_raw ~ normal(0, 1); // prior for participant-level
	
	for (s in 1:NS) {

		q[1] <- 0; // starting point for bandit 1
		q[2] <- 0; // starting point for bandit 2
		q[3] <- 0; // starting point for bandit 3
		q[4] <- 0; // starting point for bandit 4

		for (t in 1:NT) {

			alpha[1] <- inv_logit(beta_int[s] + beta_reward[s] * reward_left[s,t]); // learning rate for left option
			alpha[2] <- inv_logit(beta_int[s] + beta_reward[s] * reward_right[s,t]); // learning rate for right option

			//print(q[bandit_left[s,t]] , q[bandit_right[s,t]]);
			respond_left[s,t] ~ bernoulli_logit(itemp[s] * (q[bandit_left[s,t]] - q[bandit_right[s,t]]));
			q[bandit_left[s,t]] <- q[bandit_left[s,t]] + alpha[1] * ( reward_left[s,t] - q[bandit_left[s,t]] ); // Update Q-value for left bandit
			q[bandit_right[s,t]] <- q[bandit_right[s,t]] + alpha[2] * ( reward_right[s,t] - q[bandit_right[s,t]] ); // Update Q-value for right bandit
		}
	}
}

generated quantities {
	vector[NS*NT] log_lik; 
	int ix;
	vector[2] alpha; // learning rate holder variable
	vector[4] q; // set up a 4-item array to hold Q values for each bandit
	real q_store[NS,NT,4]; // store Q values at the BEGINNING of each trial to use in analysis later
	real prediction_error_left[NS,NT]; // store prediction error from FEEDBACK for left item on each trial
	real prediction_error_right[NS,NT]; 
	
	for (s in 1:NS) {

		q[1] <- 0; // starting point for bandit 1
		q[2] <- 0; // starting point for bandit 2
		q[3] <- 0; // starting point for bandit 3
		q[4] <- 0; // starting point for bandit 4
		
		for (t in 1:NT) {
			ix <- (s-1)*NT + t; // index of log_lik vector
			
			q_store[s,t,1] <- q[1]; // store Q values at the BEGINNING of each trial for each bandit
			q_store[s,t,2] <- q[2];
			q_store[s,t,3] <- q[3];
			q_store[s,t,4] <- q[4];

			alpha[1] <- inv_logit(beta_int[s] + beta_reward[s] * reward_left[s,t]); // learning rate for left option
			alpha[2] <- inv_logit(beta_int[s] + beta_reward[s] * reward_right[s,t]); // learning rate for right option

			log_lik[ix] <- bernoulli_logit_log(respond_left[s,t], (itemp[s] * (q[bandit_left[s,t]] - q[bandit_right[s,t]])));
			
			prediction_error_left[s,t] <- reward_left[s,t] - q[bandit_left[s,t]];
			prediction_error_right[s,t] <- reward_right[s,t] - q[bandit_right[s,t]];

			q[bandit_left[s,t]] <- q[bandit_left[s,t]] + alpha[1] * ( reward_left[s,t] - q[bandit_left[s,t]] ); // Update Q-value for left bandit
			q[bandit_right[s,t]] <- q[bandit_right[s,t]] + alpha[2] * ( reward_right[s,t] - q[bandit_right[s,t]] ); // Update Q-value for right bandit
		}
	}
}