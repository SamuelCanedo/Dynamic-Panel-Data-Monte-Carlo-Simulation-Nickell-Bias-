# dgp.py
# This script is the data generating process that will be use to determine 
# the estimators and will further on be used for the Monte Carlo Simulations

import numpy as np 
import pandas as pd 

def generate_dynamic_panel(N, T, gamma, sigma_alpha=1.0, sigma_u=1.0, burn_in=50):
    ## simulation of a Dynamic DGP AR(1): y_it = gama * y_it-1 + alpha_i + u_it
    # Fixed individual effects alpha_i ~ N(0, sigma_alpha^2)
    alpha = np.random.normal(0, sigma_alpha, size=(N, 1))

    total_T = T + burn_in
    y = np.zeros((N, total_T))
    u = np.random.normal(0, sigma_u, size=(N, total_T))

    # Initial condition for y_it-1, we can set it to zero or draw from the stationary distribution
    y[:, 0] = alpha.squeeze() / (1 - gamma) + np.random.normal(0, sigma_u / np.sqrt(1 - gamma**2), size=N)

    # Generate the time series for each individual
    for t in range(1, total_T):
        y[:, t] = gamma * y[:, t - 1] + alpha.squeeze() + u[:, t]

    # Discard the burn-in period
    y = y[:, burn_in:]

    # Transform the data into a long format DataFrame
    id_indices = np.repeat(np.arange(N), T)
    time_indices = np.tile(np.arange(T), N)
    y_long = y.flatten()

    df = pd.DataFrame({
        'id': id_indices,
        'time': time_indices,
        'y': y_long,
    })

    df['y_lag'] = df.groupby('id')['y'].shift(1)

    return df.dropna().set_index(['id', 'time'])