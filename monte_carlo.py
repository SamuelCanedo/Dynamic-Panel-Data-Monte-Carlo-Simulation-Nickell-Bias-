# monte_carlo.py

# Monte Carlo sumulations script to compare the performance of the FE and Pooled OLS estimators

import numpy as np
import pandas as pd
from dgp import generate_dynamic_panel
from estimators import estimate_models

def run_monte_carlo(N_sizes, T_sizes, true_gamma, replications=100):
    # Executes the iterative Monte Carlo simulations for different N and T sizes
    np.random.seed(42)  # For reproducibility
    results = []

    print("Starting Monte Carlo simulations...")
    for N in N_sizes:
        for T in T_sizes: 
            fe_biases = []
            pool_biases = []
            iv_biases = []
            for i in range(replications):
                df = generate_dynamic_panel(N, T, true_gamma)
                gamma_fe, gamma_pool, gamma_iv = estimate_models(df)

                fe_biases.append(gamma_fe - true_gamma)
                pool_biases.append(gamma_pool - true_gamma)
                iv_biases.append(gamma_iv - true_gamma)

            # Calculate the mean bias for each estimator
            mean_fe_bias = np.mean(fe_biases)
            mean_pool_bias = np.mean(pool_biases)
            mean_iv_bias = np.mean(iv_biases)

            results.append({
                'N': N,
                'T': T,
                'FE_bias': mean_fe_bias,
                'Pool_bias': mean_pool_bias,
                'IV_bias': mean_iv_bias
            })
            print(f'Simulations completed N={N}, T={T}: FE bias={mean_fe_bias:.4f}, Pool bias={mean_pool_bias:.4f}, IV bias={mean_iv_bias:.4f}')

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Define the parameters for the Monte Carlo simulations
    N_sizes = [50, 100, 200]  # Different numbers of individuals
    T_sizes = [5, 10, 20]     # Different time periods
    true_gamma = 0.5          # True value of gamma in the DGP

    # Run the Monte Carlo simulations
    df_results = run_monte_carlo(N_sizes, T_sizes, true_gamma, replications=200)
    df_results.to_csv('monte_carlo_results.csv', index=False)
    print("Monte Carlo simulations completed. Results saved to 'monte_carlo_results.csv'.")