# graphs.py

# Graphs script to visualize the results of the Monte Carlo simulations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bias():
    # Academic style
    sns.set_theme(style="whitegrid")

    df = pd.read_csv('monte_carlo_results.csv')

    plt.figure(figsize=(10, 6))

    N_target = 200
    if N_target not in df['N'].unique():
        N_target = df['N'].max()

    subset = df[df['N'] == N_target]

        # FE (Bias that goes down with T)
    plt.plot(subset['T'], subset['FE_bias'], 
            marker='o', 
            linestyle='-',
            linewidth=2.5,
            color='#55A868', markersize=8,
            label=f'FE Bias')
        
        # Pooled OLS (Bias that does not go down with T)
    plt.plot(subset['T'], subset['Pool_bias'], 
            marker='x', 
            linestyle='-',
            linewidth=2.5,
            color='#C44E52', markersize=8,
            label=f'Pooled OLS Bias')
    
        # IV (Bias that goes down with T)
    plt.plot(subset['T'], subset['IV_bias'], 
            marker='s', 
            linestyle='-',
            linewidth=2.5,
            color='#4C72B0', markersize=8,
            label=f'IV Bias')

    # 0 bias reference line
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.9)

    # Graph formatting
    plt.title('Nickell Bias Convergence as $T \\to \\infty$', fontsize=14, fontweight='bold')
    plt.xlabel('Time Periods ($T$)', fontsize=12)
    plt.ylabel('Mean Bias ($\\hat{\\gamma} - \\gamma$)', fontsize=12)
    
    plt.legend(title='Estimators', fontsize=11, loc='lower center', framealpha=0.95)

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig('nickell_bias.pdf', dpi=300, format='pdf')
    plt.savefig('nickell_bias.png', dpi=300)
    print("Graph saved as 'nickell_bias.pdf' and 'nickell_bias.png'.")

    plt.show()

if __name__ == "__main__":
    plot_bias()