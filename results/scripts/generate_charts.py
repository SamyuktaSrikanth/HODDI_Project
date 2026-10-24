import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_comparison():
    labels = ['PRECISION', 'RECALL', 'F1', 'AUC', 'PRAUC']
    
    # HODDI Paper (Table 7) - NA for recall is set to 0 just for shape, we'll hide it
    paper_means = [0.906, 0.0, 0.933, 0.957, 0.939]
    paper_errs = [0.002, 0.0, 0.001, 0.003, 0.008]
    
    # Our Final Reproduction of A (4 Seeds)
    our_a_means = [0.8973, 0.9342, 0.9153, 0.9399, 0.8968]
    our_a_errs = [0.0053, 0.0170, 0.0062, 0.0018, 0.0057]

    # Our Final Exp B (HyperAttDDI)
    our_b_means = [0.9183, 0.9381, 0.9280, 0.9686, 0.9671]
    our_b_errs = [0.0144, 0.0120, 0.0069, 0.0022, 0.0034]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, paper_means, width, yerr=paper_errs, label='Paper Baseline (HGNN-SA)', capsize=5, color='#4c72b0')
    rects2 = ax.bar(x, our_a_means, width, yerr=our_a_errs, label='Our Exp A (Baseline)', capsize=5, color='#dd8452')
    rects3 = ax.bar(x + width, our_b_means, width, yerr=our_b_errs, label='Our Exp B (HyperAttDDI)', capsize=5, color='#55a868')
    
    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Baseline vs. HyperAttDDI Upgrade', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim([0.85, 1.0])
    
    # Add a horizontal grid for easier reading
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('final_results_chart.png', dpi=300)
    print("Saved final_results_chart.png")

if __name__ == '__main__':
    plot_metrics_comparison()
