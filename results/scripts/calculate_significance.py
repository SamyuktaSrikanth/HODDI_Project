import numpy as np
from scipy import stats

def calculate_significance():
    print("======================================================")
    print("   STATISTICAL SIGNIFICANCE TEST (PAIRED T-TEST)      ")
    print("======================================================\n")
    print("INSTRUCTIONS:")
    print("Paste your 7-seed results from Exp A and Exp B below.")
    print("Make sure the arrays are in the exact same seed order!\n")

    # --- REPLACE THESE ARRAYS WITH YOUR ACTUAL 7-SEED OUTPUTS ---
    # Example dummy data (replace with actual Kaggle outputs)
    
    # 1. PR-AUC
    prauc_A = np.array([0.8950, 0.8971, 0.8965, 0.8992, 0.8960, 0.8970, 0.8980])
    prauc_B = np.array([0.9660, 0.9680, 0.9675, 0.9690, 0.9665, 0.9672, 0.9681])
    
    # 2. AUC
    auc_A = np.array([0.9390, 0.9410, 0.9395, 0.9405, 0.9398, 0.9402, 0.9399])
    auc_B = np.array([0.9680, 0.9692, 0.9685, 0.9695, 0.9682, 0.9688, 0.9690])
    
    # 3. F1-Score
    f1_A = np.array([0.9140, 0.9160, 0.9150, 0.9170, 0.9145, 0.9155, 0.9165])
    f1_B = np.array([0.9270, 0.9290, 0.9280, 0.9300, 0.9275, 0.9285, 0.9295])
    
    # 4. Precision
    prec_A = np.array([0.8960, 0.8980, 0.8970, 0.8990, 0.8965, 0.8975, 0.8985])
    prec_B = np.array([0.9170, 0.9190, 0.9180, 0.9200, 0.9175, 0.9185, 0.9195])
    
    # 5. Recall
    rec_A = np.array([0.9330, 0.9350, 0.9340, 0.9360, 0.9335, 0.9345, 0.9355])
    rec_B = np.array([0.9370, 0.9390, 0.9380, 0.9400, 0.9375, 0.9385, 0.9395])

    metrics = {
        "PR-AUC": (prauc_A, prauc_B),
        "AUC": (auc_A, auc_B),
        "F1-Score": (f1_A, f1_B),
        "Precision": (prec_A, prec_B),
        "Recall": (rec_A, rec_B)
    }

    for name, (A, B) in metrics.items():
        # Perform Paired T-Test
        t_stat, p_val = stats.ttest_rel(B, A)
        
        # Calculate Effect Size (Cohen's d for paired samples)
        mean_diff = np.mean(B - A)
        std_diff = np.std(B - A, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Determine significance (p < 0.05)
        sig_marker = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT SIGNIFICANT"
        
        print(f"[{name}]")
        print(f"  Exp A Mean: {np.mean(A):.4f} | Exp B Mean: {np.mean(B):.4f}")
        print(f"  Absolute Jump : +{mean_diff*100:.2f}%")
        print(f"  P-Value       : {p_val:.5f} ({sig_marker})")
        print(f"  Effect Size   : {cohens_d:.2f} (Cohen's d)\n")

if __name__ == "__main__":
    calculate_significance()
