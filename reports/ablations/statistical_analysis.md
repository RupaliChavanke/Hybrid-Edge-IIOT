# Statistical Significance & Confidence Bounds Report (Phase 10)

**Date:** 2026-10-23 23:52:49  
**Sample Size:** $N = 23,670$ records ($K=33$ zero-leakage protocol features)  
**Confidence Level:** 95% Non-Parametric Bootstrap (300 resamples)  

## Bootstrap 95% Confidence Intervals ($K=33$ Optimized Models)

| Model Candidate | Accuracy Point | 95% Accuracy CI | Macro F1 Point | 95% Macro-F1 CI |
|---|---|---|---|---|
| **CatBoost** | 99.18% | [99.07%, 99.29%] | 97.66% | [97.24%, 98.10%] |
| **XGBoost** | 99.18% | [99.06%, 99.29%] | 97.70% | [97.28%, 98.14%] |
| **LightGBM** | 99.19% | [99.08%, 99.31%] | 97.71% | [97.30%, 98.15%] |
| **Hybrid Ensemble** | 99.20% | [99.09%, 99.32%] | 97.72% | [97.31%, 98.16%] |

## Paired Edwards-Corrected McNemar Significance Tests

| Comparison Pair | Only A Correct ($b$) | Only B Correct ($c$) | McNemar $\chi^2$ | $p$-Value | Significant ($\alpha=0.05$)? |
|---|---|---|---|---|---|
| **Hybrid Ensemble vs. CatBoost** | 6 | 2 | 1.1250 | 2.8884e-01 | No (p >= 0.05) |
| **Hybrid Ensemble vs. XGBoost** | 7 | 2 | 1.7778 | 1.8242e-01 | No (p >= 0.05) |
| **Hybrid Ensemble vs. LightGBM** | 2 | 0 | 0.5000 | 4.7950e-01 | No (p >= 0.05) |
| **LightGBM vs. CatBoost** | 5 | 3 | 0.1250 | 7.2367e-01 | No (p >= 0.05) |
| **XGBoost vs. CatBoost** | 7 | 8 | 0.0000 | 1.0000e+00 | No (p >= 0.05) |

## Scientific Interpretation
1. **Highly Significant Gain from $K=25 \to K=33$ Feature Expansion:**
   - Comparing the $K=33$ Hybrid Ensemble ($99.21\%$ accuracy, $97.73\%$ Macro-F1) against the $K=25$ Baseline Ensemble ($98.94\%$ accuracy, $95.53\%$ Macro-F1) yields $n_{10}=132$ vs. $n_{01}=70$ discordant errors ($\chi^2_{\text{McNemar}} = 18.42, p = 1.77 \times 10^{-5}$) and a paired Wilcoxon signed-rank $p = 8.41 \times 10^{-32}$ across 500 bootstrap replicates.
2. **Convergence to the Empirical Bayes Ceiling Within Fixed $K$:**
   - Within the $K=33$ feature space, the empirical Bayes accuracy ceiling is $99.4254\%$ ($23,534 / 23,670$ rows) due to 348 identical feature rows carrying conflicting labels (`SQL_injection`/`Uploading`/`Password`/`XSS`). All three GBDT learners and the ensemble operate within $0.22\%$ of this theoretical ceiling, so their pairwise McNemar differences are statistically indistinguishable ($p \ge 0.099$).
3. **Defensible Conclusion:**
   - We do not claim hard-label accuracy superiority of the ensemble over LightGBM within a fixed feature space. Rather, the SLSQP + Temperature-Scaled ensemble reduces Expected Calibration Error (ECE) by $10\times$ relative to CatBoost ($0.700\% \to 0.073\%$) and achieves the lowest Brier score ($0.01232$ with Isotonic, $0.01245$ with Temperature scaling).
