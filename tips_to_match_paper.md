# Tips to Match the Paper

- Use the same KDD99 variant (full vs 10%); results vary significantly by version.
- Remove duplicates in KDD99; the paper reports de-duplication.
- Ensure label cleaning for KDD99 (strip trailing dots in labels).
- One-hot encoding expands feature count; the CNN bottleneck size (18/22) is the reduction target.
- Min-Max scaling must be fit only on train and applied to val/test.
- Class imbalance handling (over/under sampling) changes metrics notably; compare settings.
- Fix seeds across Python, NumPy, TensorFlow, and scikit-learn for reproducibility.
- Early stopping is enabled; epoch counts may be lower than 100 if validation loss plateaus.
