# SOLUTION.md

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

Output: `results.json` and `predictions.csv`.

Requires GPU or MPS for reasonable runtime (~2 min on MPS, ~5 min on CPU).

## Final approach

### Feature extraction (`aggregation.py`)

1. **Response-only tokens.** For each sample, the prompt is tokenized separately to determine its token length `L`. Hidden states are sliced to include only response tokens (positions `L` to end), isolating the model's generated answer from the context.

2. **Layer selection.** Hidden states extracted from transformer layers 12 and 13 — middle layers where semantic representations are most informative for hallucination detection.

3. **Max-pooling.** Each layer's response-token hidden states are max-pooled across the sequence dimension, producing two 896-dimensional vectors.

4. **Text features.** Three scalar features appended to the hidden-state vector:
   - Response length (characters)
   - Lexical overlap (Jaccard similarity between prompt and response word sets)
   - Response-to-prompt length ratio

Final feature dimension: **1795** (896 + 896 + 3).

### Classifier (`probe.py`)

- **L2 Logistic Regression** (`solver='lbfgs'`), without balanced class weights — train and test share the same 70/30 class prior, and balancing drops accuracy.
- **C selection**: grid search over `[0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]` via 5-fold OOF accuracy.
- **Threshold tuning**: 5-fold stratified out-of-fold (OOF) predictions collected for the entire training set. The probability threshold maximizing accuracy is selected over a grid of 199 values in `[0.01, 0.99]`.
- Final model retrained on all available data with the selected C and threshold.

### Splitting (`splitting.py`)

5-fold stratified cross-validation. Within each fold, 15% of the training portion is held out as a validation set. This provides stable averaged metrics in `results.json` and ensures the final probe (trained on all data in `solution.py`) sees every sample.

## Results

| Metric | Value |
|---|---|
| Avg test Accuracy (5-fold) | **75.03%** |
| Avg test AUROC (5-fold) | **79.28%** |
| Final OOF Accuracy (all 689 samples) | **77.21%** |
| Final OOF AUROC | **79.39%** |
| Baseline (majority class) | 70.10% |

## Experiments and failed attempts

1. **Last-token extraction + engineered features (norms, cosine similarities, random projections) + LogReg L1.** ~203 features. Accuracy ~70%, on par with baseline. Compact statistical features lost too much discriminative signal from the hidden states.

2. **PCA + Mahalanobis distance per layer + LogReg.** Class-conditional Gaussians with Ledoit-Wolf shrinkage over all 25 layers. Accuracy 67–71% depending on configuration. Unstable across folds, prone to overfitting the covariance estimates on the small training set.

3. **CatBoost on concatenated middle-late layers.** Heavy overfitting (84% train → 71% test) despite conservative hyperparameters (depth=4, l2_leaf_reg=10, subsample=0.8). Gradient boosting struggles with high-dimensional dense embeddings and few samples.

4. **Centroid classifier (cosine similarity to class means).** Accuracy ~66%. Too simplistic for this feature space.

5. **class_weight='balanced' in LogReg.** Drops accuracy by ~3% because the test set preserves the same 70/30 prior as training data. Balancing artificially inflates the minority class predictions.
