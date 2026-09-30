# experiments — R-ECO training & evaluation pipeline

Run from this directory (on Colab):

```bash
python model.py          # all 7 stages in order
python base_N_oof.py     # or a single stage
python ig/run_all.py     # IG analysis (after the pipeline)
```

- Every stage loops over sites `[1, 2, 4, 5, 6, 7, 8]`. A site is skipped, with a warning, if any of its input files is missing.
- Outputs go to `SAVE_PATH` (`…/timexer/data/results`) as `site{N}_*`.
- Each stage also writes its own log file (`local_cell*_log.txt`) under `LOG_DIR`.

---

## Shared module — `common.py`

Every stage imports from `common.py`.

| Group | Contents |
|---|---|
| Paths / sites | `ORIGIN_PATH`, `ESVD_PATH`, `SAVE_PATH`, `TARGET_SITES`, `OPERATION_HOURS`, `SITE7_EXCLUDE`, `site_paths()` |
| `Config` | lookback 672, horizon 1, residual lookback R = 96, VMD K = 4, d_model 16, 1 head, 2 fusion layers |
| Models | `BaseModel`, `ResidualModel` (below) |
| Datasets | `PVDataset` (Base), `ResidualDataset` (adds OOF prediction / residual / residual window) |
| Training | `train_base_model`, `train_residual_model`. Both use Adam + MSE with early stopping; validation loss counts only daytime samples (target > 0.0001). |
| Testing | Residual-window warm-up / push, real-time VMD + residual inference |
| Scoring | `filter_eval_window` (operation hours, Site 7 exclusion), `score` (NRMSE %, NMAE %, R²) |

### Models

**BaseModel (ExoTFT)**
- PV input `(672, 6)` = raw PV + 5 ESVD channels, split into patches (length 16, stride 8), plus one global token.
- Weather (current step) and calendar (target step) are embedded one token per variable and weighted by a Variable Selection Network.
- Each fusion layer runs self-attention over the PV tokens, then cross-attention from the global token to the exogenous tokens, then a GRN feed-forward.
- A linear head outputs ŷ_base.

**ResidualModel**
- Input: a residual window `(96, 5)` = raw residual + VMD modes 1–4, patched in the same way.
- Its exogenous tokens are the weather, calendar and PV encoders reused from a **frozen** BaseModel, plus a token for ŷ_base.
- Output: ê, the predicted Base error.

### Scaling

- A `StandardScaler` is fit on daytime `Power (MW)`, on M1–M12 for every stage except ExoTFT-FB, which uses M1–M22.
- PV and target are standardised. The ESVD channels are divided by the same scale.
- Predictions are inverse-transformed to MW before being saved.

---

## Pipeline

M = 1/24 of the samples (≈ 1 month). Test = M23–M24.

### 1. `base_N_oof.py` — Base training + rolling OOF residuals
1. Train the initial Base on M1–M10, validating on M11–M12 → `base_trained.pt`.
2. For each 2-month block M11–12, M13–14, …, M21–22:
   - (a) predict the block with the current model (out-of-sample) and record the residuals;
   - (b) fine-tune on M1 up to the block (lr 1e-4), using the block itself for early stopping.
3. The model after the last block → `base_rolling_m22.pt`. M1–M10 are filled with the initial Base's in-sample predictions.
4. Decompose each 96-step residual window with VMD (K = 4) → `esvd_windows.pt`.

Outputs: `base_trained.pt`, `base_rolling_m22.pt`, `oof_preds.pt`, `residuals.pt`, `esvd_windows.pt`

### 2. `res_model_5_fold.py` — residual model, 5-fold OOF CV
- Frozen encoder: `base_rolling_m22.pt`.
- Inputs: the OOF ŷ_base and OOF residuals from stage 1. The residual model therefore never sees in-sample Base predictions.
- Fold k validates on M(11+2k)–M(12+2k): M13–14, …, M21–22.
- Training data is M1 up to the fold, plus from 672 steps after the fold to M22. The 672-step gap keeps input windows from overlapping the fold.

Outputs: `res_fold{k}.pt`, `oof_val_fold{k}.csv` (`Time, True_Target, Base_Pred, Res_Pred`)

### 3. `init_alpha.py` — initial α
- Concatenate the 5 fold CSVs and keep only the scoring window.
- Grid-search α ∈ [0, 2] (step 0.01) for `Base_Pred + α · Res_Pred`, keeping the α with the lowest NRMSE.

Output: `best_alpha_value.csv` (`Best_Alpha`, `OOF_NRMSE`, `OOF_NMAE`, `OOF_R2`)

### 4. `create_rf.py` — ExoTFT-RF + fixed α
- Retrain the residual model on M13–M22 (90/10 split), with `base_trained.pt` as the frozen encoder → `res_final.pt`.
- Sequential test on M23–M24, one step at a time:
  - Warm-up: re-predict the 96 steps before the test with ExoTFT-RF to fill the residual window.
  - At each step: VMD on the window → ê → `ŷ = ŷ_base + best_alpha · ê` → push the true Base error into the window.

Output: `final_test_results.csv` (`Time, True_Target, Rolling_Base_Pred, Final_Ensemble_Pred`). Not scored.

### 5. `create_fb.py` — ExoTFT-FB baseline
- Train a new Base on all of M1–M22 (90/10 split), with a scaler fit on M1–M22.
- Test on M23–M24. `True_Target` is read directly from the original CSV by timestamp.

Outputs: `base_fullbatch_m22.pt`, `base_fullbatch_test_results.csv` (`Time, True_Target, Base_Pred`)

### 6. `fb_results.py` — ExoTFT-FB + scoring
- Re-runs `create_fb` (retrains and overwrites its outputs), then scores ExoTFT-FB.
- Scoring uses operation hours, drops NaN and applies `True_Target > 0.0001`.

### 7. `crc_create_N_results.py` — R-ECO (adaptive α)
- Retrain the residual model on M13–M22 in the same way as stage 4, as a separate run → `res_final_adaptive.pt`.
- Sequential test like stage 4, with α adapted online:
  - α starts at `best_alpha`.
  - Every 96 steps, run a paired t-test on |Base error| vs |R-ECO error| over those 96 steps:

    | Test result | Action |
    |---|---|
    | p < 0.05 and R-ECO worse | α × 0.5 |
    | p < 0.05 and R-ECO better | α × 1.2 |
    | otherwise | keep α |

  - α is clipped to [0, 2].
- Score ExoTFT-RF and R-ECO (operation hours, no threshold) and print the NRMSE gain.

Output: `res_final_adaptive.csv` (`Time, True_Target, Rolling_Base_Pred, Final_Ensemble_Pred, Alpha_Used`)

---

## Integrated Gradients — `ig/`

Run after the pipeline: `python ig/run_all.py` runs `fb.py`, then `crc.py`.

- Both use Captum `IntegratedGradients` on the M23–M24 test windows, with a zero baseline and no retraining.
- Importance = |attribution| summed over time, averaged over samples, then shown as a share (%) of the total.

| Script | Explained model | Inputs attributed | Output |
|---|---|---|---|
| `fb.py` | ExoTFT-FB (`base_fullbatch_m22.pt`, M1–M22 scaler) | PV raw + ESVD 1–5, 5 weather, 6 calendar | `ig_feature_importance_fullbatch_base.csv` |
| `crc.py` | R-ECO output `ŷ_base + α · ê`. ExoTFT-RF (`base_rolling_m22.pt`) + residual model on the `base_trained.pt` encoder. α fixed to `Best_Alpha`; M1–M12 scaler. | as above + residual raw error + IMF 1–4 (from the saved OOF residuals) | `ig_feature_importance_adaptive_reco.csv` |

---

## Outputs used downstream

| File | Produced by | Used by |
|---|---|---|
| `base_trained.pt` | 1 | 4, 7, `ig/crc` |
| `base_rolling_m22.pt` | 1 | 2, 4, 7, `ig/crc` |
| `oof_preds.pt`, `residuals.pt`, `esvd_windows.pt` | 1 | 2, 4, 7 (`residuals` / `esvd_windows` also `ig/crc`) |
| `oof_val_fold{k}.csv` | 2 | 3 |
| `best_alpha_value.csv` | 3 | 4, 7, `ig/crc` |
| `base_fullbatch_m22.pt` | 5 / 6 | `ig/fb` |
| `base_fullbatch_test_results.csv` | 5 / 6 | figures (ExoTFT-FB) |
| `res_final_adaptive.pt` | 7 | `ig/crc` |
| `res_final_adaptive.csv` | 7 | figures (ExoTFT-RF / ExoTFT-CRC) |
