# R-ECO — Residual Error Correction for PV Power Forecasting

R-ECO is a two-stage forecasting framework for 15-minute-ahead photovoltaic (PV) power forecasting.

1. **Base model (ExoTFT)**: a TFT-style network that fuses a 7-day PV history (decomposed with ESVD) with weather and calendar features to produce a one-step-ahead forecast ŷ_base.
2. **Residual model**: a second network that predicts the Base model's next error ê from its recent residual history, which is decomposed with VMD. The final forecast is

   ```
   ŷ = ŷ_base + α · ê
   ```

   The weight α starts from a value tuned on out-of-fold (OOF) data. During the test period it is **adapted online** with a paired t-test on recent errors.

The repository contains the preprocessing, the experiment pipeline, the Integrated Gradients (IG) analysis and the paper figures.

| Name in the paper | Meaning |
|---|---|
| **ExoTFT-FB** | Full-batch Base: one Base model trained on M1–M22 |
| **ExoTFT-RF** | Rolling Base: Base model fine-tuned every 2 months up to M22 |
| **ExoTFT-CRC** (R-ECO) | ExoTFT-RF + residual model × adaptive α |

---

## Repository layout

```
R-ECO/
├── preprocess/            raw xlsx → site CSVs, ESVD input windows
│   ├── create_X.py
│   └── create_esvd.py
├── experiments/           model training / testing pipeline  (see experiments/README.md)
│   ├── common.py          shared config, models, datasets, training & scoring utils
│   ├── model.py           runs the whole pipeline in order
│   ├── base_N_oof.py  res_model_5_fold.py  init_alpha.py
│   ├── create_rf.py   create_fb.py  fb_results.py  crc_create_N_results.py
│   └── ig/                Integrated Gradients analysis
│       ├── fb.py  crc.py  run_all.py
├── figures/
│   ├── code/              one script per figure + config.py / loader.py
│   └── fig/               generated figures
└── dataset/               local copies of the data / results used by figures
    ├── origin/            site_{N}.csv
    ├── results/           test-period prediction CSVs, aggregated metric table
    └── ig/                aggregated IG importance tables
```

---

## Data

- **Source**: *Solar and wind power data from the Chinese State Grid Renewable Energy Generation Forecasting Competition* ([figshare](https://figshare.com/articles/dataset/Solar_and_wind_power_data_from_the_Chinese_State_Grid_Renewable_Energy_Generation_Forecasting_Competition/17304221/4)). Put the `.xlsx` files in `dataset/solar_stations/`.
- **Sites**: PV stations 1, 2, 4, 5, 6, 7 and 8. Site 3 is excluded.
- **Period / resolution**: 2019-01-01 – 2020-12-31, 15-minute steps.
- **Variables**:
  - Target: `Power (MW)`
  - Weather: `GHI`, `DNI`, `TSI`, `Temperature`, `Atmospheric pressure`
  - Calendar (added in preprocessing): `hour`, `month` and their sin/cos encodings
- **Time split**: all samples are divided into 24 equal "months" (M1–M24).

  | Months | Use |
  |---|---|
  | M1–M12 | Fit the power scaler (all models except ExoTFT-FB) |
  | M1–M22 | Training, rolling fine-tuning and OOF cross-validation |
  | M23–M24 (≈ Nov–Dec 2020) | Test |

- **Scoring**:
  - Metrics: NRMSE, NMAE (% of site capacity) and R².
  - Only each site's operation hours are scored.
  - Site 7 `2020-12-14 08:00 – 2020-12-31 23:45` is always removed (data anomaly).

---

## Preprocessing (`preprocess/`)

| Script | Input → Output | What it does |
|---|---|---|
| `create_X.py` | `solar_stations/*.xlsx` → `origin/*.csv` | Renames the weather columns. Adds `hour`/`month` and their sin/cos encodings, site ID and nominal/maximum capacity. The experiments expect the output as `site_{N}.csv`. |
| `create_esvd.py` | `solar_stations/*.xlsx` → `esvd_features/site_{N}_{X_esvd,Y,Y_time}.npy` | Builds one 672-step (7-day) input window per target step using **ESVD** (below), in parallel. Uses raw MW values (no scaling) and keeps night-time zeros. |

**ESVD** (per 672-step window):

1. Run EMD on the PV power, with small jitter added.
2. Split the IMFs by sample entropy (threshold = mean SE) into a high-complexity part and a low-complexity part.
3. Decompose the high-complexity part with VMD (K = 4).
4. Each window becomes 6 channels: `[raw PV, low-complexity reconstruction, VMD mode 1–4]`.

---

## Experiments (`experiments/`)

Run everything with `python experiments/model.py`. Each stage reads the previous stage's outputs. Details, file lists and outputs are in [`experiments/README.md`](experiments/README.md).

| # | Stage | Role |
|---|---|---|
| 1 | `base_N_oof` | Train the Base (M1–M10). Roll it through M11–M22 in 2-month blocks: predict out-of-sample, then fine-tune. Collect OOF residuals and their VMD windows. |
| 2 | `res_model_5_fold` | 5-fold OOF cross-validation of the residual model on M13–M22 |
| 3 | `init_alpha` | Grid-search α ∈ [0, 2] on the OOF folds (minimum NRMSE) |
| 4 | `create_rf` | ExoTFT-RF vs ExoTFT-RF + fixed α · ê on M23–M24 |
| 5 | `create_fb` | ExoTFT-FB: train one Base on M1–M22 and test on M23–M24 |
| 6 | `fb_results` | Same as `create_fb`, plus scoring |
| 7 | `crc_create_N_results` | **R-ECO**: ExoTFT-RF + adaptive α · ê on M23–M24, plus scoring |

## Explainability — Integrated Gradients (`experiments/ig/`)

`python experiments/ig/run_all.py` runs both IG scripts on the M23–M24 test windows, using Captum with an all-zero baseline.

- **`fb.py` (ExoTFT-FB)**: attributes the full-batch Base (`base_fullbatch_m22.pt`) to its inputs: PV/ESVD channels, weather and calendar.
- **`crc.py` (ExoTFT-CRC)**: attributes the full R-ECO output `ŷ_base + α · ê`, with α fixed to `Best_Alpha`. The Base is the rolling Base and the encoder is `base_trained.pt`. It adds the residual-history inputs (raw error and IMF 1–4).

Importance per feature = |attribution| summed over time, averaged over samples, then expressed as a share (%) of the total.

## Figures (`figures/code/`)

Each `fig*.py` script produces one figure in `figures/fig/`. The scripts use `dataset/{origin,results,ig}`. Style, paths and the model registry are in `config.py`. Result merging is in `loader.py`.

| Script | Figure |
|---|---|
| `fig3_a`, `fig3_b` | Site overview: mean hourly power, distribution of generation hours |
| `fig4` | NMAE / NRMSE / R² per site and model |
| `fig_alpha_assoc`, `alpha_association`, `table_alpha` | Adaptive α behaviour and what drives it |
| `fig6_a`, `fig6_b` | One-day forecast profiles, hourly residual bands |
| `fig7` | Relative-error distributions (ridge plot) |
| `fig8_b`, `fig8_b2`, `fig8_c` | Date × hour residual heatmaps |
| `fig9`, `fig9_a`, `fig9_b` | IG importance: CRC − FB difference, CRC ratios, per-encoder differences |

---

## Environment

The experiments were run on **Google Colab** (GPU) with Drive mounted at `/content/gdrive`. All experiment paths are defined at the top of `experiments/common.py`:

| Colab path (`…/reforecast/timexer/data/`) | Local path used by the figures |
|---|---|
| `origin/` | `dataset/origin/` |
| `esvd_features/` | not stored locally |
| `results/` | `dataset/results/`, `dataset/ig/` |

Main packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`, `vmdpy`, `EMD-signal` (PyEMD), `antropy`, `captum`, `matplotlib`, `seaborn`.
