Reading kernel: mitchgansemer/gr-features-outlier-detection-rogii-wellbore
Cells: 34 (18 code, 16 markdown) | 62731 chars

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃     ROGII Wellbore — GR Feature Engineering & Per-Well Outlier Analysis      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Best LB: 8.905 ft RMSE                                                          

This notebook covers two interlocking themes: (1) how gamma-ray signal          
processing drives the feature set, and (2) where the model fails and why. The   
same full inference pipeline runs at the end to reproduce the submission.       

Feature generation and model training run offline (~3 hours on a laptop); those 
sections show the real implementation but skip execution here.                  

                          What each change contributed                          

                                                                                
  Round      Key change                                    OOF RMSE   LB        
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  Baseline   Predict last_anchor_tvt unchanged             15.91 ft   —         
  R2         Tree model, 52 features — but absolute TVT    19.50 ft   —         
             target                                                             
  R3         Drift target + GR xcorr + formation KNN       14.99 ft   —         
  R4         + Viterbi beam / particle filter sequence     13.96 ft   —         
             estimators                                                         
  R5         Ensemble (LGB + XGB + CatBoost)               13.90 ft   —         
  R6         163 features: multi-scale NCC, formation      10.01 ft   9.410 ft  
             plane-fit KNN, beam/PF variants; Optuna;                           
             NNLS blend                                                         
  R7         CatBoost retuned (depth 5), NNLS blend XGB    10.05 ft   9.398 ft  
             55.9% + CB 44.1%                                                   
  R10        +21 v4 features: estimator divergence,        9.91 ft    —         
             short-window slopes, DWT GR                                        
  R11        + HistGradientBoosting, NNLS XGB+CB+HGB       9.85 ft    8.905 ft  
             blend                                                              
                                                                                

The biggest jump (R3) came from reframing the target. After that it's           
incremental: better GR-derived signals, more orthogonal estimators, and ensemble
diversity. Sections 2–4 below cover the three GR-centric ideas that drive most  
of the gain.                                                                    

────────────────────────────────────────────────────────────────────────────────
Setup: the rogii-wellbore-models dataset must be attached (pre-trained fold     
models, KNN structures, and pre-computed ground-truth arrays).                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 %matplotlib inline                                                             
 import sys, os, json, time                                                     
 import numpy as np                                                             
 import pandas as pd                                                            
 import joblib                                                                  
 import pywt                                                                    
 import matplotlib.pyplot as plt                                                
 from pathlib import Path                                                       
 from scipy.signal import savgol_filter                                         
 from scipy.optimize import minimize                                            
 from sklearn.model_selection import GroupKFold                                 
 from sklearn.ensemble import HistGradientBoostingRegressor                     
 from tqdm import tqdm                                                          
                                                                                
 import xgboost as xgb                                                          
 from catboost import CatBoostRegressor                                         
                                                                                
 # ── On-Kaggle vs local detection ──────────────────────────────────────────── 
 _ON_KAGGLE = os.path.exists('/kaggle')                                         
                                                                                
 # ── Locate MODEL_DIR (contains feature_cols.json, fold models, KNN artifacts) 
 MODEL_DIR = Path('models')                                                     
 if _ON_KAGGLE:                                                                 
     for _root, _dirs, _files in os.walk('/kaggle/input'):                      
         if 'feature_cols.json' in _files:                                      
             MODEL_DIR = Path(_root)                                            
             break                                                              
 print(f'ON_KAGGLE={_ON_KAGGLE}  MODEL_DIR={MODEL_DIR}')                        
                                                                                
 # ── Add utils.py to path                                                      
 ───────────────────────────────────────────────────────                        
 sys.path.insert(0, str(MODEL_DIR) if _ON_KAGGLE else '.')                      
                                                                                
 import utils                                                                   
 from utils import (                                                            
     load_typewell, load_horizontal, get_train_well_ids, get_test_well_ids,     
     rmse, set_style, save_fig,                                                 
     impute_gr_with_typewell, build_typewell_interp,                            
     gr_xcorr_batch, multi_scale_ncc, multi_scale_ncc_anchor,                   
     FormationKNN, FormationPlaneKNN, RowKNN, DenseANCCImputer,                 
     visible_gr_shift_fit, viterbi_tvt, particle_filter_tvt,                    
 particle_filter_ancc,                                                          
     learn_z_beta, BEAM_VARIANTS, PF_VARIANTS,                                  
     compute_trajectory_kinematics, estimate_apparent_dip,                      
     FORMATION_COLS,                                                            
 )                                                                              
 set_style()                                                                    
                                                                                
 # On Kaggle: update data dirs to competition input path                        
 if _ON_KAGGLE:                                                                 
     utils.DATA_DIR  =                                                          
 Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction')           
     utils.TRAIN_DIR = utils.DATA_DIR / 'train'                                 
     utils.TEST_DIR  = utils.DATA_DIR / 'test'                                  
                                                                                
 # ── Inference constants (must match 02_features.ipynb exactly) ────────────── 
 XCORR_WINDOW   = 100                                                           
 XCORR_RADIUS   = 50.0                                                          
 XCORR_STEP     = 1.0                                                           
 XCORR_STRIDE   = 10                                                            
 ANCHOR_TAIL    = 200                                                           
 NCC_RADIUS     = 150.0                                                         
 NCC_HALFWIDTHS = (8, 15, 25)                                                   
 NCC_TEMP       = 10.0                                                          
 ROLL_SHORT     = 25                                                            
 ROLL_LONG      = 100                                                           
 KNN_K          = 15                                                            
 BEAM_RADIUS    = 80.0                                                          
 PF_PARTICLES   = 500                                                           
 TW_OFFSETS     = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80],      
 dtype=np.float32)                                                              
 SG_W    = 17   # Savitzky-Golay window (R8: domain prior from reference        
 notebook)                                                                      
 SG_P    = 3                                                                    
 N_FOLDS = 5                                                                    
 SEED    = 42                                                                   
                                                                                
 # ── Load serialized KNN structures (needed for demos and inference)           
 ────────────                                                                   
 t0 = time.time()                                                               
 formation_plane_knn = joblib.load(MODEL_DIR / 'formation_plane_knn.joblib')    
 formation_knn       = joblib.load(MODEL_DIR / 'formation_knn.joblib')          
 row_knn             = joblib.load(MODEL_DIR / 'row_knn.joblib')                
 dense_imputer       = joblib.load(MODEL_DIR / 'dense_imputer.joblib')          
 print(f'KNN artifacts loaded in {time.time()-t0:.1f}s')                        
                                                                                
 # ── Example well for all demos                                                
 ─────────────────────────────────────────────────                              
 EXAMPLE_WELL = '000d7d20'                                                      
 tw     = load_typewell(EXAMPLE_WELL)                                           
 hw     = load_horizontal(EXAMPLE_WELL)                                         
 anchor = hw[hw['TVT_input'].notna()].copy().reset_index(drop=True)             
 evzone = hw[hw['TVT_input'].isna()].copy().reset_index(drop=True)              
                                                                                
 print(f'\nExample well: {EXAMPLE_WELL}')                                       
 print(f'  Anchor: {len(anchor):,} rows  TVT                                    
 {anchor["TVT"].min():.0f}–{anchor["TVT"].max():.0f} ft')                       
 print(f'  Eval:   {len(evzone):,} rows  TVT                                    
 {evzone["TVT"].min():.0f}–{evzone["TVT"].max():.0f} ft')                       
 print(f'  Typewell: {len(tw):,} rows')                                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                      1. Data structure and the role of GR                      

Each well has two files. The typewell is a nearby vertical reference well —     
because it drilled straight down, the depth-to-formation relationship is        
unambiguous. Its gamma-ray (GR) log is the reference sequence: high GR means    
shale, low GR means carbonate, and the pattern is stratigraphically unique.     

The horizontal well drills laterally through the same rock. GR still measures   
the same formations, but each measured depth now corresponds to an unknown      
stratigraphic position (TVT). The task is to reconstruct TVT in the evaluation  
zone where TVT_input is NaN.                                                    

The anchor zone (rows where TVT_input is not NaN) records a geologist's manual  
correlation — they slid the horizontal GR trace against the typewell and        
identified matching positions. The evaluation zone is where that process hasn't 
been done yet. Our models are replicating the geologist's pattern-matching,     
automatically.                                                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig, axes = plt.subplots(1, 2, figsize=(13, 5))                                
                                                                                
 axes[0].plot(anchor['MD'], anchor['TVT'], 'steelblue', lw=1.2, label='Anchor   
 (TVT known)')                                                                  
 axes[0].plot(evzone['MD'], evzone['TVT'], 'coral',     lw=1.2, label='Eval zon 
 (target)')                                                                     
 axes[0].set_xlabel('MD (ft)'); axes[0].set_ylabel('TVT (ft)')                  
 axes[0].set_title('TVT along the wellbore — anchor vs eval split')             
 axes[0].legend()                                                               
                                                                                
 hw_full = pd.concat([anchor, evzone]).sort_values('MD')                        
 axes[1].plot(anchor['GR'], anchor['MD'],  'steelblue', lw=0.8, alpha=0.8,      
 label='Horizontal GR (anchor)')                                                
 axes[1].plot(evzone['GR'], evzone['MD'],  'coral',     lw=0.8, alpha=0.8,      
 label='Horizontal GR (eval)')                                                  
 axes[1].plot(tw['GR'],     tw['TVT'],     'k',         lw=0.8, alpha=0.4,      
 label='Typewell GR (reference)')                                               
 axes[1].invert_yaxis()                                                         
 axes[1].set_xlabel('GR (API)'); axes[1].set_ylabel('Depth / TVT (ft)')         
 axes[1].set_title('GR barcode: horizontal well vs typewell reference')         
 axes[1].legend(fontsize=8)                                                     
                                                                                
 plt.tight_layout(); plt.show(); save_fig('writeup_structure')                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                       2. Predict drift, not absolute TVT                       

Raw TVT is 11,000–12,000 ft. Asking a tree model to predict it forces the model 
to learn the per-well offset first and the signal second. Reframing the target  
as drift = TVT − last_anchor_tvt (range ±16 ft) lets the model focus entirely on
the stratigraphic signal. This single change moved RMSE from 19.5 ft (worse than
null!) to 14.99 ft.                                                             

The null model (predicting last_anchor_tvt unchanged) achieves 15.91 ft RMSE — a
surprisingly strong baseline because most wells don't drift far from their      
anchor. Everything after R3 is learning to detect the minority of wells where   
the formation dips and drift accumulates.                                       

This framing also makes the outlier problem crisp: a well with large formation  
dip accumulates 50–100 ft of drift; a well with flat geology drifts near zero.  
The per-well OOF distribution (Section 10) shows that most wells are easy but a 
small tail drives almost all the aggregate error.                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Ground truth: TVT, drift, and MD_from_anchor for all 773 training wells      
 # Pre-computed in 02_features.ipynb; saved as a small parquet for Kaggle       
 gt              = pd.read_parquet(MODEL_DIR / 'train_gt.parquet')              
 y_tvt           = gt['TVT'].values.astype(np.float32)                          
 last_anchor_tvt = gt['last_anchor_tvt'].values.astype(np.float32)              
 drift           = y_tvt - last_anchor_tvt                                      
                                                                                
 fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))                              
 axes[0].hist(y_tvt,  bins=60, color='steelblue', edgecolor='none')             
 axes[0].set_title(f'Raw TVT  (std={y_tvt.std():.0f} ft \u2014 all signal is    
 noise at this scale)')                                                         
 axes[1].hist(drift,  bins=60, color='coral',     edgecolor='none')             
 axes[1].set_title(f'Drift = TVT \u2212 last_anchor_tvt  (std={drift.std():.1f} 
 ft \u2014 model focuses here)')                                                
 for ax in axes: ax.set_xlabel('ft')                                            
 plt.tight_layout(); plt.show(); save_fig('writeup_drift')                      
                                                                                
 print(f'Null model RMSE (just use last_anchor_tvt): {rmse(y_tvt,               
 last_anchor_tvt):.2f} ft')                                                     
 print(f'Drift mean={drift.mean():.2f} ft  std={drift.std():.2f} ft             
 range=[{drift.min():.1f}, {drift.max():.1f}]')                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

           3. GR as a stratigraphic barcode — the full feature family           

Gamma ray is the primary geological signal. A GR trace is a fingerprint of the  
stratigraphic column: the same formation always produces the same pattern,      
regardless of where the well was drilled. This means we can locate the          
horizontal well in the formation column by pattern-matching its GR against the  
typewell.                                                                       

Six layers of GR signal processing produce the bulk of the feature set:         

Imputation. Horizontal wells have sparse GR NaN gaps. Before any signal         
processing, missing values are filled using the typewell GR interpolated at the 
last known anchor TVT — a physics-informed fill that preserves the expected     
formation pattern rather than smearing adjacent values.                         

Rolling statistics. GR mean and std over ±12 and ±50 row windows capture local  
formation texture and the transition between formations.                        

Cross-correlation (NCC). A short horizontal GR window is slid against the       
typewell to find the best-matching depth. This gives a direct TVT estimate      
(GR_xcorr_tvt) and a confidence (GR_xcorr_conf). Multi-scale NCC repeats this at
3 window widths and 5 search radii — the multi-scale version is more robust to  
GR amplitude shifts between wells.                                              

Viterbi beam / particle filter. Both treat the typewell GR as a hidden Markov   
model: the hidden state is TVT, the emission is the difference between observed 
and expected GR. They add directional sequence constraints that are orthogonal  
to the point-estimate NCC approach.                                             

Estimator divergence (v4). When NCC, the formation KNN, and the beam search     
agree, confidence is high. When they disagree, the well is likely an outlier.   
Twelve pairwise divergence features (form_vs_ncc, ncc_vs_beam, etc.) encode this
uncertainty directly — the model learns to widen its predictions when estimators
disagree.                                                                       

DWT GR features (v4). A 5-level Daubechies-4 wavelet decomposition separates the
GR log into low-frequency trend (gr_dwt_approx5) and high-frequency oscillation 
(gr_dwt_detail_energy). The residual (gr_dwt_residual = GR − approx) captures   
rapid formation transitions that the rolling statistics miss. These three       
features add about 0.08 ft OOF improvement.                                     

The function below shows the NCC step; the full feature pipeline runs in        
build_features() (Section 14).                                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def ncc_match(hw_window, tw_gr, tw_tvt, seed_tvt, radius=150.0):               
     mask   = (tw_tvt >= seed_tvt - radius) & (tw_tvt <= seed_tvt + radius)     
     tw_sub, tvt_sub = tw_gr[mask], tw_tvt[mask]                                
     hw_w   = hw_window - hw_window.mean()                                      
     hw_w  /= hw_w.std() + 1e-6                                                 
     n      = len(hw_w)                                                         
     corrs  = np.array([                                                        
         float(np.dot(hw_w, (tw_sub[i:i+n] - tw_sub[i:i+n].mean()) /            
                            (tw_sub[i:i+n].std() + 1e-6)) / n)                  
         for i in range(len(tw_sub) - n)                                        
     ])                                                                         
     best_i = corrs.argmax()                                                    
     return float(tvt_sub[best_i + n // 2]), float(corrs[best_i]),              
 tvt_sub[:len(corrs)], corrs                                                    
                                                                                
 # Demo at one eval-zone position                                               
 tw_gr  = tw['GR'].values.astype(np.float32)                                    
 tw_tvt = tw['TVT'].values.astype(np.float32)                                   
 hw_all = pd.concat([anchor, evzone]).sort_values('MD').reset_index(drop=True)  
 hw_all['GR'] = hw_all['GR'].ffill().bfill()                                    
                                                                                
 HALF_W   = 20                                                                  
 demo_pos = len(anchor) + len(evzone) // 2                                      
 window   = hw_all['GR'].values[demo_pos - HALF_W : demo_pos].astype(np.float32 
 seed     = float(anchor['TVT'].iloc[-1])                                       
 tvt_est, best_corr, cand_tvt, cand_corrs = ncc_match(window, tw_gr, tw_tvt,    
 seed)                                                                          
 true_tvt = float(evzone['TVT'].iloc[len(evzone)//2])                           
                                                                                
 print(f'At eval-zone row {len(evzone)//2}:')                                   
 print(f'  True TVT:      {true_tvt:.1f} ft')                                   
 print(f'  NCC estimate:  {tvt_est:.1f} ft   (error {tvt_est - true_tvt:+.1f} f 
 peak r={best_corr:.3f})')                                                      
                                                                                
 fig, ax = plt.subplots(figsize=(9, 3))                                         
 ax.plot(cand_tvt, cand_corrs, 'steelblue', lw=1)                               
 ax.axvline(tvt_est,  color='coral', lw=1.5, ls='--', label=f'NCC estimate      
 {tvt_est:.0f} ft')                                                             
 ax.axvline(true_tvt, color='k',     lw=1.5, ls=':',  label=f'True TVT          
 {true_tvt:.0f} ft')                                                            
 ax.set_xlabel('Typewell TVT (ft)'); ax.set_ylabel('Pearson r')                 
 ax.set_title('NCC correlation curve: horizontal GR window vs typewell at each  
 candidate TVT')                                                                
 ax.legend(fontsize=9)                                                          
 plt.tight_layout(); plt.show(); save_fig('writeup_ncc_curve')                  
                                                                                
 # Pre-computed multi-scale NCC across all 773 training wells (from             
 train_gt.parquet + utils)                                                      
 print()                                                                        
 print(f'{"Feature":<32} {"r":>8} {"RMSE":>10}')                                
 print('-' * 52)                                                                
 rows = [                                                                       
     ('GR_ncc_tvt (multi-scale NCC)', 0.9993, 28.10),                           
     ('GR_xcorr_tvt (single-scale)',  0.9825, 118.48),                          
     ('tvt_extrap (linear)',          0.9818, 120.74),                          
     ('last_anchor_tvt (null)',       0.9997,  15.91),                          
 ]                                                                              
 for lbl, r, e in rows:                                                         
     print(f'{lbl:<32} {r:>8.4f} {e:>10.2f} ft')                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                4. Formation KNN — neighbors know the structure                 

Formation top depths (ANCC, EGFDU, etc.) give a direct physics formula: TVT ≈ −Z
+ formation_depth + b_well. But these columns are absent from test wells. The   
fix: find the 15 nearest training wells by (X, Y) and build a weighted plane fit
through their formation depths. A per-well bias b_well is calibrated from the   
anchor zone.                                                                    

Two KNN variants are combined: a plane-fit (captures regional dip direction) and
a row-level ANCC imputer (trained on 3.8M anchor rows for high spatial density).
Together they account for about 30 of the 184 features and carry the most weight
of any feature group in the SHAP analysis.                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # formation_plane_knn already loaded in setup                                  
 # predict() returns (values_array, distances); [0] gets the values array       
                                                                                
 x_med, y_med = float(hw['X'].median()), float(hw['Y'].median())                
 pred = formation_plane_knn.predict(np.array([[x_med, y_med]]))[0].flatten()    
 true = [float(hw[f].median()) for f in FORMATION_COLS]                         
                                                                                
 print(f'FormationPlaneKNN predictions vs training-well truth (well             
 {EXAMPLE_WELL}):')                                                             
 print(f'  {"Formation":<8} {"True":>8} {"Predicted":>10} {"Error":>8}')        
 for i, f in enumerate(FORMATION_COLS):                                         
     print(f'  {f:<8} {true[i]:>8.1f} {float(pred[i]):>10.1f}                   
 {float(pred[i])-true[i]:>+8.1f}')                                              
                                                                                
 # b_well calibration from anchor zone                                          
 ancc_anchor = formation_plane_knn.predict(anchor[['X','Y']].values)[0][:,      
 FORMATION_COLS.index('ANCC')]                                                  
 b_well      = float(np.median(anchor['TVT'].values + anchor['Z'].values -      
 ancc_anchor))                                                                  
 print(f'\nb_well = {b_well:.2f} ft  (anchor-zone calibration bias)')           
                                                                                
 ancc_eval = formation_plane_knn.predict(evzone[['X','Y']].values)[0][:,        
 FORMATION_COLS.index('ANCC')]                                                  
 tvt_phys  = -evzone['Z'].values + ancc_eval + b_well                           
 print(f'Physics TVT estimate RMSE (eval zone): {rmse(evzone["TVT"].values,     
 tvt_phys):.2f} ft')                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                5. Anchor zone features and slope extrapolation                 

Before the eval zone begins we have history: the TVT rate over the last 200     
anchor rows, GR statistics in the anchor zone, and the depth of the last        
confirmed anchor point. A linear extrapolation provides a directional prior:    

                                                                                
 tvt_extrap = last_anchor_tvt + tvt_rate × MD_from_anchor                       
                                                                                

Short-window slope features (v4) refine this: slopes over the last 10, 25, 50,  
and 100 anchor rows capture whether the well is accelerating its dip.           
slope_accel_10_50 = slope_k10 − slope_k50 tells the model if the formation dip  
is increasing near the anchor boundary — a strong signal for wells that are     
entering a steeply dipping interval.                                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 TAIL = 200                                                                     
 tail     = anchor.tail(TAIL)                                                   
 tvt_rate = (tail['TVT'].iloc[-1] - tail['TVT'].iloc[0]) / (tail['MD'].iloc[-1] 
 tail['MD'].iloc[0])                                                            
 last_tvt = float(anchor['TVT'].iloc[-1])                                       
 last_md  = float(anchor['MD'].iloc[-1])                                        
                                                                                
 md_from_anchor = evzone['MD'].values - last_md                                 
 tvt_extrap     = last_tvt + tvt_rate * md_from_anchor                          
                                                                                
 print(f'TVT rate (last 200 anchor rows): {tvt_rate:.4f} ft/ft MD')             
 print(f'Extrapolation RMSE: {rmse(evzone["TVT"].values, tvt_extrap):.2f} ft')  
 print(f'Null RMSE:          {rmse(evzone["TVT"].values, np.full(len(evzone),   
 last_tvt)):.2f} ft')                                                           
                                                                                
 fig, ax = plt.subplots(figsize=(10, 3.5))                                      
 ax.plot(md_from_anchor, evzone['TVT'].values - last_tvt, 'k',          lw=1.2, 
 label='True drift')                                                            
 ax.plot(md_from_anchor, tvt_extrap - last_tvt,           'steelblue',  lw=1.2, 
 label='Linear extrap from anchor')                                             
 ax.axhline(0, color='gray', lw=0.7, ls='--')                                   
 ax.set_xlabel('MD from anchor (ft)'); ax.set_ylabel('Drift (ft)')              
 ax.set_title(f'Linear extrapolation vs true drift \u2014 well {EXAMPLE_WELL}') 
 ax.legend(); plt.tight_layout(); plt.show(); save_fig('writeup_extrap')        
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                    6. Physics-informed sequence estimators                     

Three additional estimator families add orthogonal drift signals:               

 • Viterbi beam search (4 variants): an HMM over typewell GR. At each step the  
   most likely TVT is found by forward-beam Viterbi with varying emission and   
   transition sigmas.                                                           
 • Particle filter (3 variants): 500 particles track TVT through the typewell   
   GR. One variant uses TVT rate, one uses the ANCC plane-fit prior.            

These have larger standalone RMSE than NCC because they make directional errors 
on outlier wells. But they carry signal orthogonal to the KNN and NCC families, 
which is why their delta features (beam_med2_delta, pf_ancc_delta) rank in the  
top 20 SHAP features. The v4 divergence features (ncc_vs_beam, beam_vs_pf)      
explicitly capture when these estimators disagree — that disagreement is itself 
a signal that the well is in a hard interval.                                   

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                7. Feature engineering — full pipeline (offline)                

184 features are built per-well in 02_features.ipynb via build_features(). The  
feature groups are:                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 with open(MODEL_DIR / 'feature_cols.json') as f:                               
     FEATURE_COLS = json.load(f)                                                
                                                                                
 groups = [                                                                     
     ('Formation KNN (TVT physics)',  lambda c: c.startswith('tvtF') or c ==    
 'tvt_form_wls'),                                                               
     ('b_well offsets (6 fmts × 5)', lambda c: c.startswith('bw')),             
     ('Row/Dense ANCC KNN',           lambda c: 'knn_row' in c or 'dense' in c) 
     ('GR xcorr + NCC family',        lambda c: 'xcorr' in c or 'ncc' in c),    
     ('Viterbi beam',                 lambda c: 'beam' in c),                   
     ('Particle filter',              lambda c: c.startswith('pf_')),           
     ('Typewell GR offsets / TDSC',   lambda c: c.startswith('tw_diff') or      
 c.startswith('tdsc')),                                                         
     ('GR signal + DWT',              lambda c: (c.startswith('GR') and 'ncc' n 
 in c and 'xcorr' not in c)                                                     
                                                or c.startswith('gr_dwt')),     
     ('v4 divergence + slopes',       lambda c: c in                            
 {'form_vs_ncc','form_vs_pf','form_vs_beam',                                    
                                                       'ncc_vs_pf','ncc_vs_beam 
 'beam_vs_pf',                                                                  
                                                       'estimator_drift_range', 
 stimator_drift_max',                                                           
                                                       'estimator_drift_min','e 
 rap_k50_vs_extrap200',                                                         
                                                       'form_vs_extrap','tvt_ex 
 ap_k50','tvt_extrap_k10',                                                      
                                                       'tvt_extrap_k25','anchor 
 lope_k10','anchor_slope_k25',                                                  
                                                       'slope_accel_10_50','slo 
 _accel_25_100',                                                                
                                                       'prefix_tvt_md_slope50'} 
     ('Anchor / geometry / position', lambda c: True),                          
 ]                                                                              
                                                                                
 remaining = set(FEATURE_COLS)                                                  
 print(f'Total: {len(FEATURE_COLS)} features\n')                                
 print(f'{"Group":<35} {"Count":>6}')                                           
 print('-' * 43)                                                                
 for name, fn in groups[:-1]:                                                   
     matched = [c for c in remaining if fn(c)]                                  
     remaining -= set(matched)                                                  
     print(f'{name:<35} {len(matched):>6}')                                     
 print(f'{"Anchor / geometry / position":<35} {len(remaining):>6}')             
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

             8. Cross-validation — why GroupKFold is non-negotiable             

Each well has a distinctive GR fingerprint. Random KFold would put rows from the
same well in both train and val, letting the model memorize that fingerprint    
rather than learning transferable physics. GroupKFold(5) by well_id ensures no  
well appears in both train and val — simulating the unseen-well generalization  
needed for the 200 hidden test wells.                                           

This constraint is stricter than it sounds: two rows from the same well that are
5,000 ft apart in MD are still in the same group. The model must learn to read  
GR patterns it has never seen before, not interpolate between known GR traces.  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 groups_arr = gt['well_id'].values                                              
 gkf        = GroupKFold(n_splits=5)                                            
                                                                                
 print(f'{"Fold":<6} {"Train rows":>12} {"Wells":>8} {"Val rows":>10}           
 {"Wells":>8} {"Overlap":>9}')                                                  
 print('-' * 57)                                                                
 X_dummy = np.zeros((len(groups_arr), 1))                                       
 for i, (tr, val) in enumerate(gkf.split(X_dummy, groups=groups_arr)):          
     overlap = len(set(groups_arr[tr]) & set(groups_arr[val]))                  
     print(f'{i:<6} {len(tr):>12,} {len(np.unique(groups_arr[tr])):>8} '        
           f'{len(val):>10,} {len(np.unique(groups_arr[val])):>8} {overlap:>9}' 
 print('\nOverlap = 0 in every fold: no well leaks between train and val.')     
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                     9. Model training and Optuna (offline)                     

Three GBDTs (XGBoost, CatBoost, HistGradientBoosting) are trained in            
03_modeling.ipynb using GroupKFold(5). XGB and CB use early stopping (5000 max  
trees, stop at 50 non-improving rounds). HGB uses max_iter=5000 with            
early_stopping=False — the internal validation_fraction split would randomly mix
wells between train and val, leaking per-well GR patterns. Optuna TPE sampler   
finds the hyperparameters below.                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Load best hyperparameters found by Optuna (stored in MODEL_DIR on Kaggle)    
 with open(MODEL_DIR / 'xgb_drift_best_params.json') as f: xgb_p = json.load(f) 
 with open(MODEL_DIR / 'cb_drift_best_params.json')   as f: cb_p  = json.load(f 
                                                                                
 print('Best hyperparameters (from Optuna):')                                   
 for name, p in [('XGB', xgb_p), ('CB', cb_p)]:                                 
     print(f'  {name}: {p}')                                                    
                                                                                
 hgb_p = dict(max_iter=5000, learning_rate=0.05, max_depth=6,                   
              l2_regularization=0.1, max_features=0.7, early_stopping=False,    
 random_state=42)                                                               
 print(f'  HGB: {hgb_p}')                                                       
                                                                                
 # Training loop pattern (runs in scripts/save_v4_fold_models.py, ~3 hours for  
 full CV):                                                                      
 # ──────────────────────────────────────────────────────────────────────────── 
 # for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):         
 #     # XGBoost / CatBoost                                                     
 #     m = xgb.XGBRegressor(**xgb_p, n_estimators=5000, early_stopping_rounds=5 
 #     m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],         
 verbose=False)                                                                 
 #     m.save_model(f'models/xgb_fold{fold_i}.json')                            
 #     # HGB: early_stopping=False because internal validation_fraction random  
 split                                                                          
 #     # leaks per-well GR patterns; GroupKFold OOF is the true validation metr 
 #     hgb = HistGradientBoostingRegressor(**hgb_p)                             
 #     hgb.fit(X_sk[tr_idx], y[tr_idx])   # X_sk = nan_to_num(X, nan=0.0)       
 #     joblib.dump(hgb, f'models/hgb_fold{fold_i}.joblib', compress=3)          
 # ──────────────────────────────────────────────────────────────────────────── 
 print('\nNote: HGB uses early_stopping=False — internal validation_fraction do 
 not use')                                                                      
 print('GroupKFold and would leak per-well patterns. OOF CV is the correct      
 stopping criterion.')                                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

             10. Out-of-fold results and per-well outlier analysis              

OOF RMSE is our primary generalization estimate — each well is predicted by a   
fold model that never saw it.                                                   

The aggregate number (9.85 ft) hides a highly skewed distribution. Most wells   
are straightforward:                                                            

 • 277 / 773 wells (35.8%) achieve < 5 ft RMSE — the model reads their GR       
   cleanly                                                                      
 • Median per-well RMSE: 6.2 ft                                                 
 • 30 wells (3.9%) exceed 20 ft RMSE — these drive the majority of aggregate    
   error                                                                        

The hard wells share a pattern: large formation dip (true drift mean > 30 ft),  
long evaluation zones, and GR ambiguity where the horizontal well enters an     
interval the typewell didn't fully sample. The worst well (86454a6f) has 56.3 ft
OOF RMSE — its true drift reaches 96.7 ft across 7,964 evaluation rows, far     
outside the training distribution.                                              

The estimator divergence features (v4) partially address this: when NCC,        
formation KNN, and beam search all point in different directions on a well, the 
model learns to be more conservative. But structural ambiguity in the GR signal 
is the fundamental ceiling — the rock looks the same in two different parts of  
the formation.                                                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Load OOF drift predictions — v4 (saved after scripts/save_v4_fold_models.py) 
 oof_drift = {                                                                  
     'xgb': np.load(MODEL_DIR / 'xgb_oof_v4.npy'),                              
     'cb':  np.load(MODEL_DIR / 'cb_oof_v4.npy'),                               
     'hgb': np.load(MODEL_DIR / 'hgb_oof_v4.npy'),                              
 }                                                                              
                                                                                
 print('OOF TVT RMSE (v4 features, 184 total):')                                
 print(f'  null (last_anchor_tvt): {rmse(y_tvt, last_anchor_tvt):.4f} ft')      
 for name, od in oof_drift.items():                                             
     print(f'  {name}: {rmse(y_tvt, od + last_anchor_tvt):.4f} ft')             
                                                                                
 print('\nPairwise OOF drift correlation:')                                     
 print(pd.DataFrame(oof_drift).corr().round(4))                                 
                                                                                
 best_name = min(oof_drift, key=lambda n: rmse(y_tvt, oof_drift[n] +            
 last_anchor_tvt))                                                              
 residuals = y_tvt - (oof_drift[best_name] + last_anchor_tvt)                   
 rng = np.random.default_rng(0)                                                 
 idx = rng.choice(len(residuals), size=min(15_000, len(residuals)),             
 replace=False)                                                                 
                                                                                
 fig, axes = plt.subplots(1, 2, figsize=(12, 4))                                
 axes[0].scatter((oof_drift[best_name] + last_anchor_tvt)[idx], residuals[idx], 
                 alpha=0.08, s=2, color='steelblue')                            
 axes[0].axhline(0, color='r', lw=1)                                            
 axes[0].set(xlabel='Predicted TVT (ft)', ylabel='Residual (ft)',               
             title=f'{best_name.upper()} OOF residuals vs predicted')           
 axes[1].hist(residuals, bins=80, color='steelblue', edgecolor='none')          
 axes[1].set(xlabel='Residual (ft)',                                            
             title=f'Residual distribution  RMSE={rmse(y_tvt,                   
 oof_drift[best_name]+last_anchor_tvt):.2f} ft')                                
 plt.tight_layout(); plt.show(); save_fig('writeup_residuals')                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                           11. Ensemble — NNLS blend                            

Models are highly correlated (OOF Pearson ~0.96–0.97), so elaborate stacking    
risks overfitting OOF noise. Non-negative least squares (NNLS) finds the optimal
non-negative weights — simple and overfitting-resistant.                        

HistGradientBoosting uses histogram binning with a different split criterion    
than XGB and CB, giving it structural diversity despite reading the same        
features. It earns 36% NNLS weight despite weaker solo RMSE. LightGBM earned 0% 
in early runs and was dropped.                                                  

On the outlier wells, the ensemble benefit is most pronounced: when XGB and CB  
disagree on a hard well, HGB often has an independent read that pulls the blend 
toward the right answer.                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Load blend weights from MODEL_DIR (same weights used in inference)           
 with open(MODEL_DIR / 'blend_weights.json') as f:                              
     BLEND_WEIGHTS = json.load(f)                                               
 print('Blend weights:', {k: f'{v:.4f}' for k, v in BLEND_WEIGHTS.items()})     
                                                                                
 blend_oof = sum(BLEND_WEIGHTS.get(k, 0.0) * oof_drift[k] for k in ['xgb', 'cb' 
 'hgb'])                                                                        
 print(f'Blend OOF TVT RMSE: {rmse(y_tvt, blend_oof + last_anchor_tvt):.4f} ft' 
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                              12. Post-processing                               

Savitzky-Golay smoothing (window 17, order 3) removes row-to-row noise without  
blurring formation boundaries. Applied per-well to blended drift predictions.   

On most wells this has negligible effect (< 0.1 ft). On outlier wells with      
high-frequency GR noise it can make a visible difference — the smooth is trading
off slightly worse sharp-boundary detection for better noise suppression on     
long, drifting eval zones.                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 SG_W_DEMO, SG_P_DEMO  = 17, 3                                                  
 dmd              = gt['MD_from_anchor'].values.astype(np.float32)              
 groups_arr_gt    = gt['well_id'].values                                        
                                                                                
 drift_sg = blend_oof.copy()                                                    
 for wid in np.unique(groups_arr_gt):                                           
     mask  = groups_arr_gt == wid                                               
     chunk = blend_oof[mask]                                                    
     if len(chunk) >= SG_W_DEMO:                                                
         drift_sg[mask] = savgol_filter(chunk, SG_W_DEMO, SG_P_DEMO)            
 print(f'After SG smooth (w={SG_W_DEMO}, p={SG_P_DEMO}): {rmse(y_tvt, drift_sg  
 last_anchor_tvt):.4f} ft  (vs raw blend: {rmse(y_tvt, blend_oof +              
 last_anchor_tvt):.4f} ft)')                                                    
                                                                                
 wmask = groups_arr_gt == EXAMPLE_WELL                                          
 wmd   = dmd[wmask]                                                             
 fig, ax = plt.subplots(figsize=(10, 3.5))                                      
 ax.plot(wmd, blend_oof[wmask], 'steelblue', lw=0.8, alpha=0.7, label='Raw      
 blend')                                                                        
 ax.plot(wmd, drift_sg[wmask],  'coral',     lw=1.5,             label=f'After  
 (w={SG_W_DEMO})')                                                              
 ax.plot(wmd, y_tvt[wmask] - last_anchor_tvt[wmask], 'k', lw=1.2, label='True   
 drift')                                                                        
 ax.set(xlabel='MD from anchor (ft)', ylabel='Drift (ft)',                      
        title=f'Post-processing — well {EXAMPLE_WELL}')                         
 ax.legend(); plt.tight_layout(); plt.show(); save_fig('writeup_postproc')      
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                                  13. Results                                   

See the table at the top for the full round-by-round breakdown.                 

Best LB: 8.905 ft — R11: v4 features + HistGradientBoosting, NNLS blend XGB     
20.8% + CB 43.2% + HGB 36.0%.                                                   

The OOF → LB gap is 0.94 ft (OOF 9.85 → LB 8.905), larger than earlier rounds   
(~0.65 ft). This suggests the v4 features — especially the estimator divergence 
and DWT features — generalize particularly well to the hidden test distribution,
likely because they encode uncertainty rather than committing to a single TVT   
estimate.                                                                       

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                                What didn't work                                

Absolute TVT target. 19.50 ft — 3.6 ft worse than the null model. The per-well  
offset swamps the signal.                                                       

BiLSTM / TCN. ~14.6 ft OOF. Sequential models memorize per-well GR fingerprints 
rather than learning transferable physics. GroupKFold(5) with 773 wells doesn't 
provide enough diversity for a sequence model to generalize to unseen GR traces.

DTW features. Regressed OOF from 10.008 to 10.07 ft. DTW uses L2 distance       
(amplitude-sensitive) while NCC uses Pearson r (amplitude-invariant). Adding    
amplitude-sensitive features alongside amplitude-invariant ones confused the    
model rather than helping it.                                                   

Online learning (4 variants). All variants degraded OOF. The augmented training 
signal has drift mean in the hundreds of feet vs. eval-zone mean of 1.6 ft — the
distribution mismatch is fundamental, not a tuning problem.                     

ExtraTrees. Solo RMSE 10.89 ft — too weak on its own. OOF correlation with XGB  
(0.93) does show structural diversity, but the accuracy gap prevents it from    
earning meaningful NNLS weight.                                                 

Coordinate-overlap post-processing. Only 12/773 training wells share coordinates
with any other well (< 2%). The apparent gain in a reference notebook was an    
artifact of test-set leakage (visible test wells = training wells on Kaggle). No
generalizable benefit.                                                          

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                          14. Full inference pipeline                           

The cells below run the complete pipeline on the test wells: feature building,  
fold model prediction, NNLS blend, SG smoothing, and submission.csv output.     

On Kaggle (~200 test wells): ~15–20 minutes for feature building (GR signal     
processing dominates), ~2–3 minutes for model inference. Locally (3 visible test
wells): ~3 minutes total.                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 TW_OFFSETS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80],          
 dtype=np.float32)                                                              
                                                                                
 def build_features(well_id: str, split: str = 'train', loo: bool = True) ->    
 pd.DataFrame:                                                                  
     """                                                                        
     Build eval-zone features for one well.                                     
     loo=True excludes this well from the formation KNN (use for training wells 
     """                                                                        
     tw = load_typewell(well_id, split=split)                                   
     hw = load_horizontal(well_id, split=split)                                 
                                                                                
     anchor_mask = hw['TVT_input'].notna()                                      
     anchor = hw[anchor_mask].copy().reset_index(drop=True)                     
     eval_z = hw[~anchor_mask].copy().reset_index(drop=True)                    
                                                                                
     if len(eval_z) == 0:                                                       
         return pd.DataFrame()                                                  
                                                                                
     # ---- Anchor zone summary ----                                            
     last_tvt  = float(anchor['TVT_input'].iloc[-1])                            
     last_md   = float(anchor['MD'].iloc[-1])                                   
     last_z    = float(anchor['Z'].iloc[-1])                                    
     n_anchor  = len(anchor)                                                    
                                                                                
     tail = anchor.tail(ANCHOR_TAIL)                                            
     if len(tail) >= 2:                                                         
         dtvt     = tail['TVT_input'].iloc[-1] - tail['TVT_input'].iloc[0]      
         dmd_tail = tail['MD'].iloc[-1] - tail['MD'].iloc[0]                    
         tvt_rate = dtvt / dmd_tail if abs(dmd_tail) > 1e-6 else 0.0            
     else:                                                                      
         tvt_rate = 0.0                                                         
                                                                                
     md_from_anchor = eval_z['MD'].values - last_md                             
     tvt_extrap     = last_tvt + tvt_rate * md_from_anchor                      
                                                                                
     if n_anchor >= 2:                                                          
         tvt_step_per_row = abs(                                                
             (anchor['TVT_input'].iloc[-1] - anchor['TVT_input'].iloc[0]) /     
 (n_anchor - 1)                                                                 
         )                                                                      
         tvt_step_per_row = max(0.05, tvt_step_per_row)                         
     else:                                                                      
         tvt_step_per_row = 1.0                                                 
                                                                                
     _tail_for_rate = anchor.tail(ANCHOR_TAIL)                                  
     if len(_tail_for_rate) >= 2:                                               
         _tail_tvt_diff = abs(                                                  
             _tail_for_rate['TVT_input'].iloc[-1] -                             
 _tail_for_rate['TVT_input'].iloc[0]                                            
         )                                                                      
         tail_tvt_step = max(0.001, _tail_tvt_diff / (len(_tail_for_rate) - 1)) 
     else:                                                                      
         tail_tvt_step = 0.01                                                   
                                                                                
     # ---- Prefix features (anchor tail TVT behavior) ----                     
     anchor_tvt_vals = anchor['TVT_input'].values                               
     anchor_md_vals  = anchor['MD'].values                                      
     anchor_z_vals   = anchor['Z'].values                                       
                                                                                
     def _tail_mean_diff(vals, n):                                              
         v = vals[-(n + 1):]                                                    
         return float(np.diff(v).mean()) if len(v) >= 2 else 0.0                
                                                                                
     def _tail_slope(y, x, n):                                                  
         y, x = y[-n:], x[-n:]                                                  
         if len(y) < 2: return 0.0                                              
         cx = x - x.mean()                                                      
         d  = float(np.dot(cx, cx))                                             
         return float(np.dot(cx, y - y.mean()) / d) if d > 0 else 0.0           
                                                                                
     prefix_tvt_step20       = _tail_mean_diff(anchor_tvt_vals, 20)             
     prefix_tvt_step100      = _tail_mean_diff(anchor_tvt_vals, 100)            
     prefix_tvt_md_slope100  = _tail_slope(anchor_tvt_vals, anchor_md_vals, 100 
                                                                                
     # ---- GR imputation ----                                                  
     tw_sorted = tw.sort_values('TVT').dropna(subset=['GR'])                    
     tw_tvt    = tw_sorted['TVT'].values                                        
     tw_gr     = tw_sorted['GR'].values                                         
     tw_interp = build_typewell_interp(tw)                                      
                                                                                
     gr_full   = impute_gr_with_typewell(hw, tw).values.astype(float)           
     gr_eval   = eval_z['GR'].copy().values.astype(float)                       
     null_mask = np.isnan(gr_eval)                                              
     if null_mask.any():                                                        
         gr_eval[null_mask] = tw_interp(tvt_extrap[null_mask])                  
     gr_full[n_anchor:] = gr_eval                                               
                                                                                
     anchor_gr_vals = gr_full[:n_anchor]                                        
     finite_gr      = anchor_gr_vals[np.isfinite(anchor_gr_vals)]               
     p5, p95        = (np.nanpercentile(finite_gr, [5, 95]) if len(finite_gr) > 
 10                                                                             
                       else (finite_gr.min(), finite_gr.max()))                 
     gr_norm_p5p95  = (gr_eval - p5) / max(p95 - p5, 1.0)                       
                                                                                
     # ---- Fractional position in eval zone ----                               
     n_eval  = len(eval_z)                                                      
     frac    = np.arange(n_eval) / max(n_eval - 1, 1)                           
     frac2   = frac ** 2                                                        
     sqrt_frac = np.sqrt(frac)                                                  
                                                                                
     # ---- GR signal features (1st/2nd derivative, envelope) ----              
     gr_d1  = np.diff(gr_eval, prepend=gr_eval[0]).astype(np.float32)           
     gr_d2  = np.diff(gr_d1,   prepend=gr_d1[0]).astype(np.float32)             
     gr_env = (pd.Series(gr_eval)                                               
                .rolling(25, center=True, min_periods=1).max()                  
                .values.astype(np.float32))                                     
                                                                                
     # ---- Formation KNN: plane-fit (new primary) + legacy IDW ----            
     x_ev = eval_z['X'].values                                                  
     y_ev = eval_z['Y'].values                                                  
     z_ev = eval_z['Z'].values                                                  
                                                                                
     exclude = well_id if loo else None                                         
     xy_anchor_q = np.column_stack([anchor['X'].values, anchor['Y'].values])    
     xy_eval_q   = np.column_stack([x_ev, y_ev])                                
                                                                                
     # Plane-fit KNN (new primary spatial imputer)                              
     form_eval_plane,   knn_dist_plane = formation_plane_knn.predict(xy_eval_q, 
 exclude_wid=exclude)                                                           
     form_anchor_plane, _              = formation_plane_knn.predict(xy_anchor_ 
 exclude_wid=exclude)                                                           
                                                                                
     # Legacy IDW KNN (kept for backward compat)                                
     form_eval_idw,   knn_dist_idw = formation_knn.predict(xy_eval_q,           
 exclude_wid=exclude)                                                           
     form_anchor_idw, _            = formation_knn.predict(xy_anchor_q,         
 exclude_wid=exclude)                                                           
                                                                                
     # Use plane-fit as primary source for b_well offsets                       
     form_anchor_pred = form_anchor_plane                                       
     form_eval_pred   = form_eval_plane                                         
     knn_dist         = knn_dist_plane                                          
                                                                                
     # ---- Per-formation segmented b_well offsets ----                         
     form_b = {}                                                                
     for fi, fn in enumerate(FORMATION_COLS):                                   
         bv   = anchor_tvt_vals + anchor_z_vals - form_anchor_pred[:, fi]       
         n_bv = len(bv)                                                         
         t1, t2 = n_bv // 3, 2 * n_bv // 3                                      
         b_full  = float(np.median(bv))                                         
         b_late  = float(np.median(bv[-50:] if n_bv >= 50 else bv))             
         b_early = float(np.median(bv[:max(1, t1)]) if t1 > 0 else b_full)      
         b_mid   = float(np.median(bv[t1:max(t1 + 1, t2)]) if t2 > t1 else      
 b_full)                                                                        
         w_exp   = np.exp(0.02 * np.arange(n_bv))                               
         b_wls   = float(np.dot(w_exp / w_exp.sum(), bv))                       
         form_b[fn] = dict(full=b_full, early=b_early, mid=b_mid, late=b_late,  
 wls=b_wls)                                                                     
                                                                                
     tvt_form_feats = {}                                                        
     for fi, fn in enumerate(FORMATION_COLS):                                   
         fb = form_b[fn]                                                        
         tvt_form_feats[f'tvtF_{fn}']   = (-z_ev + form_eval_pred[:, fi] +      
 fb['full']).astype(np.float32)                                                 
         tvt_form_feats[f'tvtFw_{fn}']  = (-z_ev + form_eval_pred[:, fi] +      
 fb['wls']).astype(np.float32)                                                  
         tvt_form_feats[f'tvtF50_{fn}'] = (-z_ev + form_eval_pred[:, fi] +      
 fb['late']).astype(np.float32)                                                 
                                                                                
     b_full  = form_b['ANCC']['full']                                           
     b_late  = form_b['ANCC']['late']                                           
     b_wls   = form_b['ANCC']['wls']                                            
     b_early = form_b['ANCC']['early']                                          
     b_mid   = form_b['ANCC']['mid']                                            
     ancc_eval  = form_eval_pred[:, 0]                                          
     egfdu_eval = form_eval_pred[:, 3]                                          
     buda_eval  = form_eval_pred[:, 5]                                          
     tvt_form_full  = tvt_form_feats['tvtF_ANCC']                               
     tvt_form_late  = tvt_form_feats['tvtF50_ANCC']                             
     tvt_form_wls   = tvt_form_feats['tvtFw_ANCC']                              
     tvt_form_egfdu = tvt_form_feats['tvtF_EGFDU']                              
     tvt_form_buda  = tvt_form_feats['tvtF_BUDA']                               
                                                                                
     # ---- RowKNN: row-level ANCC imputer ----                                 
     row_ancc_eval, row_ancc_std_eval, row_dist_eval = row_knn.predict(xy_eval_ 
 exclude_wid=exclude)                                                           
     row_ancc_anch, _,                 _             =                          
 row_knn.predict(xy_anchor_q, exclude_wid=exclude)                              
                                                                                
     # b_well offset from RowKNN                                                
     if n_anchor > 0:                                                           
         b_per_row_row = anchor_tvt_vals + anchor_z_vals -                      
 row_ancc_anch.astype(np.float64)                                               
         w_row = np.exp(0.02 * np.arange(len(b_per_row_row)))                   
         b_well_row = float(np.dot(w_row / w_row.sum(), b_per_row_row))         
     else:                                                                      
         b_well_row = 0.0                                                       
     knn_row_tvt_pred = -z_ev + row_ancc_eval.astype(np.float64) + b_well_row   
     knn_row_tvt_pred_delta = (knn_row_tvt_pred - last_tvt).astype(np.float32)  
                                                                                
     # ---- DenseANCCImputer ----                                               
     dense_ancc_eval, dense_dist_eval, dense_std_eval =                         
 dense_imputer.predict(xy_eval_q, exclude_wid=exclude)                          
     dense_ancc_anch, _, _                            =                         
 dense_imputer.predict(xy_anchor_q, exclude_wid=exclude)                        
                                                                                
     if n_anchor > 0:                                                           
         b_per_row_dense = anchor_tvt_vals + anchor_z_vals -                    
 dense_ancc_anch.astype(np.float64)                                             
         w_den = np.exp(0.02 * np.arange(len(b_per_row_dense)))                 
         b_well_dense = float(np.dot(w_den / w_den.sum(), b_per_row_dense))     
     else:                                                                      
         b_well_dense = 0.0                                                     
     dense_tvt_pred = -z_ev + dense_ancc_eval.astype(np.float64) + b_well_dense 
     dense_tvt_pred_delta = (dense_tvt_pred - last_tvt).astype(np.float32)      
                                                                                
     # ---- Rolling GR ----                                                     
     gr_series   = pd.Series(gr_full)                                           
     roll_s_mean = gr_series.rolling(ROLL_SHORT, center=True,                   
 min_periods=1).mean().values                                                   
     roll_s_std  = gr_series.rolling(ROLL_SHORT, center=True,                   
 min_periods=1).std().fillna(0).values                                          
     roll_l_mean = gr_series.rolling(ROLL_LONG,  center=True,                   
 min_periods=1).mean().values                                                   
     roll_l_std  = gr_series.rolling(ROLL_LONG,  center=True,                   
 min_periods=1).std().fillna(0).values                                          
                                                                                
     gr_tw_last_anchor = float(tw_interp(last_tvt))                             
     gr_diff           = gr_eval - gr_tw_last_anchor                            
                                                                                
     # ---- xcorr ----                                                          
     xcorr_tvt, xcorr_conf = gr_xcorr_batch(                                    
         gr_full=gr_full, n_anchor=n_anchor, tvt_seeds=tvt_extrap,              
         tw_tvt=tw_tvt, tw_gr=tw_gr, tvt_step_per_row=tail_tvt_step,            
         window_rows=XCORR_WINDOW, search_radius=XCORR_RADIUS,                  
         xcorr_step=XCORR_STEP, stride=XCORR_STRIDE,                            
     )                                                                          
     xcorr_delta    = xcorr_tvt - tvt_extrap                                    
     xcorr_smooth   = pd.Series(xcorr_tvt.astype(float)).rolling(10, center=Tru 
 min_periods=1).mean().values                                                   
     tw_gr_at_xcorr = tw_interp(xcorr_tvt).astype(float)                        
     xcorr_residual = gr_eval - tw_gr_at_xcorr                                  
                                                                                
     # ---- Visible GR shift ----                                               
     gr_shift_ft, gr_shift_corr, gr_shift_bias = visible_gr_shift_fit(          
         anchor_tvt_vals, anchor_gr_vals, tw_tvt, tw_gr,                        
     )                                                                          
                                                                                
     # ---- Multi-scale NCC vs typewell (original, per-row seeds) ----          
     ncc_tvt, ncc_conf = multi_scale_ncc(                                       
         eval_gr=gr_eval, anchor_gr=anchor_gr_vals,                             
         tw_tvt=tw_tvt, tw_gr=tw_gr,                                            
         last_anchor_tvt=last_tvt, tvt_rate=tail_tvt_step,                      
         search_radius=NCC_RADIUS, half_widths=NCC_HALFWIDTHS,                  
 temperature=NCC_TEMP,                                                          
     )                                                                          
     ncc_delta = (ncc_tvt - last_tvt).astype(np.float32)                        
                                                                                
     # ---- Multi-scale NCC vs ANCHOR zone (new: anchor GR as template) ----    
     ncc_anchor_tvt, ncc_anchor_conf = multi_scale_ncc_anchor(                  
         anchor_gr=anchor_gr_vals.astype(np.float32),                           
         anchor_tvt=anchor_tvt_vals.astype(np.float32),                         
         eval_gr=gr_eval.astype(np.float32),                                    
         half_widths=NCC_HALFWIDTHS,                                            
         stride=3,                                                              
     )                                                                          
     ncc_anchor_delta = (ncc_anchor_tvt - last_tvt).astype(np.float32)          
                                                                                
     # ---- Softmax blend: NCC × formation ----                                 
     _ncc_w = 1.0 / (1.0 + np.exp(-3.0 * ncc_conf.astype(np.float64)))          
     ncc_form_blend = (_ncc_w * ncc_tvt + (1.0 - _ncc_w) *                      
 tvt_form_feats['tvtFw_ASTNU']).astype(np.float32)                              
     ncc_form_delta = (ncc_form_blend - last_tvt).astype(np.float32)            
                                                                                
     # ---- GR signal divergence: std across estimators ----                    
     pf_placeholder = np.full(n_eval, last_tvt, dtype=np.float32)  # will be    
 filled after PF                                                                
     form_preds_stack = np.stack([tvt_form_feats[f'tvtF_{fn}'] for fn in        
 FORMATION_COLS], axis=1)                                                       
     form_std = form_preds_stack.std(axis=1).astype(np.float32)                 
                                                                                
     # ---- Z-velocity physics prior ----                                       
     z_beta, z_intercept, _ = learn_z_beta(anchor_tvt_vals, anchor_z_vals,      
 anchor_md_vals)                                                                
                                                                                
     # ---- GR for beam/PF ----                                                 
     _gr_raw = eval_z['GR'].copy().values.astype(float)                         
     gr_beam = (pd.Series(_gr_raw)                                              
                .interpolate(limit_direction='both')                            
                .fillna(float(np.nanmean(tw_gr)))                               
                .rolling(5, center=True, min_periods=1).mean()                  
                .values.astype(np.float32))                                     
                                                                                
     # ---- Viterbi beam search (4 variants) ----                               
     beam_results = {}                                                          
     for name, emit_s, move_s in BEAM_VARIANTS:                                 
         path = viterbi_tvt(                                                    
             gr_beam, tw_tvt, tw_gr,                                            
             last_tvt, tvt_step_per_row,                                        
             emit_sigma=emit_s, move_sigma=move_s,                              
             search_radius=BEAM_RADIUS, grid_step=1.0,                          
         )                                                                      
         beam_results[f'{name}_delta'] = (path - last_tvt).astype(np.float32)   
                                                                                
     # ---- Particle filter: TVT-based (2 variants) ----                        
     eval_z_vals = eval_z['Z'].values.astype(np.float64)                        
     eval_md_vals = eval_z['MD'].values.astype(np.float64)                      
                                                                                
     pf_results = {}                                                            
     for name, mom, gr_s, jit_s in PF_VARIANTS:                                 
         est = particle_filter_tvt(                                             
             gr_beam, tw_tvt, tw_gr,                                            
             last_tvt, tail_tvt_step,                                           
             n_particles=PF_PARTICLES, momentum=mom,                            
             gr_sigma=gr_s, jitter_sigma=jit_s, seed=42,                        
         )                                                                      
         pf_results[f'{name}_delta'] = (est - last_tvt).astype(np.float32)      
                                                                                
     # ---- Particle filter: ANCC-based ----                                    
     pf_ancc_tvt, pf_ancc_std = particle_filter_ancc(                           
         eval_gr=gr_beam.astype(np.float64),                                    
         eval_z=eval_z_vals,                                                    
         eval_md=eval_md_vals,                                                  
         tw_tvt=tw_tvt, tw_gr=tw_gr,                                            
         last_anchor_tvt=last_tvt,                                              
         last_anchor_z=last_z,                                                  
         last_anchor_md=last_md,                                                
         anchor_tvt=anchor_tvt_vals,                                            
         anchor_z=anchor_z_vals,                                                
         anchor_md=anchor_md_vals,                                              
         n_particles=PF_PARTICLES,                                              
         seed=42,                                                               
     )                                                                          
     pf_ancc_delta = (pf_ancc_tvt - last_tvt).astype(np.float32)                
                                                                                
     # ---- Signal divergence (after PF is available) ----                      
     pf_z_est  = last_tvt + pf_results['pf_z_delta']                            
     sig_std   = np.stack([                                                     
         pf_z_est,                                                              
         last_tvt + beam_results['beam_med2_delta'],                            
         ncc_anchor_tvt,                                                        
         tvt_form_feats['tvtF_ANCC'],                                           
     ], axis=1).std(axis=1).astype(np.float32)                                  
                                                                                
     pf_vs_spatial = (pf_ancc_delta - (form_std / (form_std.mean() +            
 1e-3))).astype(np.float32)                                                     
     sc_vs_beam    = (ncc_anchor_delta -                                        
 beam_results['beam_med2_delta']).astype(np.float32)                            
                                                                                
     # ---- Typewell GR residuals anchored at NCC-anchor estimate ----          
     tdsc_offsets = [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]                  
     tdsc_feats = {                                                             
         f'tdsc_{int(o)}': (gr_eval - np.interp(                                
             ncc_anchor_tvt + o, tw_tvt, tw_gr                                  
         )).astype(np.float32)                                                  
         for o in tdsc_offsets                                                  
     }                                                                          
                                                                                
     # ---- Typewell GR offset dictionary ----                                  
     tw_diff = {                                                                
         f'tw_diff_{int(off)}': (gr_eval - float(tw_interp(last_tvt +           
 float(off)))).astype(np.float32)                                               
         for off in TW_OFFSETS                                                  
     }                                                                          
                                                                                
     # ---- Geometry ----                                                       
     md_ev  = eval_z['MD'].values                                               
     hw_md  = hw['MD'].values                                                   
     hw_x, hw_y, hw_z = hw['X'].values, hw['Y'].values, hw['Z'].values          
     dmd    = np.where(np.abs(np.diff(hw_md, prepend=hw_md[0])) < 1e-6, np.nan, 
                       np.diff(hw_md, prepend=hw_md[0]))                        
     dz_dmd = pd.Series(np.diff(hw_z, prepend=hw_z[0]) /                        
 dmd).bfill().ffill().values                                                    
     dx_dmd = pd.Series(np.diff(hw_x, prepend=hw_x[0]) /                        
 dmd).bfill().ffill().values                                                    
     dy_dmd = pd.Series(np.diff(hw_y, prepend=hw_y[0]) /                        
 dmd).bfill().ffill().values                                                    
                                                                                
     # ---- Trajectory kinematics (inclination, azimuth, DLS, build rate) ----  
     kin       = compute_trajectory_kinematics(hw_md, hw_x, hw_y, hw_z)         
     incl_all  = kin['incl_deg']                                                
     azi_all   = kin['azi_deg']                                                 
     dls_all   = kin['dls']                                                     
     build_all = kin['build_rate']                                              
                                                                                
     incl_ev     = incl_all[n_anchor:]                                          
     azi_ev      = azi_all[n_anchor:]                                           
     dls_ev      = dls_all[n_anchor:]                                           
     build_ev    = build_all[n_anchor:]                                         
     cos_incl_ev = np.cos(np.radians(incl_ev)).astype(np.float32)               
     sin_incl_ev = np.sin(np.radians(incl_ev)).astype(np.float32)               
                                                                                
     # ---- Apparent dip from anchor zone kinematics ----                       
     dip = estimate_apparent_dip(anchor_tvt_vals, anchor_md_vals,               
 incl_all[:n_anchor])                                                           
     b_dip_full  = float(dip['b_full'])                                         
     b_dip_late  = float(dip['b_late'])                                         
     b_dip_early = float(dip['b_early'])                                        
     b_dip_slope = float(dip['b_slope'])                                        
                                                                                
     # Physics TVT extrapolation: dTVT_i = (cos(incl_i) + sin(incl_i)*b) * dMD_ 
     _dmd_steps = np.diff(eval_z['MD'].values, prepend=last_md)                 
     tvt_dip_full_arr = (last_tvt + np.cumsum(                                  
         (cos_incl_ev + sin_incl_ev * np.float32(b_dip_full)) * _dmd_steps      
     )).astype(np.float32)                                                      
     tvt_dip_late_arr = (last_tvt + np.cumsum(                                  
         (cos_incl_ev + sin_incl_ev * np.float32(b_dip_late)) * _dmd_steps      
     )).astype(np.float32)                                                      
                                                                                
     # Signed azimuth deviation from anchor circular mean                       
     _azi_anch_rad = np.radians(azi_all[:n_anchor])                             
     _azi_mean_rad = float(np.arctan2(np.sin(_azi_anch_rad).mean(),             
 np.cos(_azi_anch_rad).mean()))                                                 
     azi_delta_ev  = np.degrees(np.arctan2(                                     
         np.sin(np.radians(azi_ev) - _azi_mean_rad),                            
         np.cos(np.radians(azi_ev) - _azi_mean_rad),                            
     )).astype(np.float32)                                                      
                                                                                
     # Formation plane dip direction at well centroid (numerical gradient of    
 plane-fit KNN)                                                                 
     _xc, _yc = float(np.median(x_ev)), float(np.median(y_ev))                  
     _h = 500.0                                                                 
     _fc  = formation_plane_knn.predict(np.array([[_xc,      _yc     ]]),       
 exclude_wid=exclude)[0][0]                                                     
     _fdx = formation_plane_knn.predict(np.array([[_xc + _h, _yc     ]]),       
 exclude_wid=exclude)[0][0]                                                     
     _fdy = formation_plane_knn.predict(np.array([[_xc,      _yc + _h]]),       
 exclude_wid=exclude)[0][0]                                                     
     _ai  = FORMATION_COLS.index('ANCC')                                        
     plane_dip_x = float((_fdx[_ai] - _fc[_ai]) / _h)  # d(ANCC_depth)/dX       
     plane_dip_y = float((_fdy[_ai] - _fc[_ai]) / _h)  # d(ANCC_depth)/dY       
                                                                                
     # Directional apparent dip: formation-deepening rate in wellbore azimuth   
 direction                                                                      
     apparent_dip_dir = (                                                       
         plane_dip_x * np.sin(np.radians(azi_ev)) +                             
         plane_dip_y * np.cos(np.radians(azi_ev))                               
     ).astype(np.float32)                                                       
                                                                                
                                                                                
     # ---- Well context ----                                                   
     anchor_gr_mean   = float(np.nanmean(anchor_gr_vals))                       
     anchor_gr_std    = float(np.nanstd(anchor_gr_vals)) + 1e-6                 
     anchor_tvt_mean  = float(np.nanmean(anchor_tvt_vals))                      
     anchor_tvt_range = float(np.nanmax(anchor_tvt_vals) -                      
 np.nanmin(anchor_tvt_vals))                                                    
     n_anchor_rows    = int(n_anchor)                                           
     tw_in_range = tw_sorted[(tw_sorted['TVT'] >= anchor_tvt_vals.min()) &      
                             (tw_sorted['TVT'] <= anchor_tvt_vals.max())]       
     tw_gr_anchor_mean = float(tw_in_range['GR'].mean()) if len(tw_in_range) >  
 else anchor_gr_mean                                                            
                                                                                
     # ---- GR lag/lead (normalized per-well) ----                              
     gr_norm   = (gr_eval - anchor_gr_mean) / anchor_gr_std                     
     gr_norm_s = pd.Series(gr_norm)                                             
     gr_lag_5   = gr_norm_s.shift(5).values;  gr_lag_10  =                      
 gr_norm_s.shift(10).values                                                     
     gr_lag_25  = gr_norm_s.shift(25).values; gr_lag_50  =                      
 gr_norm_s.shift(50).values                                                     
     gr_lag_100 = gr_norm_s.shift(100).values                                   
     gr_lead_5  = gr_norm_s.shift(-5).values; gr_lead_10 =                      
 gr_norm_s.shift(-10).values                                                    
     gr_lead_25 = gr_norm_s.shift(-25).values                                   
                                                                                
     eval_original_idx = hw[~anchor_mask].index.values                          
                                                                                
     out = pd.DataFrame({                                                       
         'well_id':   well_id,                                                  
         'row_index': eval_original_idx,                                        
         # Formation plane-fit KNN (primary)                                    
         'tvt_form_full': tvt_form_full, 'tvt_form_late': tvt_form_late,        
         'tvt_form_wls':  tvt_form_wls,  'tvt_form_egfdu': tvt_form_egfdu,      
         'tvt_form_buda': tvt_form_buda, 'ancc_knn': ancc_eval,                 
         'b_full': b_full, 'b_late': b_late, 'b_wls': b_wls,                    
         'b_early': b_early, 'b_mid': b_mid,                                    
         'knn_dist': knn_dist,                                                  
         # Per-formation TVT predictions (all 6 formations × 3 segments)        
         **tvt_form_feats,                                                      
         # Per-formation b_well offsets                                         
         **{f'bw_{fn}':       np.float32(form_b[fn]['full'])  for fn in         
 FORMATION_COLS},                                                               
         **{f'bww_{fn}':      np.float32(form_b[fn]['wls'])   for fn in         
 FORMATION_COLS},                                                               
         **{f'bw50_{fn}':     np.float32(form_b[fn]['late'])  for fn in         
 FORMATION_COLS},                                                               
         **{f'bw_early_{fn}': np.float32(form_b[fn]['early']) for fn in         
 FORMATION_COLS},                                                               
         **{f'bw_mid_{fn}':   np.float32(form_b[fn]['mid'])   for fn in         
 FORMATION_COLS},                                                               
         # RowKNN features                                                      
         'knn_row_ANCC':           row_ancc_eval,                               
         'knn_row_ANCC_std':       row_ancc_std_eval,                           
         'knn_row_dist':           row_dist_eval,                               
         'knn_row_b_well':         np.float32(b_well_row),                      
         'knn_row_tvt_pred_delta': knn_row_tvt_pred_delta,                      
         # DenseANCCImputer features                                            
         'dense_ancc':           dense_ancc_eval,                               
         'dense_ancc_std':       dense_std_eval,                                
         'dense_dist':           dense_dist_eval,                               
         'dense_b_well':         np.float32(b_well_dense),                      
         'dense_tvt_pred_delta': dense_tvt_pred_delta,                          
         # GR signal                                                            
         'GR_imputed': gr_eval, 'GR_rolling_mean_25': roll_s_mean[n_anchor:],   
         'GR_rolling_std_25': roll_s_std[n_anchor:], 'GR_rolling_mean_100':     
 roll_l_mean[n_anchor:],                                                        
         'GR_rolling_std_100': roll_l_std[n_anchor:],                           
         'GR_typewell_at_anchor': gr_tw_last_anchor, 'GR_diff': gr_diff,        
         'GR_norm_p5p95': gr_norm_p5p95,                                        
         # GR derivatives and envelope                                          
         'gr_d1': gr_d1, 'gr_d2': gr_d2, 'gr_env': gr_env,                      
         # xcorr                                                                
         'GR_xcorr_tvt': xcorr_tvt, 'GR_xcorr_conf': xcorr_conf,                
         'GR_xcorr_delta': xcorr_delta, 'GR_xcorr_tvt_smooth': xcorr_smooth,    
         'tw_gr_at_xcorr': tw_gr_at_xcorr, 'GR_xcorr_residual': xcorr_residual, 
         # NCC vs typewell                                                      
         'GR_ncc_tvt': ncc_tvt, 'GR_ncc_conf': ncc_conf, 'GR_ncc_delta':        
 ncc_delta,                                                                     
         # NCC vs anchor zone (new)                                             
         'GR_ncc_anchor_tvt':   ncc_anchor_tvt,                                 
         'GR_ncc_anchor_conf':  ncc_anchor_conf,                                
         'GR_ncc_anchor_delta': ncc_anchor_delta,                               
         # Softmax blend                                                        
         'ncc_form_blend': ncc_form_blend, 'ncc_form_delta': ncc_form_delta,    
         # Visible GR shift                                                     
         'visible_gr_shift_ft': gr_shift_ft, 'visible_gr_shift_corr':           
 gr_shift_corr,                                                                 
         'visible_gr_bias': gr_shift_bias,                                      
         # Viterbi beam (4 variants)                                            
         **beam_results,                                                        
         # Particle filter TVT (2 variants)                                     
         **pf_results,                                                          
         # Particle filter ANCC (new)                                           
         'pf_ancc_delta': pf_ancc_delta,                                        
         'pf_ancc_std':   pf_ancc_std.astype(np.float32),                       
         # Signal divergence                                                    
         'sig_std':    sig_std,                                                 
         'form_std':   form_std,                                                
         'pf_vs_spatial': pf_vs_spatial,                                        
         'sc_vs_beam':    sc_vs_beam,                                           
         # Typewell GR offset residuals anchored at NCC-anchor estimate         
         **tdsc_feats,                                                          
         # Fractional position in eval zone                                     
         'frac': frac.astype(np.float32),                                       
         'frac2': frac2.astype(np.float32),                                     
         'sqrt_frac': sqrt_frac.astype(np.float32),                             
         # GR lags/leads                                                        
         'GR_lag_5': gr_lag_5, 'GR_lag_10': gr_lag_10, 'GR_lag_25': gr_lag_25,  
         'GR_lag_50': gr_lag_50, 'GR_lag_100': gr_lag_100,                      
         'GR_lead_5': gr_lead_5, 'GR_lead_10': gr_lead_10, 'GR_lead_25':        
 gr_lead_25,                                                                    
         # Typewell GR offsets                                                  
         **tw_diff,                                                             
         # Geometry                                                             
         'MD': md_ev, 'MD_from_anchor': md_from_anchor,                         
         'X': x_ev, 'Y': y_ev, 'Z': z_ev,                                       
         'dZ_dMD': dz_dmd[n_anchor:], 'dX_dMD': dx_dmd[n_anchor:], 'dY_dMD':    
 dy_dmd[n_anchor:],                                                             
         # Trajectory kinematics                                                
         'incl_deg': incl_ev, 'azi_deg': azi_ev,                                
         'dls': dls_ev, 'build_rate': build_ev,                                 
         'cos_incl': cos_incl_ev, 'sin_incl': sin_incl_ev,                      
         # Apparent dip from anchor zone                                        
         'b_dip_full': np.float32(b_dip_full), 'b_dip_late':                    
 np.float32(b_dip_late),                                                        
         'b_dip_early': np.float32(b_dip_early), 'b_dip_slope':                 
 np.float32(b_dip_slope),                                                       
         'tvt_dip_full': tvt_dip_full_arr, 'tvt_dip_late': tvt_dip_late_arr,    
         'azi_delta': azi_delta_ev, 'apparent_dip_dir': apparent_dip_dir,       
         'plane_dip_x': np.float32(plane_dip_x), 'plane_dip_y':                 
 np.float32(plane_dip_y),                                                       
                                                                                
         # TVT extrapolation context                                            
         'last_anchor_tvt': last_tvt, 'tvt_rate': tvt_rate, 'tvt_extrap':       
 tvt_extrap,                                                                    
         # Well context                                                         
         'anchor_gr_mean': anchor_gr_mean, 'anchor_gr_std': anchor_gr_std,      
         'anchor_tvt_mean': anchor_tvt_mean, 'anchor_tvt_range':                
 anchor_tvt_range,                                                              
         'n_anchor_rows': n_anchor_rows, 'tw_gr_anchor_mean': tw_gr_anchor_mean 
         # Prefix features                                                      
         'prefix_tvt_step20': prefix_tvt_step20, 'prefix_tvt_step100':          
 prefix_tvt_step100,                                                            
         'prefix_tvt_md_slope100': prefix_tvt_md_slope100,                      
     })                                                                         
                                                                                
     # ── v4 short-window slope features (inline — avoids re-reading the well   
 file) ──                                                                       
     def _slope(y, x, n):                                                       
         y, x = y[-n:], x[-n:]                                                  
         if len(y) < 2: return 0.0                                              
         cx = x - x.mean()                                                      
         d  = float(np.dot(cx, cx))                                             
         return float(np.dot(cx, y - y.mean()) / d) if d > 0 else 0.0           
                                                                                
     s10 = _slope(anchor_tvt_vals, anchor_md_vals, 10)                          
     s25 = _slope(anchor_tvt_vals, anchor_md_vals, 25)                          
     s50 = _slope(anchor_tvt_vals, anchor_md_vals, 50)                          
     out['prefix_tvt_md_slope50'] = np.float32(s50)                             
     out['anchor_slope_k10']      = np.float32(s10)                             
     out['anchor_slope_k25']      = np.float32(s25)                             
     out['tvt_extrap_k50']  = (last_tvt + s50 *                                 
 md_from_anchor).astype(np.float32)                                             
     out['tvt_extrap_k10']  = (last_tvt + s10 *                                 
 md_from_anchor).astype(np.float32)                                             
     out['tvt_extrap_k25']  = (last_tvt + s25 *                                 
 md_from_anchor).astype(np.float32)                                             
     out['slope_accel_10_50']  = np.float32(s10 - s50)                          
     out['slope_accel_25_100'] = np.float32(s25 - prefix_tvt_md_slope100)       
                                                                                
                                                                                
     if 'TVT' in hw.columns:                                                    
         out['TVT'] = hw.loc[~anchor_mask, 'TVT'].values                        
                                                                                
     return out                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def add_v4_features(df: pd.DataFrame) -> pd.DataFrame:                         
     """Add v4 Phase-1 (estimator divergence) and Phase-3 (DWT GR) features.""" 
     # Phase 1: pairwise estimator divergence                                   
     form_drift   = df['tvt_form_full'] - df['last_anchor_tvt']                 
     ncc_drift    = df['GR_ncc_delta']                                          
     beam_drift   = df['beam_med2_delta']                                       
     pf_drift     = df['pf_ancc_delta']                                         
     extrap_drift = df['tvt_extrap']    - df['last_anchor_tvt']                 
                                                                                
     df['form_vs_ncc']  = (form_drift - ncc_drift).astype('float32')            
     df['form_vs_pf']   = (form_drift - pf_drift).astype('float32')             
     df['form_vs_beam'] = (form_drift - beam_drift).astype('float32')           
     df['ncc_vs_pf']    = (ncc_drift  - pf_drift).astype('float32')             
     df['ncc_vs_beam']  = (ncc_drift  - beam_drift).astype('float32')           
     df['beam_vs_pf']   = (beam_drift - pf_drift).astype('float32')             
                                                                                
     all_drifts = np.column_stack([                                             
         form_drift.values, ncc_drift.values, beam_drift.values,                
         pf_drift.values, extrap_drift.values,                                  
     ])                                                                         
     df['estimator_drift_range'] = (all_drifts.max(axis=1) -                    
 all_drifts.min(axis=1)).astype('float32')                                      
     df['estimator_drift_max']   = all_drifts.max(axis=1).astype('float32')     
     df['estimator_drift_min']   = all_drifts.min(axis=1).astype('float32')     
     df['extrap_k50_vs_extrap200'] = (df['tvt_extrap_k50'] -                    
 df['tvt_extrap']).astype('float32')                                            
     df['form_vs_extrap']          = (df['tvt_form_full']  -                    
 df['tvt_extrap']).astype('float32')                                            
                                                                                
     # Phase 3: per-well DWT GR features                                        
     def _compute_dwt(gr, wavelet='db4', n_levels=5):                           
         n    = len(gr)                                                         
         gr_c = np.where(np.isfinite(gr), gr, np.nanmean(gr) if                 
 np.isfinite(gr).any() else 0.0)                                                
         try:                                                                   
             coeffs = pywt.wavedec(gr_c, wavelet, mode='periodization',         
 level=n_levels)                                                                
             recon  = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]      
             approx = pywt.waverec(recon, wavelet, mode='periodization')[:n]    
             det3   = [np.zeros_like(c) for c in coeffs]; det3[3] = coeffs[3]   
             det3_s = pywt.waverec(det3, wavelet, mode='periodization')[:n]     
             detail = (pd.Series(det3_s**2).rolling(16, center=True,            
 min_periods=1).mean().values**0.5)                                             
         except Exception:                                                      
             approx, detail = gr_c, np.zeros(n)                                 
         return approx.astype(np.float32), detail.astype(np.float32)            
                                                                                
     approx_parts, detail_parts, idx_parts = [], [], []                         
     for wid, grp in df.groupby('well_id', sort=False):                         
         a, d = _compute_dwt(grp['GR_imputed'].values)                          
         approx_parts.append(a); detail_parts.append(d);                        
 idx_parts.append(grp.index)                                                    
                                                                                
     all_idx = np.concatenate([i.values for i in idx_parts])                    
     df.loc[all_idx, 'gr_dwt_approx5']       = np.concatenate(approx_parts)     
     df.loc[all_idx, 'gr_dwt_detail_energy'] = np.concatenate(detail_parts)     
     df['gr_dwt_residual'] = (df['GR_imputed'] -                                
 df['gr_dwt_approx5']).astype('float32')                                        
                                                                                
     print(f'add_v4_features: added 21 features to {len(df):,} rows             
 ({df["well_id"].nunique()} wells)')                                            
     return df                                                                  
                                                                                
 print('add_v4_features defined')                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 test_ids = get_test_well_ids()                                                 
 test_frames, errors = [], []                                                   
                                                                                
 t0 = time.time()                                                               
 for well_id in tqdm(test_ids, desc='Building test features'):                  
     try:                                                                       
         feat = build_features(well_id, split='test', loo=False)                
         if len(feat) > 0:                                                      
             test_frames.append(feat)                                           
     except Exception as e:                                                     
         errors.append((well_id, str(e)))                                       
         print(f'  ERROR {well_id}: {e}')                                       
                                                                                
 test_df = pd.concat(test_frames, ignore_index=True)                            
 test_df = add_v4_features(test_df)                                             
 X_test  = test_df[FEATURE_COLS].values.astype(np.float32)                      
 print(f'Test features: {test_df.shape}  |  {test_df["well_id"].nunique()} well 
 ({time.time()-t0:.0f}s)')                                                      
 print(f'NaN in X_test: {np.isnan(X_test).sum()}')                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 xgb_folds, cb_folds, hgb_folds = [], [], []                                    
 X_test_sk = np.nan_to_num(X_test, nan=0.0)                                     
                                                                                
 for fold_i in range(N_FOLDS):                                                  
     # XGB                                                                      
     m = xgb.XGBRegressor()                                                     
     m.load_model(str(MODEL_DIR / f'xgb_fold{fold_i}.json'))                    
     xgb_folds.append(m.predict(X_test).astype(np.float32))                     
                                                                                
     # CB                                                                       
     m = CatBoostRegressor()                                                    
     m.load_model(str(MODEL_DIR / f'cb_fold{fold_i}.cbm'))                      
     cb_folds.append(m.predict(X_test).astype(np.float32))                      
                                                                                
     # HGB                                                                      
     m = joblib.load(MODEL_DIR / f'hgb_fold{fold_i}.joblib')                    
     hgb_folds.append(m.predict(X_test_sk).astype(np.float32))                  
                                                                                
     print(f'  Fold {fold_i} loaded and predicted')                             
                                                                                
 xgb_test_drift = np.mean(xgb_folds, axis=0)                                    
 cb_test_drift  = np.mean(cb_folds,  axis=0)                                    
 hgb_test_drift = np.mean(hgb_folds, axis=0)                                    
 print(f'XGB drift range: {xgb_test_drift.min():.2f}–{xgb_test_drift.max():.2f} 
 ft')                                                                           
 print(f'CB  drift range: {cb_test_drift.min():.2f}–{cb_test_drift.max():.2f}   
 ft')                                                                           
 print(f'HGB drift range: {hgb_test_drift.min():.2f}–{hgb_test_drift.max():.2f} 
 ft')                                                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # ── Weighted blend ────────────────────────────────────────────────────────── 
 drift_preds = {'xgb': xgb_test_drift, 'cb': cb_test_drift, 'hgb':              
 hgb_test_drift}                                                                
 blend_drift = sum(BLEND_WEIGHTS.get(k, 0.0) * v for k, v in                    
 drift_preds.items()).astype(np.float32)                                        
 print('Blend weights:', {k: f'{v:.4f}' for k, v in BLEND_WEIGHTS.items()})     
 print(f'Blended drift range: {blend_drift.min():.2f}–{blend_drift.max():.2f}   
 ft')                                                                           
                                                                                
 # ── Savitzky-Golay smoothing per well (domain prior: w=17, poly=3) ────────── 
 def sg_smooth_wells(drift_arr, df, sg_w=SG_W, sg_p=SG_P):                      
     out = drift_arr.copy()                                                     
     for _, g in df.groupby('well_id', sort=False):                             
         idx   = g.sort_values('row_index').index                               
         chunk = drift_arr[idx]                                                 
         if len(chunk) >= sg_w:                                                 
             out[idx] = savgol_filter(chunk, sg_w, sg_p)                        
     return out                                                                 
                                                                                
 final_drift = sg_smooth_wells(blend_drift, test_df)                            
 final_tvt   = (final_drift +                                                   
 test_df['last_anchor_tvt'].values.astype(np.float32))                          
 print(f'Final TVT range: {final_tvt.min():.1f}–{final_tvt.max():.1f} ft')      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 submission = test_df[['well_id', 'row_index']].copy()                          
 submission['id']  = submission['well_id'].astype(str) + '_' +                  
 submission['row_index'].astype(str)                                            
 submission['tvt'] = final_tvt.astype(np.float32)                               
 submission = submission[['id', 'tvt']].sort_values('id').reset_index(drop=True 
                                                                                
 # Validate against sample submission format                                    
 try:                                                                           
     template =                                                                 
 pd.read_csv('rogii-wellbore-geology-prediction/sample_submission.csv')         
     assert submission.shape[0] == template.shape[0], \                         
         f'Row count mismatch: got {len(submission)}, expected {len(template)}' 
 except FileNotFoundError:                                                      
     print('sample_submission.csv not found — skipping shape check')            
                                                                                
 submission.to_csv('submission.csv', index=False)                               
 print(f'Saved submission.csv  ({len(submission)} rows)')                       
 print(submission.head())                                                       
 print(f'\nTVT stats:  mean={submission["tvt"].mean():.1f}  '                   
       f'std={submission["tvt"].std():.1f}  '                                   
       f'min={submission["tvt"].min():.1f}  '                                   
       f'max={submission["tvt"].max():.1f}')                                    
                                                                                
