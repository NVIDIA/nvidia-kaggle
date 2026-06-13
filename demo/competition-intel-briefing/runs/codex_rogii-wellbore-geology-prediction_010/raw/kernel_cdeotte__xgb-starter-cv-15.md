Reading kernel: cdeotte/xgb-starter-cv-15
Cells: 21 (13 code, 8 markdown) | 17613 chars

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃             XGBoost Starter - ROGII Wellbore Geology Prediction              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

This notebook is a compact XGBoost starter for the ROGII - Wellbore Geology     
Prediction competition. This version 3 is the same basic flow as version 2, but 
the residual baseline is the final known TVT_input value for each well.         

The task is to predict tvt (True Vertical Thickness) for the hidden/evaluation  
interval of each horizontal well. The visible TVT_input column gives the known  
interpreted TVT before the prediction start; after that point it becomes missing
and we must infer the continuation.                                             

This starter:                                                                   

 • Uses only fields that are available at test time.                            
 • Builds per-well context features from the known TVT_input segment.           
 • Adds simple Typewell gamma-ray correlation features inspired by the          
   competition task deck.                                                       
 • Uses well-grouped K-fold validation so rows from the same well do not leak   
   across folds.                                                                
 • Trains an XGBoost residual model over the strong last-known-TVT_input        
   baseline.                                                                    
 • Saves submission.csv.                                                        

────────────────────────────────────────────────────────────────────────────────

                          1. Imports and Configuration                          

Set FAST_DEBUG = True for a quick smoke test. The default is full-data 5-fold   
validation.                                                                     

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from pathlib import Path                                                       
 from collections import defaultdict                                            
 import os                                                                      
 import gc                                                                      
 import warnings                                                                
                                                                                
 import numpy as np                                                             
 import pandas as pd                                                            
 import matplotlib.pyplot as plt                                                
                                                                                
 from sklearn.model_selection import GroupKFold                                 
 from xgboost import XGBRegressor                                               
                                                                                
 warnings.filterwarnings("ignore")                                              
 pd.set_option("display.max_columns", 200)                                      
                                                                                
 RANDOM_STATE = 42                                                              
 FAST_DEBUG = False                                                             
                                                                                
 N_FOLDS = 3 if FAST_DEBUG else 5                                               
 MAX_TRAIN_WELLS = 60 if FAST_DEBUG else None                                   
                                                                                
 XGB_PARAMS = {                                                                 
     "n_estimators": 80 if FAST_DEBUG else 450,                                 
     "learning_rate": 0.06 if FAST_DEBUG else 0.035,                            
     "max_depth": 5,                                                            
     "min_child_weight": 20,                                                    
     "subsample": 0.85,                                                         
     "colsample_bytree": 0.85,                                                  
     "reg_lambda": 4.0,                                                         
     "reg_alpha": 0.05,                                                         
     "objective": "reg:squarederror",                                           
     "eval_metric": "rmse",                                                     
     "tree_method": "hist",                                                     
     "max_bin": 256,                                                            
     "random_state": RANDOM_STATE,                                              
     "n_jobs": -1,                                                              
     "device": "cuda",                                                          
 }                                                                              
                                                                                
 print("FAST_DEBUG:", FAST_DEBUG)                                               
 print("N_FOLDS:", N_FOLDS)                                                     
 print("XGB_PARAMS:", XGB_PARAMS)                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                                 2. Locate Data                                 

The helper below works on Kaggle and with a local extracted copy of the         
competition data.                                                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def find_data_root():                                                          
     candidates = [                                                             
         Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),  
         Path.cwd(),                                                            
     ]                                                                          
     candidates.extend(Path.cwd().parents)                                      
                                                                                
     for root in candidates:                                                    
         if (root / "train").is_dir() and (root /                               
 "sample_submission.csv").is_file():                                            
             return root.resolve()                                              
     raise FileNotFoundError("Could not find train/ and sample_submission.csv") 
                                                                                
 DATA_ROOT = find_data_root()                                                   
 TRAIN_DIR = DATA_ROOT / "train"                                                
 TEST_DIR = DATA_ROOT / "test"                                                  
 SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"                          
                                                                                
 print("DATA_ROOT:", DATA_ROOT)                                                 
 print("TRAIN_DIR:", TRAIN_DIR)                                                 
 print("TEST_DIR:", TEST_DIR)                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def well_id_from_path(path):                                                   
     return Path(path).name.split("__", 1)[0]                                   
                                                                                
 train_horizontal_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))      
 test_horizontal_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))        
                                                                                
 if MAX_TRAIN_WELLS is not None:                                                
     train_horizontal_paths = train_horizontal_paths[:MAX_TRAIN_WELLS]          
                                                                                
 print(f"Training horizontal wells used: {len(train_horizontal_paths)}")        
 print(f"Visible test horizontal wells:  {len(test_horizontal_paths)}")         
                                                                                
 sample_sub = pd.read_csv(SAMPLE_SUB_PATH)                                      
 sample_sub["well_id"] = sample_sub["id"].str.rsplit("_", n=1).str[0]           
 sample_sub["row_index"] = sample_sub["id"].str.rsplit("_",                     
 n=1).str[1].astype(int)                                                        
                                                                                
 display(sample_sub.head())                                                     
 display(sample_sub.groupby("well_id").size().rename("rows_to_predict").reset_i 
 ex().head())                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────

                             3. Feature Engineering                             

The model predicts the residual from the strongest last-known-TVT baseline we   
found locally:                                                                  

baseline_tvt = last_known_tvt                                                   

A zero residual therefore equals the last-known-TVT baseline, which scored about
15.9 RMSE locally. XGBoost only needs to learn corrections away from that flat  
baseline.                                                                       

The slope-continuation values are still kept as features:                       

 • baseline_tvt_all_slope                                                       
 • baseline_tvt_recent_slope                                                    
 • slope_tvt_md_all                                                             
 • slope_tvt_md_recent                                                          

Training uses only rows where TVT_input is missing, because those match the rows
we must predict in test.                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def rmse(y_true, y_pred):                                                      
     y_true = np.asarray(y_true, dtype=np.float64)                              
     y_pred = np.asarray(y_pred, dtype=np.float64)                              
     return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))                     
                                                                                
                                                                                
 def robust_slope(x, y, default=0.0):                                           
     x = np.asarray(x, dtype=np.float64)                                        
     y = np.asarray(y, dtype=np.float64)                                        
     mask = np.isfinite(x) & np.isfinite(y)                                     
     if mask.sum() < 2:                                                         
         return default                                                         
     x = x[mask]                                                                
     y = y[mask]                                                                
     if np.nanstd(x) < 1e-6:                                                    
         return default                                                         
     return float(np.polyfit(x, y, 1)[0])                                       
                                                                                
                                                                                
 def safe_interp(x, xp, fp):                                                    
     xp = np.asarray(xp, dtype=np.float64)                                      
     fp = np.asarray(fp, dtype=np.float64)                                      
     mask = np.isfinite(xp) & np.isfinite(fp)                                   
     if mask.sum() < 2:                                                         
         return np.full_like(np.asarray(x, dtype=np.float64), np.nan,           
 dtype=np.float64)                                                              
     order = np.argsort(xp[mask])                                               
     xp = xp[mask][order]                                                       
     fp = fp[mask][order]                                                       
     return np.interp(np.asarray(x, dtype=np.float64), xp, fp, left=np.nan,     
 right=np.nan)                                                                  
                                                                                
                                                                                
 def make_test_row_map(submission):                                             
     row_map = defaultdict(list)                                                
     for well_id, row_idx in zip(submission["well_id"], submission["row_index"] 
         row_map[well_id].append(int(row_idx))                                  
     return {k: np.array(v, dtype=np.int64) for k, v in row_map.items()}        
                                                                                
                                                                                
 def load_typewell(well_id, split):                                             
     base = TRAIN_DIR if split == "train" else TEST_DIR                         
     path = base / f"{well_id}__typewell.csv"                                   
     if path.exists():                                                          
         return pd.read_csv(path)                                               
     return pd.DataFrame({"TVT": [], "GR": []})                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def build_features_for_well(horizontal_path, split="train", test_row_map=None) 
     well_id = well_id_from_path(horizontal_path)                               
     h = pd.read_csv(horizontal_path)                                           
     tw = load_typewell(well_id, split)                                         
                                                                                
     h["row_index"] = np.arange(len(h), dtype=np.int64)                         
                                                                                
     # Target/evaluation rows.                                                  
     if split == "train":                                                       
         row_mask = h["TVT_input"].isna() & h["TVT"].notna()                    
         selected_idx = h.index[row_mask].to_numpy(dtype=np.int64)              
     else:                                                                      
         if test_row_map is not None and well_id in test_row_map:               
             selected_idx = test_row_map[well_id]                               
         else:                                                                  
             selected_idx =                                                     
 h.index[h["TVT_input"].isna()].to_numpy(dtype=np.int64)                        
                                                                                
     if len(selected_idx) == 0:                                                 
         return pd.DataFrame()                                                  
                                                                                
     known = h[h["TVT_input"].notna()].copy()                                   
     if len(known) == 0:                                                        
         # Very defensive fallback. This should not happen in the competition   
 data.                                                                          
         known = h.head(1).copy()                                               
         known["TVT_input"] = np.nan                                            
                                                                                
     first_missing = h.index[h["TVT_input"].isna()]                             
     ps_idx = int(first_missing[0]) if len(first_missing) else int(len(h))      
     last_known = known.iloc[-1]                                                
                                                                                
     # Slopes from the known segment.                                           
     slope_all = robust_slope(known["MD"], known["TVT_input"])                  
     recent = known.tail(min(200, len(known)))                                  
     slope_recent = robust_slope(recent["MD"], recent["TVT_input"],             
 default=slope_all)                                                             
     slope_z_recent = robust_slope(recent["Z"], recent["TVT_input"])            
                                                                                
     # Current rows.                                                            
     cur = h.loc[selected_idx, ["MD", "X", "Y", "Z", "GR", "row_index"]].copy() 
     cur["well_id"] = well_id                                                   
     cur["id"] = cur["well_id"] + "_" + cur["row_index"].astype(str)            
                                                                                
     # Per-well known-context constants.                                        
     ps_md = float(last_known.get("MD", np.nan))                                
     ps_x = float(last_known.get("X", np.nan))                                  
     ps_y = float(last_known.get("Y", np.nan))                                  
     ps_z = float(last_known.get("Z", np.nan))                                  
     ps_gr = float(last_known.get("GR", np.nan))                                
     last_known_tvt = float(last_known.get("TVT_input", np.nan))                
                                                                                
     cur["n_rows"] = len(h)                                                     
     cur["prediction_start_index"] = ps_idx                                     
     cur["last_known_tvt"] = last_known_tvt                                     
     cur["known_tvt_min"] = known["TVT_input"].min()                            
     cur["known_tvt_max"] = known["TVT_input"].max()                            
     cur["known_tvt_range"] = known["TVT_input"].max() - known["TVT_input"].min 
     cur["known_tvt_mean"] = known["TVT_input"].mean()                          
     cur["known_tvt_std"] = known["TVT_input"].std()                            
     cur["known_gr_mean"] = known["GR"].mean() if "GR" in known else np.nan     
     cur["known_gr_std"] = known["GR"].std() if "GR" in known else np.nan       
     cur["known_gr_min"] = known["GR"].min() if "GR" in known else np.nan       
     cur["known_gr_max"] = known["GR"].max() if "GR" in known else np.nan       
     cur["last_known_gr"] = ps_gr                                               
     cur["slope_tvt_md_all"] = slope_all                                        
     cur["slope_tvt_md_recent"] = slope_recent                                  
     cur["slope_tvt_z_recent"] = slope_z_recent                                 
                                                                                
     # Relative position after prediction start.                                
     cur["row_from_ps"] = cur["row_index"] - ps_idx                             
     cur["row_frac"] = cur["row_index"] / max(len(h) - 1, 1)                    
     cur["md_from_ps"] = cur["MD"] - ps_md                                      
     cur["x_from_ps"] = cur["X"] - ps_x                                         
     cur["y_from_ps"] = cur["Y"] - ps_y                                         
     cur["z_from_ps"] = cur["Z"] - ps_z                                         
     cur["xy_dist_from_ps"] = np.sqrt(cur["x_from_ps"] ** 2 + cur["y_from_ps"]  
 2)                                                                             
     cur["xyz_dist_from_ps"] = np.sqrt(cur["xy_dist_from_ps"] ** 2 +            
 cur["z_from_ps"] ** 2)                                                         
                                                                                
     # Baselines. XGB2 uses the flat last-known-TVT baseline for residual       
 learning.                                                                      
     # The slope continuation baselines are retained only as model features.    
     cur["baseline_tvt_all_slope"] = last_known_tvt + slope_all *               
 cur["md_from_ps"]                                                              
     cur["baseline_tvt_recent_slope"] = last_known_tvt + slope_recent *         
 cur["md_from_ps"]                                                              
     cur["baseline_tvt"] = last_known_tvt                                       
                                                                                
     # Horizontal GR features. Interpolate missing GR for rolling/context       
 features but keep a missing flag.                                              
     cur["gr_missing"] = cur["GR"].isna().astype(np.int8)                       
     gr_filled = h["GR"].astype(float).interpolate(limit_direction="both")      
     for window in [11, 51, 151]:                                               
         roll = gr_filled.rolling(window=window, center=True, min_periods=max(2 
 window // 5))                                                                  
         cur[f"gr_roll_mean_{window}"] =                                        
 roll.mean().iloc[selected_idx].to_numpy()                                      
         cur[f"gr_roll_std_{window}"] = roll.std().iloc[selected_idx].to_numpy( 
     cur["gr_diff_1"] = gr_filled.diff(1).iloc[selected_idx].to_numpy()         
     cur["gr_diff_10"] = gr_filled.diff(10).iloc[selected_idx].to_numpy()       
     cur["gr_minus_last_known"] = cur["GR"] - ps_gr                             
                                                                                
     # Typewell features.                                                       
     if {"TVT", "GR"}.issubset(tw.columns) and len(tw) > 1:                     
         tw_tvt = tw["TVT"].astype(float)                                       
         tw_gr = tw["GR"].astype(float)                                         
         cur["typewell_tvt_min"] = tw_tvt.min()                                 
         cur["typewell_tvt_max"] = tw_tvt.max()                                 
         cur["typewell_tvt_range"] = tw_tvt.max() - tw_tvt.min()                
         cur["typewell_gr_mean"] = tw_gr.mean()                                 
         cur["typewell_gr_std"] = tw_gr.std()                                   
         cur["typewell_gr_min"] = tw_gr.min()                                   
         cur["typewell_gr_max"] = tw_gr.max()                                   
         cur["tw_gr_at_last_known_tvt"] = safe_interp(np.array([last_known_tvt] 
 tw_tvt, tw_gr)[0]                                                              
         cur["tw_gr_at_baseline_tvt"] =                                         
 safe_interp(cur["baseline_tvt"].to_numpy(), tw_tvt, tw_gr)                     
         cur["tw_gr_at_baseline_all_slope"] =                                   
 safe_interp(cur["baseline_tvt_all_slope"].to_numpy(), tw_tvt, tw_gr)           
         cur["gr_minus_tw_baseline"] = cur["GR"] - cur["tw_gr_at_baseline_tvt"] 
         cur["gr_minus_tw_last_known"] = cur["GR"] -                            
 cur["tw_gr_at_last_known_tvt"]                                                 
     else:                                                                      
         for col in [                                                           
             "typewell_tvt_min", "typewell_tvt_max", "typewell_tvt_range",      
             "typewell_gr_mean", "typewell_gr_std", "typewell_gr_min",          
 "typewell_gr_max",                                                             
             "tw_gr_at_last_known_tvt", "tw_gr_at_baseline_tvt",                
 "tw_gr_at_baseline_all_slope",                                                 
             "gr_minus_tw_baseline", "gr_minus_tw_last_known",                  
         ]:                                                                     
             cur[col] = np.nan                                                  
                                                                                
     if split == "train":                                                       
         cur["target_tvt"] = h.loc[selected_idx, "TVT"].to_numpy()              
         cur["target_residual"] = cur["target_tvt"] - cur["baseline_tvt"]       
                                                                                
     # Downcast numeric columns to reduce memory.                               
     for col in cur.columns:                                                    
         if col not in ["well_id", "id"]:                                       
             cur[col] = pd.to_numeric(cur[col], downcast="float")               
                                                                                
     return cur.reset_index(drop=True)                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────

                            4. Build Training Matrix                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_parts = []                                                               
 for i, path in enumerate(train_horizontal_paths, start=1):                     
     train_parts.append(build_features_for_well(path, split="train"))           
     if i % 100 == 0:                                                           
         print(f"Processed {i}/{len(train_horizontal_paths)} train wells")      
                                                                                
 train_df = pd.concat(train_parts, ignore_index=True)                           
 del train_parts                                                                
 gc.collect()                                                                   
                                                                                
 print("train_df shape:", train_df.shape)                                       
 display(train_df.head())                                                       
 print("Target rows:", len(train_df))                                           
 print("Unique wells:", train_df["well_id"].nunique())                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 exclude_cols = {"well_id", "id", "target_tvt", "target_residual"}              
 feature_cols = [c for c in train_df.columns if c not in exclude_cols]          
                                                                                
 X = train_df[feature_cols].astype(np.float32)                                  
 y_resid = train_df["target_residual"].astype(np.float32)                       
 y_true = train_df["target_tvt"].astype(np.float32)                             
 baseline = train_df["baseline_tvt"].astype(np.float32)                         
 groups = train_df["well_id"].values                                            
                                                                                
 print(f"Number of features: {len(feature_cols)}")                              
 print(feature_cols)                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────

                       5. Well-Grouped K-Fold Validation                        

This validation splits by well ID. That is stricter and more realistic than     
random row K-fold because rows from the same well are highly correlated.        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 gkf = GroupKFold(n_splits=N_FOLDS)                                             
 oof = np.zeros(len(train_df), dtype=np.float32)                                
 fold_rows = []                                                                 
 models = []                                                                    
                                                                                
 for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y_resid, groups),       
 start=1):                                                                      
     print(f"\n===== Fold {fold}/{N_FOLDS} =====")                              
     print(f"Train rows: {len(trn_idx):,} | Valid rows: {len(val_idx):,}")      
     print(f"Train wells: {len(np.unique(groups[trn_idx])):,} | Valid wells:    
 {len(np.unique(groups[val_idx])):,}")                                          
                                                                                
     model = XGBRegressor(**XGB_PARAMS)                                         
     model.fit(                                                                 
         X.iloc[trn_idx],                                                       
         y_resid.iloc[trn_idx],                                                 
         eval_set=[(X.iloc[val_idx], y_resid.iloc[val_idx])],                   
         verbose=100,                                                           
     )                                                                          
                                                                                
     val_resid_pred = model.predict(X.iloc[val_idx]).astype(np.float32)         
     val_pred = baseline.iloc[val_idx].to_numpy() + val_resid_pred              
     oof[val_idx] = val_pred                                                    
                                                                                
     fold_rmse = rmse(y_true.iloc[val_idx], val_pred)                           
     baseline_rmse = rmse(y_true.iloc[val_idx], baseline.iloc[val_idx])         
                                                                                
     fold_rows.append({                                                         
         "fold": fold,                                                          
         "valid_rows": len(val_idx),                                            
         "valid_wells": len(np.unique(groups[val_idx])),                        
         "baseline_rmse": baseline_rmse,                                        
         "xgb_rmse": fold_rmse,                                                 
         "improvement": baseline_rmse - fold_rmse,                              
     })                                                                         
     models.append(model)                                                       
                                                                                
     print(f"Fold {fold} baseline RMSE: {baseline_rmse:.5f}")                   
     print(f"Fold {fold} XGB RMSE:      {fold_rmse:.5f}")                       
                                                                                
 cv = pd.DataFrame(fold_rows)                                                   
 display(cv)                                                                    
 print(f"OOF baseline RMSE: {rmse(y_true, baseline):.5f}")                      
 print(f"OOF XGB RMSE:      {rmse(y_true, oof):.5f}")                           
 print(f"Mean fold RMSE:    {cv['xgb_rmse'].mean():.5f} +/-                     
 {cv['xgb_rmse'].std():.5f}")                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)     
                                                                                
 axes[0].scatter(y_true, oof, s=1, alpha=0.08)                                  
 lo = float(np.nanpercentile(y_true, 1))                                        
 hi = float(np.nanpercentile(y_true, 99))                                       
 axes[0].plot([lo, hi], [lo, hi], color="crimson", lw=1)                        
 axes[0].set_xlim(lo, hi)                                                       
 axes[0].set_ylim(lo, hi)                                                       
 axes[0].set_title("OOF predictions vs target")                                 
 axes[0].set_xlabel("Actual TVT")                                               
 axes[0].set_ylabel("Predicted TVT")                                            
                                                                                
 resid = y_true.to_numpy() - oof                                                
 axes[1].hist(np.clip(resid, -100, 100), bins=80, color="steelblue",            
 edgecolor="white")                                                             
 axes[1].set_title("OOF residuals clipped to [-100, 100] ft")                   
 axes[1].set_xlabel("Actual - predicted TVT")                                   
 axes[1].set_ylabel("Rows")                                                     
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Average feature importance over folds.                                       
 importance = pd.DataFrame({"feature": feature_cols})                           
 for fold, model in enumerate(models, start=1):                                 
     importance[f"fold_{fold}"] = model.feature_importances_                    
 importance["mean_importance"] = importance[[c for c in importance.columns if   
 c.startswith("fold_")]].mean(axis=1)                                           
 importance = importance.sort_values("mean_importance", ascending=False)        
                                                                                
 display(importance.head(25))                                                   
                                                                                
 fig, ax = plt.subplots(figsize=(8, 8))                                         
 top = importance.head(25).sort_values("mean_importance")                       
 ax.barh(top["feature"], top["mean_importance"], color="seagreen")              
 ax.set_title("Mean XGBoost feature importance")                                
 ax.set_xlabel("Importance")                                                    
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────

                        6. Build Test Matrix and Predict                        

The fold models are averaged for the final submission. This avoids training an  
additional full-data model and keeps the starter reasonably fast.               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 test_parts = []                                                                
 for i, path in enumerate(test_horizontal_paths, start=1):                      
     test_parts.append(build_features_for_well(path, split="test",              
 test_row_map=None))                                                            
     if i % 50 == 0:                                                            
         print(f"Processed {i}/{len(test_horizontal_paths)} test wells")        
                                                                                
 test_df = pd.concat(test_parts, ignore_index=True)                             
 del test_parts                                                                 
 gc.collect()                                                                   
                                                                                
 print("test_df shape:", test_df.shape)                                         
 display(test_df.head())                                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X_test = test_df[feature_cols].astype(np.float32)                              
 test_baseline = test_df["baseline_tvt"].to_numpy(dtype=np.float32)             
                                                                                
 test_resid_pred = np.zeros(len(test_df), dtype=np.float32)                     
 for fold, model in enumerate(models, start=1):                                 
     pred = model.predict(X_test).astype(np.float32)                            
     test_resid_pred += pred / len(models)                                      
     print(f"Predicted test with fold model {fold}")                            
                                                                                
 test_df["tvt"] = test_baseline + test_resid_pred                               
                                                                                
 display(test_df[["id", "baseline_tvt", "tvt"]].head())                         
 print(test_df["tvt"].describe())                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                               7. Save Submission                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 submission = test_df[["id", "tvt"]].copy()                                     
 submission["tvt"] = pd.to_numeric(submission["tvt"], errors="coerce")          
 submission.to_csv("submission.csv", index=False)                               
 print("Saved submission.csv")                                                  
 print(submission.shape)                                                        
 display(submission.head())                                                     
                                                                                
