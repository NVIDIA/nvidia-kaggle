╭─ A geophysicist's take: domain priors + Q-3D tortuosity (public notebook + r─╮
│ Author: Matteo Niccoli | Votes: 16 | Comments: 0 | Created: 2026-05-21       │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/702131                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                  Kaggle discussion post - ROGII competition                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
Sharing a notebook and a public repo that approach this competition from a      
geological reasoning angle rather than pure ML optimization.                    

Notebook:                                                                       
https://www.kaggle.com/code/mycarta/rogii-wellbore-geology-prediction-toolkit   

GitHub repo:  https://github.com/mycarta/rogii-geosteering-toolkit              

Single LightGBM, no particle filters or stacking. Mid-pack on the leaderboard,  
but the findings may be useful to others working on this problem:               

 • Within-well TVT-Z decoupling. The global TVT-vs-Z correlation is r = -0.96,  
   but within a single lateral it's essentially zero (mean slope +0.057). The   
   global signal is cross-well structural elevation dominated by build-section  
   geometry. Features based on the global relationship don't work until you     
   account for this.                                                            
 • Q-3D tortuosity (Jing et al. 2022) was the most useful domain feature in the 
   ablation (-0.107 RMSE). High tortuosity = active steering = formation        
   deviating from plan.                                                         
 • Signed drilling azimuth matters. Opposite directions along the same line see 
   the formation in opposite sequence. The updip/downdip distinction is real and
   the model uses it via sin/cos azimuth encoding paired with dZ/dMD.           
 • Well-level AEON features (Catch22 + ClaSP) made the model worse (+0.476      
   RMSE). Well-level features under GroupKFold overfit on cross-well noise.     
   Documented as a negative result in the ablation table.                       
 • Considered Verde's BlockKFold for spatial CV, rejected it. Validation wells  
   are spatially interleaved with training (interpolation, not extrapolation),  
   so spatial blocking is more pessimistic than the actual test condition. Used 
   StratifiedGroupKFold stratified by signed azimuth, median TVT, and spatial   
   location instead.                                                            

The notebook includes a cumulative ablation table, Phase 1 spatial recon figures
(well center map, azimuth rose, lateral slope distribution), a                  
MASS-equivalent-to-NCC pedagogical demo, and a Q-3D tortuosity visualization.   

The GitHub repo has the reusable toolkit modules (wellbore tortuosity from XYZ  
trajectories, log despiking, sliding distance correlation) and two methodology  
write-ups on the TVT-Z decoupling finding and the AEON evaluation.              

Feedback welcome, especially from anyone with geosteering or horizontal drilling
experience. What domain features would you add?                                 

No comments.
