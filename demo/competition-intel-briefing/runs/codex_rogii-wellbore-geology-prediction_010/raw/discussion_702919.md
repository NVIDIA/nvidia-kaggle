╭─ Dynamic Programming for TVT Tracking: What Worked, What Didn't, and What th─╮
│ Author: Matteo Niccoli | Votes: 5 | Comments: 0 | Created: 2026-05-27        │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/702919                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Dynamic Programming for TVT Tracking: What Worked, What Didn't, and What the Gap
Tells Us                                                                        

────────────────────────────────────────────────────────────────────────────────
Following up on my earlier discussion on domain priors, I tried a different     
approach to TVT tracking: dynamic programming (DP).                             

The idea in plain terms: The bit traces a physical path through the formation.  
TVT at one position constrains what TVT can be at the next position; it can't   
jump 50 ft in one sample. Most trackers in this competition (particle filters,  
beam search) use heuristics to exploit this constraint. DP is the brute-force   
alternative: evaluate every possible TVT path through the typewell state space  
and pick the one that best matches the observed GR log, subject to a smoothness 
penalty. No randomness, no pruning, guaranteed global optimum for each parameter
set. The trade-off is that you discretize the state space (~400 candidate TVT   
positions in a +/-200 ft window around the anchor).                             

I ran five configurations with different smoothness settings, from "stiff"      
(strongly resist TVT changes between positions, producing smooth paths that     
capture the broad trend) to "loose" (allow rapid TVT changes, producing         
responsive but noisier paths that track local dip). Ensembling these gives the  
model both structural trend and local sensitivity. These paths + their ensemble 
statistics + GR residuals at various offsets = 24-75 features fed into the same 
LightGBM.                                                                       

For the geophysicists: this is the same algorithm as Hale's dynamic warping of  
seismic images. Not conventional seed-based autotracking, but the DP-based      
approach that finds optimal shifts to align two signals. The cost functions are 
structurally identical:                                                         

Dynamic image warping (Hale, 2013) — find shifts u(t) that align image f to     
image g:                                                                        

                                                                                
 min sum_i [ (f(t_i) - g(t_i + u_i))^2 + lambda * |u_i - u_{i-1}| ]             
                                                                                

Geosteering TVT tracking (this notebook) — find TVT states that align lateral GR
to typewell GR:                                                                 

                                                                                
 min sum_i [ (GR_obs_i - GR_typewell[state_i])^2 / sigma + mu * |state_i -      
 state_{i-1}| ]                                                                 
                                                                                

Both minimize a data-fit term (how well the observed signal matches the         
reference at the candidate position) plus a regularization term (how much the   
shift or state jumps between adjacent positions). Replace "seismic trace" with  
"lateral GR log", "reference image" with "typewell", "shift" with "TVT state",  
and you have the same algorithm. The same DP structure appears in speech signal 
alignment (Sakoe and Chiba, 1978), sequence decoding (Viterbi, 1967), and       
seismic image warping (Hale, 2013). All are instances of Bellman's (1962)       
dynamic programming principle. Pure numpy, no Numba, deterministic, ~25 min for 
773 wells.                                                                      

What worked:                                                                    

 • The Viterbi ensemble mean became the #1 feature by LightGBM gain importance, 
   ahead of all NCC and trajectory features                                     
 • 7 of the top 20 features are Viterbi-derived                                 
 • OOF improved by 0.46 total (14.806 to 14.346) across two iterations (v1:     
   +0.154 from 5 Viterbi paths, v2: +0.226 from expanded GR residual families + 
   confidence features + 2-seed blend)                                          

What didn't work:                                                               

 • LB barely moved: 14.081 vs 14.082 baseline (a one-thousandth of a foot       
   improvement)                                                                 
 • This is a textbook CV-overfitting signature: OOF improved 0.46 ft but LB     
   improved 0.001 ft, while fold variance increased from 0.44 to 0.87. The      
   Viterbi features fit the training distribution well but don't generalize to  
   the held-out test wells                                                      
 • Post-processing (alpha/tau grid-searched on OOF) added only ~0.05            
 • Confidence features (cost gap between best and second-best Viterbi path)     
   contributed near-zero gain despite being theoretically the unique advantage  
   of full-state DP over beam search                                            
 • NCC-centered GR residual family was the least informative; anchor-centered   
   was the most informative                                                     

The interesting tension: the model ranks Viterbi features #1 by importance but  
the score barely improves on LB. This means the Viterbi paths carry signal that 
is partially redundant with existing NCC features. Both measure GR-typewell     
similarity; the Viterbi adds sequential consistency but the observation model   
(point-by-point GR comparison) is too noisy for the consistency to help much.   

The lesson - it's the observation model, not the DP architecture:               

The Viterbi architecture is correct: it keeps all states alive (unlike beam     
search which prunes to 8-20) and guarantees the global optimum per parameter    
set. But its observation cost is crude. A single GR value is a weak             
discriminator between adjacent TVT positions. Independent validation from       
sleep3r's shuffled-GR experiment: their normal-GR top-10 heatmap coverage was   
essentially indistinguishable from shuffled GR (the shuffled version actually   
scored marginally higher, which is the noise floor). Point-by-point GR matching 
operates in the noise regime.                                                   

From reading the published top-scoring notebooks and discussion posts, the      
pattern I see is that the leading solutions compensate not with better decoders,
but with a fundamentally different signal source: spatial structural information
from neighboring wells. The ~5 ft gap between ~14 and ~9 RMSE appears to be not 
about more features or better inference over the GR-typewell match, but about   
adding a second, independent information channel.                               

Quantifying the problem structure - two numbers that look contradictory but     
aren't:                                                                         

We tested the naive prediction TVT = -Z + C (flat formation assumption). In the 
lateral section: 55 ft RMSE. Per-well correlation between dTVT and -dZ          
increments: median r = 0.79 (r-squared = 0.62).                                 

These are consistent, not contradictory. The r = 0.79 measures increment        
direction correlation: when Z goes down, TVT usually goes up, and vice versa.   
But the within-lateral dTVT/dZ slope is ~+0.057, not the -1 that the naive model
assumes. Z wiggles ~14x more than TVT in the lateral because the bit is drilling
roughly horizontal through a nearly flat formation. High directional correlation
+ near-zero slope + large cumulative drift coexist: the naive model accumulates 
a growing error despite "predicting the right direction" 62% of the time. Our GR
matching reduces 55 to 14.3 ft by capturing the actual formation dip; the       
remaining gap to sub-10 requires spatial structure from neighboring wells.      

A cautionary tale - GR scaling:                                                 

The first run produced completely flat paths (path_range = 0.0 for all five     
configs). Root cause: clean_gr() z-scores the GR, shrinking values to [-3, 3].  
With z-scored GR, the observation cost was ~0.03/position while movement costs  
were 4-35. The DP never justified leaving the anchor. Fix: raw GR in API units  
for the Viterbi cost function, while NCC features continue using z-scored GR    
upstream. If you're building a physics-based cost function on top of normalized 
features, check the scaling.                                                    

Cross-domain connections:                                                       

The same DP structure appears across fields under different names:              

 • Seismic image warping: Hale (2013) - "dynamic warping"; Yan and Wu (2021) -  
   "DP-based horizon extraction"                                                
 • Speech recognition: Sakoe and Chiba (1978) - "dynamic time warping"          
 • Communications: Viterbi (1967) - "Viterbi algorithm"                         
 • Bioinformatics: gene structure decoding - "HMM decoding"                     
 • Automated geosteering: Zeng, Bhaidasna, and Zou (IADC/SPE-230729-MS, 2026) - 
   particle filter + DTW, validating the log-correlation approach in a          
   commercial context                                                           

The geoscience community knows this as "dynamic warping" or "DP-based horizon   
extraction"; the signal processing community calls it the "Viterbi algorithm."  
Four-phase structure is identical: initialize, forward-accumulate, minimize     
terminal state, backtrack.                                                      

A second algorithmic pattern: structural guide + local matcher                  

The DP discussion above covers the sequential tracker side of the problem. But  
from reviewing the top-scoring public notebooks in this competition             
(particularly the v43 spatial-pooling notebook and the inference-stack          
notebook), I see a second pattern at work: a smooth regional surface provides a 
structural backbone, and local signal matching corrects the residual.           

In seismic interpretation, Gogia et al. (2020, Interpretation/SEG-dGB) built a  
hybrid horizon tracker that combines inversion-based dip flattening (a smooth   
surface fitted to the regional dip field) with similarity-based autotracking    
(local event correlation that snaps the surface to the correct reflector).      
Without the autotracker, the dip-only surface drifts off-phase; without the dip 
surface, the autotracker creates holes and loop-skips. The hybrid outperforms   
either component alone.                                                         

The same decomposition appears in this competition. The spatial structural      
backbone (formation plane fits from neighboring wells, implemented as           
FormationPlaneKNN in the top-scoring public notebooks) is the dip-flattening    
analog: a smooth regional surface that predicts TVT from spatial position alone.
GR-typewell matching (NCC, Viterbi, particle filters) is the autotracker analog:
it locks the prediction to the correct stratigraphic position using the local   
log signature. Neither alone reaches sub-10; the combination does.              

This reframes the 10-vs-14 gap. Our notebook explored the local-matcher side    
thoroughly (three decoder variants, one observation model, one ceiling) but     
lacked the structural guide. The top-scoring notebooks I reviewed have both. The
lesson generalizes: in any tracking problem where a smooth prior surface exists 
and local observations are noisy, the hybrid architecture (structural guide +   
local matcher) dominates either component.                                      

References:                                                                     

 • Hale, D. (2013). Dynamic warping of seismic images. Geophysics, 78(2),       
   S105-S115.                                                                   
 • Yan, S. and Wu, X. (2021). Seismic horizon extraction with dynamic           
   programming. Geophysics, 86(2), IM51-IM62.                                   
 • Sakoe, H. and Chiba, S. (1978). Dynamic programming algorithm optimization   
   for spoken word recognition. IEEE Trans. ASSP, 26, 43-49.                    
 • Gogia, R., Singh, R., de Groot, P., et al. (2020). Tracking 3D seismic       
   horizons with a new hybrid tracking algorithm. Interpretation, 8(4), 1-7.    
 • Zeng, Y., Bhaidasna, K., and Zou, A. (2026). IADC/SPE-230729-MS.             

Previous discussion on domain priors: link                                      

No comments.
