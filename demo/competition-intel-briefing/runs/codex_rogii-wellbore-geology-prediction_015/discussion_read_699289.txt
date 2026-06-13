╭─ Paradigm Shift: Why pure Tabular Models might be hitting a wall (Spatial & ─╮
│ Author: Amged Alfaqih | Votes: 21 | Comments: 4 | Created: 2026-05-13        │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/699289                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

Hi everyone,                                                                    

After spending some time diving deep into the data and experimenting with       
various preprocessing pipelines, I noticed a common trap we might all be falling
into: treating this purely as a standard tabular regression problem.            

If you are just passing [X, Y, Z, MD, GR] into LightGBM or XGBoost, you will    
eventually hit a hard ceiling on your CV/LB score. Why? Because the models are  
missing the physical and spatial realities of the wellbore trajectory.          


                1. The Sequential Reality (Physics of the Drill)                

We aren't just looking at random rows; we are tracing a path. Features like MD  
(Measured Depth) and Z dictate a trajectory.                                    

Instead of simple lags and rolling windows, has anyone experimented with        
Particle Filters (PF) or Beam Search? By treating the expected TVT as a moving  
particle that updates its state based on the Gamma Ray (GR) observations and    
spatial constraints, we can create incredibly strong baseline predictions to    
feed into our Gradient Boosting models.                                         


                 2. The Spatial Reality (Geology is Continuous)                 

A well doesn't exist in a vacuum; it shares the same geological formation as its
neighbors.                                                                      

Relying solely on X and Y coordinates in a tree model is inefficient. A better  
approach is using Spatial Imputation (like cKDTree) to find the nearest known   
wells and calculate the median TVT or formation depth in that specific localized
area.                                                                           

Feeding this "spatial neighborhood consensus" as a feature massively stabilizes 
the predictions for unseen evaluation rows.                                     

By shifting the focus from "tuning model hyperparameters" to "building          
physics-aware and spatial-aware features", the performance jump is massive.     

Are you guys currently using any sequential tracking (like Particle             
Filters/Kalman Filters) or mostly relying on heavy rolling/lag statistical      
features?                                                                       

Would love to hear your thoughts!                                               

── 4 Comments ──

hengck23 ▲11 2026-05-13
Check rogii patent on its product startsteer. Its viterbi/ beam search includes 
using dip equation                                                              
US20190106974A1 — “Systems and methods for horizontal well geosteering”         
US20230019126A1 — “Methods for Assisted and Automated Horizontal Well           
Geosteering”                                                                    

 • select big segment size.                                                     
 • select dip range.                                                            
 • select big segment.                                                          
 • select basic algorithm.                                                      
 • run through segment with identified dip.                                     
 • calculate accuracy by K-value.                                               
 • select best algorithm / best K-value.                                        
 • repeat over segment sizes and dip ranges.                                    

                                                                                
 for segment_size in choices:                                                   
     for dip_model in choices:                                                  
         for algorithm in choices:                                              
             project lateral log into TVT                                       
             compare with typewell log                                          
             score by similarity K                                              
 choose best-scoring interpretation                                             
                                                                                

These are the same as shown in the YouTube video the competition host           
introduced. The state transversal is not point by point but rather in           
hierarchical order: match big segments first, then reiterating by breaking big  
segments to smaller ones to increase matching accuracy                          

faizan ▲1 2026-05-14
Really well-articulated post, and I think you've put your finger on something a 
lot of people are quietly running into but haven't named this clearly. The point
about sequential context is especially underrated. Most people's first instinct 
with lag features is a fixed rolling window (last 3–5 rows of GR or MD delta),  
but that's essentially a flat prior — it treats all neighbors equally regardless
of geological consistency. A Kalman Filter or even a lightweight Particle Filter
changes that entirely by letting the model track state and weight recent        
observations by how well they fit the expected trajectory. The signal quality   
improvement on noisy GR readings alone makes it worth the implementation effort.
On the spatial imputation side, I've had good results with a similar cKDTree    
approach, but one thing worth experimenting with is distance-weighted           
interpolation (IDW) instead of a flat median — nearby wells in the same         
formation should contribute proportionally more than outliers at the edge of    
your search radius. Also, if the competition data has any formation labels or   
lithology markers, conditioning the spatial neighborhood on those (rather than  
raw Euclidean XY distance) tends to produce much cleaner consensus features. One
thing I'd add to your framework: azimuth and inclination derived from the       
trajectory (X, Y, Z, MD) can be reconstructed and used to estimate the          
wellbore's entry angle into each formation. That angular context, combined with 
your spatial neighborhood TVT, gives the gradient boosting model a much richer  
"where am I and where am I headed" signal rather than just "what are the raw    
coordinates." Fully agree that hyperparameter chasing on a feature-poor dataset 
has heavily diminishing returns. The ceiling you're describing is real — the    
delta comes from feature engineering, not from squeezing one more tree out of   
XGBoost.                                                                        

想去看海 ▲-1 2026-05-16
你好，我是初入机器学习的小白选手，对于您的“但值得尝试的一件事是距离加权插值（IDW
）而不是平坦的中值和mdash;”这一观点十分赞同，但我还有一个小小的疑问，我该怎么寻 
找附近的井？是根据每个样本的xyz值吗？这样的话是不是要将train中所有的样本数据综合
起来分析？                                                                      

Durga Kumari ▲1 2026-05-16
Really good point. I think many people are treating this as a pure tabular      
problem while the data is clearly sequential + spatial. The idea of combining   
trajectory-aware features with spatial neighborhood information makes a lot of  
sense, especially for stabilizing predictions across nearby wells.              

