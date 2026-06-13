╭─ cv and lb correlations ..... ───────────────────────────────────────────────╮
│ Author: Gaurav Rawat | Votes: 12 | Comments: 16 | Created: 2026-05-19        │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/701691                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

Was seeing some notebooks getting better in LB based on the cv , So far wanted  
to check how others have been getting the correlation going for tabular and non 
tabular models . For me have been using the standard GroupKfold on wells .      

Train Log                                                                       

                                               
  Version         CV RMSE (ft)   LB RMSE (ft)  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  train_v2.py     31.3871        35.843        
  train_v2.1.py   14.7065        13.949        
  v2.2            14.4634        13.777        
  v2.5            11.9993        12.383        
  v2.6            11.2693        12.383        
  v2.7            10.7485        10.606        
  v2.7.1          10.2486        -             
  v2.8            10.6543        -             
  v2.7.2 online   -              10.520        
  v2.9            10.3256        9.816         
  v2.10           10.3730        10.384        
  v2.9.7          10.37          9.585         
  v2.9.11         10.6           8.739         
                                               

── 16 Comments ──

Tucker Arrants ▲5 2026-05-19
With the plain jane GBDT models, CV around 11.00 split on well ID and           
leaderboard around 9.6                                                          

Large gap, but very stable -> all CV improvements have led to LB improvements   
(so far).                                                                       

A lot of the public notebooks have leakage which leads to a smaller CV-LB gap,  
but is not as trustworthy.                                                      

Gaurav Rawat ▲1 2026-05-19
feel need to beat the 10 mark in cv to see marked improvements for my           
experiments . NN so far for me havent been doign that great maybe need to dive  
deep to design them better .                                                    

Durga Kumari ▲-4 2026-05-19
Interesting that v2.10 had slightly worse CV but matched LB almost perfectly.   
Usually a good sign the model is generalizing more consistently rather than     
optimizing fold-specific patterns.                                              

Hassan Gasim ▲-3 2026-05-20
Outstanding progress so far. Regarding your note on Neural Networks             
underperforming: standard MLPs usually struggle with the spatial, sequential    
nature of wellbore data compared to GBDTs. Since well logs are essentially      
depth-series data, have you considered a hybrid architecture? Implementing a    
1D-CNN or a light Transformer backbone (like TabNet or FT-Transformer) grouped  
by Well ID can capture the vertical stratigraphy layers much better than vanilla
NNs. Looking forward to seeing how your NN experiments evolve once the          
architecture matches the geological domain!                                     

shanzhong8 ▲1 2026-05-21
CV 10.7  , LB 9.9                                                               

Gaurav Rawat 2026-05-22
nice GBDT or NN .. was wondering how much folks are gettign with NN CV          

shanzhong8 ▲2 2026-05-23
Transformer                                                                     

Gaurav Rawat ▲1 2026-05-28
Adding NN experiments now , just baselines now                                  

 • CV 14.4 LB 17                                                                
 • cv 8 lb 9                                                                    

Tucker Arrants ▲6 2026-05-29
Single model NN update:                                                         

CV 8.5, LB 7.5                                                                  

Inference in 2 minutes lol                                                      

Gaurav Rawat ▲1 2026-05-29
awesome ya I see NN infer like 2-3 mins .. :) maybe u framed the right Arch ..  
my cv not going down                                                            

Tucker Arrants ▲3 2026-06-08
LB feels noisy to me. I often observe CV improvements of 0.7 feet or more,      
leading to regressions in LB. I think some of my submissions were "lucky" e.g.  
CV around 8 scoring 6.6 on LB.                                                  

Latest results:                                                                 

CV 6.7, LB 6.3 -> the "lucky gap" between CV and LB I previously observed (and  
can be observed on all the public notebooks) is starting to shrink. Think we    
need to be careful with LB here. Small number of public test set wells and heavy
tail problem…                                                                   

Ruby ▲5 2026-06-08
recent two experiments: CV 6.74 LB 6.48 CV 6.22 LB 7.18 I guess it is dominated 
by some bad cases                                                               

Jack 2026-06-08
I'd be questioning where those CV improvements are coming from in relation to   
past runs - might be insightful. I'm still trying to figure out what the heck   
you're doing for 2 mins inference.. well, that and orbit wars lol               

Gaurav Rawat 2026-06-08
I dunno CV strategy needs to be alteare in one I got cv 7.4 but LB is like 9 .  
maybe need to have cv mixed with hard wells vs easy ones per fold or some custom
way                                                                             

Jack 2026-06-08
But how would you define hard vs easy wells? I can think of some more direct    
ways of balancing folds                                                         

YYH 2026-06-09
Are the top-ranked solutions currently all based on physics models combined with
machine learning? It seems that some physics models have performed quite well in
this competition.It is easy to find numerous hard samples in the dataset that   
cannot be predicted accurately by conventional physical models, which drives up 
the RMSE score. I am currently working to address this issue.                   

