╭─ Can a single model achieve LB/CV below 10.0? ───────────────────────────────╮
│ Author: NobelK | Votes: 12 | Comments: 10 | Created: 2026-05-13              │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/699207                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

Hi everyone,                                                                    

I would like to ask whether it seems possible to achieve a score below 10.0 on  
the LB or local CV using a single model.                                        

At the moment, my single CatBoost model has plateaued. I am using mostly tabular
features with CatBoost, and my local CV and LB are no longer improving much     
after basic parameter tuning. I suspect that either my validation strategy or   
feature engineering may be the bottleneck, but I am not sure how to diagnose it.

Since I am still a beginner, I would really appreciate any discussion or        
exchange of ideas around this topic.                                            

For example, I am interested in hearing about:                                  

 • whether people have seen single-model scores below 10.0                      
 • whether CatBoost alone seems strong enough for this competition              
 • which direction is more promising: feature engineering, validation design,   
   post-processing, or ensembling                                               
 • common mistakes that may cause a CatBoost baseline to get stuck              

I am not asking for anyone’s private solution, but I would be grateful for any  
general hints, observations, or advice that could help beginners understand     
where to focus next.                                                            

Thanks in advance!                                                              

── 10 Comments ──

Yang Wei Hao ▲-1 2026-05-13
i think it must do that                                                         

Tucker Arrants ▲2 2026-05-13
Of course, the competition has only been live for a week. I have a simple LGB   
that scores 9.7 on the leaderboard - nothing fancy, just some feature           
engineering. I’m sure the final scores will be much lower, maybe around 5 feet, 
but it is hard to tell. Look at the leaderboard...there are some heavy hitters  
competing here and it has only just begun - it's going to be a good one.        

NobelK ▲1 2026-05-13
9.7... That's fantastic.                                                        

I still seem to lack the fundamentals, so I'll do my best to catch up.          

Thank you for the helpful information; let's both do our best.                  

Tom ▲1 2026-05-17
Current GBDTs are definitely not a good solution for this challenge. Based on my
current EDA, there is still significant room for improvement. Ultimately, it    
depends on how the problem is reformulated. I'll post some new directions after 
I understand the data deeper.                                                   

hengck23 ▲1 2026-05-17
my experment results: upper bound 3.5 (due to ambigous annotation). i think a   
good model is about 4.5. GBDTs is ok if the input is good.                      

but problem is not feature engineering, it is problem formulation. It is easier 
to work with CNN and transformer ( top k path query)                            

NobelK ▲1 2026-05-17
I'd like to use CNNs and transformers, but I lack the knowledge to build a model
properly.                                                                       

I would be grateful if you could share any helpful resources or resources you   
know of.                                                                        

Andrew Lukyanenko ▲3 2026-05-20
This is definitely possible. I got 9.463 with a single model.                   

NobelK 2026-05-20
9.463 with a single model!? That's amazing! I'm very interested in your         
approach. I've run out of ideas right now...                                    

Vishal Kishore ▲4 2026-05-24
Yeah it is possible, I scored 8.8 with a single dl model approach only. Only    
matters how you formulate it in your model                                      

NobelK 2026-05-24
That's a fantastic score!                                                       

I have a basic question: what does it mean to "formulate it in your model"?     

