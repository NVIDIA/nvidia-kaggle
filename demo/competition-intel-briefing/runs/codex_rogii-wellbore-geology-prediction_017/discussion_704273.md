╭─ How much should we trust the LB score? ─────────────────────────────────────╮
│ Author: 寿! | Votes: 12 | Comments: 6 | Created: 2026-06-04                  │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/704273                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯



I'm relatively new to this competition, but it seems there's some distribution  
shift between train and test. My local CV and public RMSE diverge by up to 2 in 
some cases. On top of that, which one is more optimistic (local vs. public)     
varies depending on the prediction approach. Methods that rely on specific      
assumptions or modeling tend to show a larger gap. In my case, a spatial method 
using offset wells gives CV < LB (public is more pessimistic), while the        
particle filter approach that's been popular in recent notebooks gives CV > LB  
(public is more optimistic).  My thinking is that since the training set has 773
wells and the public test set only has 52, we should trust local CV over the LB,
assuming our validation strategy is sound.  What do you all think?              

── 6 Comments ──

Ulrich G. ▲1 2026-06-04
I think we could trust, for the time being there is a line-up between CV and LB 
for me                                                                          

Tim Krige ▲2 2026-06-04
In my opinion, both are important. I think that leaderboard probing is a real   
risk here, and your comment of trusting local CV therefore has some merit,      
however, dataleaks are of critical importance. I think an honest take would be  
to consider estimated confidence scores based on the population size of the test
set. I.e., determine the confidence of the local CV being similar or different  
to the LB score with something like a null hypothesis test. This may help to    
determine if the lb score is true or fake, but requires honest engineering of   
the model too.                                                                  

Hope this helps!                                                                

寿! ▲1 2026-06-04
That makes sense. I also feel like there tends to be a trend where LB improves  
when CV improves, though the ranges seem to be on different scales.             

寿! ▲1 2026-06-04
Thank you for the insightful advice! You're right that a statistically-grounded 
approach seems to be key here.                                                  

Tucker Arrants 2026-06-04
Yes, when I make larger pipeline changes, my CV-LB correlation "resets" but then
any CV improvements with small tuning within that new pipeline always lead to LB
improvements.                                                                   

Perhaps has to do the "harder" training wells that massively hurt CV, so        
improvements on the hard wells at the cost of performance on the easy wells can 
result in better CV but not necessarily translate to LB (perhaps because public 
LB has many "easy" wells but these are all relative terms to your model).       

Jack ▲2 2026-06-04
🌆 
inbox%2F21922722%2F780d4fa058448626816e8d6a0864ab3e%2Fdfa08573-5a0e-4345-bab0-b7
52fa2e3dc5.png?generation=1780610842567617&alt=media                            

