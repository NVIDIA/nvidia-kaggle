╭─ Is online learning / test-time fine-tuning allowed? ────────────────────────╮
│ Author: Kh0a | Votes: 17 | Comments: 5 | Created: 2026-05-08                 │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/698002                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

With the current evaluation setup, participants can train the model offline and 
then fine-tune it using the test dataset at submission time by peeking at the   
actual test dataset. They can only extract the available TVT_input (calibration 
reference) and other features but can make a solid improvement through domain   
adaptation.                                                                     

Specifically, at submission time in a Kaggle notebook:                          

 1 Load the hidden test data (wellbores + GR, formation parameters, TVT_input)  
 2 Fine-tune the pre-trained model using test features as input (self-supervised
   learning using TVT_input as calibration targets)                             
 3 Generate final predictions on the adapted model                              

Is this approach considered allowed under the competition rules? Looking forward
to feedback from organizers.                                                    

── 5 Comments ──

Kh0a ▲2 2026-05-08
I have tested with same training setup:                                         

online training: 10.953                                                         

no online training: 11.323                                                      

The features processing idea was from                                           
https://www.kaggle.com/code/shinyanagai123/triple-signal-beam-search-dual-pf-lig
htgbm                                                                           

Tucker Arrants ▲2 2026-05-14
Thank you for sharing your augmentation + online training technique. I          
consistently get about a 0.15 - 0.2 improvement from it.                        

Most recent training run:                                                       

5 fold LGB without -> 9.812                                                     

5 fold LGB, n_aug_splits = 1, online_training = True -> 9.649                   

5 fold CatBoost without -> 9.869                                                

5 fold CatBoost, n_aug_splits = 1, online_training = True -> 9.713              

Kh0a ▲1 2026-05-15
Well done, although i am not sure if this is allowed yet.                       

PC Jimmmy 2026-05-16
Test time learning has been an acceptable method for all the years I have been  
on kaggle.  The only issue I ever had was the compute time limit - in some past 
competitions my methods were too slow for the compute/time that was available.  

hengck23 2026-05-16
it should be allowed. Such methods had been used in previous kaggle competitions
before.                                                                         
it also includes things like                                                    

 • finding statistic  like mean, std of test data                               
 • creating multiple window templates for matching                              
 • self-supervsied learning or fine-tuning or adaptation etc                    

