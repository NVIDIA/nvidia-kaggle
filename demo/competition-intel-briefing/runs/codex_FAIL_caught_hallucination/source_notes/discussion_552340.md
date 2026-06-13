╭─ Rank 8 approach - simple ML models and Optuna ensemble ─────────────────────╮
│ Author: Ravi Ramakrishnan | Votes: 12 | Comments: 0 | Created: 2024-12-19    │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/552340      │
╰──────────────────────────────────────────────────────────────────────────────╯

Hello all,                                                                      

My 8th place kernels are public and are available at the below links-           

 1 https://www.kaggle.com/code/ravi20076/spaceshiptitanic-models/notebook       
 2 https://www.kaggle.com/code/ravi20076/spaceshiptitanic-blending              

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                   Approach                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


                              Feature engineering                               

Thanks to @arunklenin for the FE notebook from the other Spaceship Titanic      
competition. Kernel is here                                                     


                                 Model training                                 

 1 I used a stratified 10-fold CV scheme                                        
 2 I resorted to a gamut of gbdt models including Catboost, LightGBM and XgBoost
   along with scikit-learn models like Random Forest and Histogram Gradient     
   Boosting Classifier for the pipeline                                         


                                    Blending                                    

I used Optuna to blend the results of the individual models                     


                              Areas of improvement                              

 1 I could have used AutoML tools                                               
 2 I could have experimented with NNs                                           
 3 Better blending strategies may have created a better result                  

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              Concluding remarks                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Thanks to @carlmcbrideellis for the competition. Happy learning and best        
regards!                                                                        

── 2 Comments ──

Minato Namikaze ▲1 2024-12-19
Amazing @ravi20076!                                                             

Rahul kumar Sharma ▲3 2024-12-25
Amazing work! The approach is very well-structured and the use of Optuna for    
blending is impressive.                                                         

