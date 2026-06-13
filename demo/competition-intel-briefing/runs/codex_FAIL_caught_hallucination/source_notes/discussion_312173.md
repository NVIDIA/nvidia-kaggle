╭─ Tip. Increase your score! ──────────────────────────────────────────────────╮
│ Author: Manhow | Votes: 14 | Comments: 0 | Created: 2022-03-10               │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/312173      │
╰──────────────────────────────────────────────────────────────────────────────╯

Everybody looking for an idea on how to increase your score I dedicate this     
topic. Being a novice I found a method that allows me to stay in top 10         
(10.03.2022). Try to average received probabilities on each validation split    
(list(probabilities)/len(probabilities)), predict_proba may help you. Averaging 
the model is a powerful tool to get a high validation score.💯🙂                

── 4 Comments ──

Andy CHEUNG ▲1 2022-03-12
Hi Micheal thanks for your tip! However I'm not sure about averaging a model, is
it cross validation?                                                            

 2022-03-12

Manhow ▲2 2022-03-12
It is! On each validation split train chosen classifier, after that you can     
predict probabilities. After the loop the average probabilities should be       
calculated to make predictions. Something like this:                            

                                                                                
 folds = StratifiedKFold(n_splits = 5, shuffle=True)                            
 for fold, (train_id, test_id) in enumerate(folds.split(X, y)):                 
                                                                                
     print("fold : ", fold + 1, end = ' ')                                      
     # Split data                                                               
     X_train = X.iloc[train_id]                                                 
     y_train = y.iloc[train_id]                                                 
     X_valid = X.iloc[test_id]                                                  
     y_valid = y.iloc[test_id]                                                  
                                                                                
     # Train RF                                                                 
     model =  RandomForestClassifier()                                          
     model.fit(X_train, y_train)                                                
                                                                                
     # Print validation score to see how it works                               
     valid_pred = model.predict(X_valid)                                        
     valid_score = accuracy_score(y_valid, valid_pred)                          
     print( "Validation score: ", valid_score, end = ' ')                       
                                                                                
     # test                                                                     
     y_probs.append(model.predict_proba(test))                                  
     print(" ")                                                                 
                                                                                
 y_prob_3 = sum(y_probs) / len(y_probs)                                         
 # After assign True/False with given probabilities (if right prob > left, assi 
 True) to sample_submission. 👌😃                                               
                                                                                

Omikumar Bhavinkumar Makadia 2023-08-15
Thank you for the help !                                                        

