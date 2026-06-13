Reading kernel: viktortaran/space-titanic
Cells: 79 (59 code, 20 markdown) | 21127 chars

🌆 multiple-display-space-planet-atmosphere-wallpaper-thumb.jpg                 

────────────────────────────────────────────────────────────────────────────────

                     ##### 1.Importing necessary libraries                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from sklearn.impute import SimpleImputer                                       
 from sklearn.preprocessing import OneHotEncoder                                
 from sklearn.metrics import mean_absolute_error                                
 import pandas as pd                                                            
 import numpy as np                                                             
 import base64                                                                  
 import seaborn as sns                                                          
 import matplotlib.pyplot as plt                                                
 import os                                                                      
 import random                                                                  
 import gc                                                                      
                                                                                
 from sklearn.impute import SimpleImputer                                       
 from sklearn.preprocessing import OneHotEncoder                                
 from sklearn.preprocessing import StandardScaler                               
 from sklearn.utils import shuffle                                              
 from sklearn.model_selection import train_test_split                           
 from sklearn.utils import shuffle                                              
 from sklearn.metrics import log_loss                                           
 from sklearn.metrics import accuracy_score                                     
 import optuna                                                                  
 import xgboost as xgb                                                          
 from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,      
 GradientBoostingClassifier,                                                    
                               ExtraTreesClassifier, VotingClassifier)          
 from sklearn.discriminant_analysis import LinearDiscriminantAnalysis           
 from sklearn.linear_model import LogisticRegression                            
 from sklearn.neighbors import KNeighborsClassifier                             
 from sklearn.tree import DecisionTreeClassifier                                
 from sklearn.neural_network import MLPClassifier                               
 from catboost import CatBoostClassifier                                        
 from sklearn.svm import SVC                                                    
 from sklearn import datasets, linear_model                                     
 import lightgbm as lgb                                                         
 from sklearn.model_selection import cross_val_score                            
 from sklearn.model_selection import StratifiedKFold                            
                                                                                
 pd.set_option('display.max_columns', None)                                     
                                                                                
 from sklearn.model_selection import train_test_split                           
 from sklearn.ensemble import RandomForestClassifier                            
 import eli5                                                                    
 from eli5.sklearn import PermutationImportance                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 import eli5                                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from lightgbm import LGBMClassifier                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────

                            ##### 2.Loading datasets                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')                 
 sample = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')  
 train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                       ##### 3.Miss information analysis                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(train.isnull().sum())                                                    
 sns.heatmap(train.isnull())                                                    
 print(train.info())                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(test.isnull().sum())                                                     
 sns.heatmap(test.isnull())                                                     
 print(test.info())                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────

                             ##### 4.Preprocessing                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ################################################################               
 ## The passengers were all from the same ship. This means we  ##               
 ## don't have to search for lost information separately.      ##               
 ################################################################               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def get_score(model,X,y):                                                      
     n = cross_val_score(model,X,y,scoring ='accuracy',cv=20)                   
     return n                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 params_XGB_best ={'lambda': 3.0610042624477543,                                
              'alpha': 4.581902571574289,                                       
              'colsample_bytree': 0.9241969052729379,                           
              'subsample': 0.9527591724824661,                                  
              'learning_rate': 0.06672065863100594,                             
              'n_estimators': 725, #initial value is 651                        
              'max_depth': 5,                                                   
              'min_child_weight': 1,                                            
              'num_parallel_tree': 1}                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def t_fold(X,y,n_splits):                                                      
     params= {'lambda': 3.0610042624477543,                                     
              'alpha': 4.581902571574289,                                       
              'colsample_bytree': 0.9241969052729379,                           
              'subsample': 0.9527591724824661,                                  
              'learning_rate': 0.06672065863100594,                             
              'n_estimators': 7250, #initial value is 725                       
              'max_depth': 5,                                                   
              'min_child_weight': 1,                                            
              'num_parallel_tree': 1,                                           
              'early_stopping_rounds':200,}                                     
     results=[]                                                                 
     n_iterations=[]                                                            
     skf = StratifiedKFold(n_splits=n_splits)                                   
     for train_index, test_index in skf.split(X, y):                            
         train_X, valid_X = X.iloc[train_index], X.iloc[test_index]             
         train_y, valid_y = y.iloc[train_index], y.iloc[test_index]             
         model = xgb.XGBClassifier(**params).fit(train_X,train_y,               
                                       eval_set=[(valid_X,valid_y)],            
                                       verbose=0                                
                                      )                                         
         n_iteration = model.get_booster().best_iteration                       
         n_iterations.append(n_iteration)                                       
         result = accuracy_score(valid_y,(model.predict(valid_X)))              
         results.append(result)                                                 
         i=int(sum(n_iterations)/len(n_iterations))                             
     print("Average n_ite=" + str(i))                                           
     print("% of scatter =" + str(np.std(n_iterations)/i))                      
     n=sum(results)/len(results)                                                
     print (n)                                                                  
     print("FIIINISH__________________________________\n")                      
     return n                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test=train.append(test)                                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(train_test.isnull().sum())                                               
 sns.heatmap(train_test.isnull())                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                          ##### 4.1.Expenses+CryoSleep                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 Expenses_columns = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ############################################################################## 
 ################################                                               
 ### Let's fill in missing cost with "0" if a passenger was in CryoSleep. If yo 
 sleep you don't spend money. ###                                               
 ############################################################################## 
 ################################                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test.loc[:,Expenses_columns]=train_test.apply(lambda x: 0 if x.CryoSleep 
 == True else x,axis =1)                                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────

                         ##### 4.2.CryoSleep + Expenses                         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ############################################################################## 
 #############################                                                  
 ### From the description of the contest guidelines we can conclude that if a   
 person is in cryosleep,      ###                                               
 ### he cannot spend money to buy additional services. Moreover, when filling i 
 the missing information   ###                                                  
 ### in the columns: 'Age,RoomService, FoodCourt, ShoppingMall, Spa, VRDec'k, w 
 should remember this      ###                                                  
 ### same feature of cryosleep. Let's fill in the missing information based on  
 this conclusion.            ###                                                
 ############################################################################## 
 #############################                                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ############################################################################## 
 ### First of all, let's create an additional column and call it "expenses".  # 
 ### Expenses = Age + RoomService + FoodCourt + ShoppingMall + Spa + VRDec'k  # 
 ############################################################################## 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test['Expenses'] = train_test.loc[:,Expenses_columns].sum(axis=1)        
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test.loc[:,['CryoSleep']]=train_test.apply(lambda x: True if x.Expenses  
 0 and pd.isna(x.CryoSleep) else x,axis =1)                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────

                                 ##### 4.3.Name                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ############################################################################## 
 ### I am not sure that we are able to restore some information about Names.  # 
 ### Let's fill in this columns with "Unknown Unknown", because maybe in the  # 
 ### future we will have to split the column.                                 # 
 ############################################################################## 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test.Name = train_test.Name.fillna('Unknown Unknown')                    
                                                                                

────────────────────────────────────────────────────────────────────────────────

                                ##### 4.4.Cabin                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ##################################################################             
 ### We have to check PassengerId carefully. Because it         ###             
 ### contains room numbers. So it will help us to find          ###             
 ### the miss Cabin, VIP status, HomePlanet and Destination.    ###             
 ##################################################################             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test.loc[:,['Room']] = train_test.PassengerId.apply(lambda x: x[0:4] )   
 #Here I want to create handbooks to fill in informations in Cabin, VIP status, 
 HomePlanet and Destination.                                                    
 guide_VIP=train_test.loc[:,['Room','VIP']].dropna().drop_duplicates('Room')    
 guide_Cabin=train_test.loc[:,['Room','Cabin']].dropna().drop_duplicates('Room' 
 guide_HomePlanet=train_test.loc[:,['Room','HomePlanet']].dropna().drop_duplica 
 s('Room')                                                                      
 guide_Destination=train_test.loc[:,['Room','Destination']].dropna().drop_dupli 
 tes('Room')                                                                    
 train_test=pd.merge(train_test,guide_Cabin,how="left",on='Room',suffixes=('',' 
 '))                                                                            
 train_test=pd.merge(train_test,guide_VIP,how="left",on='Room',suffixes=('','_y 
 )                                                                              
 train_test=pd.merge(train_test,guide_HomePlanet,how="left",on='Room',suffixes= 
 ','_y'))                                                                       
 train_test=pd.merge(train_test,guide_Destination,how="left",on='Room',suffixes 
 '','_y'))                                                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test.loc[:,['VIP']]=train_test.apply(lambda x: x.VIP_y if pd.isna(x.VIP) 
 else x,axis=1)                                                                 
 train_test.loc[:,['Cabin']]=train_test.apply(lambda x:  x.Cabin_y if           
 pd.isna(x.Cabin) else x,axis=1)                                                
 train_test.loc[:,['HomePlanet']]=train_test.apply(lambda x:  x.HomePlanet_y if 
 pd.isna(x.HomePlanet) else x,axis=1)                                           
 train_test.loc[:,['Destination']]=train_test.apply(lambda x:  x.Destination_y  
 pd.isna(x.Destination) else x,axis=1)                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test[train_test.Cabin.isnull()]                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig,ax = plt.subplots(figsize=(30,10))                                         
 pd.set_option('display.max_rows', 100)                                         
 print(train_test.isnull().sum())                                               
 sns.heatmap(train_test.isnull())                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                                 ##### 4.5.Age                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ##########################################################################     
 ### Let's see how Age depends on other columns to find some miss data. ###     
 ##########################################################################     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig, ax = plt.subplots(figsize=(15,5))                                         
 sns.heatmap(train_test.corr(),annot=True)                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig, ax = plt.subplots(figsize=(15,5))                                         
 analys = train_test.loc[:,['Age','Expenses']]                                  
 ax.scatter(analys.Age,analys.Expenses)                                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 pd.set_option('display.max_rows', 100)                                         
 print(train_test.isnull().sum())                                               
 sns.heatmap(train_test.isnull())                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                      ##### 4.6.HomePlanet + Destination.                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 analys = train_test.loc[:,['HomePlanet','Destination']]                        
 analys['numeric'] =1                                                           
 analys.groupby(['Destination','HomePlanet']).count()                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 pd.set_option('display.max_rows', 100)                                         
 print(train_test.isnull().sum())                                               
 sns.heatmap(train_test.isnull())                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                             ##### 4.7.Split Cabin.                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #######################################################                        
 ### Let's split the column and drop usless columns. ###                        
 #######################################################                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test.loc[:,['Cabin_1']] =                                                
 train_test.Cabin.str.split("/",expand=True).iloc[:,0]                          
 train_test.loc[:,['Cabin_2']] =                                                
 train_test.Cabin.str.split("/",expand=True).iloc[:,1]                          
 train_test.loc[:,['Cabin_3']] =                                                
 train_test.Cabin.str.split("/",expand=True).iloc[:,2]                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_test.loc[:,['FirstName']] = train_test.Name.str.split("                  
 ",expand=True).iloc[:,0]                                                       
 train_test.loc[:,['SecondName']] = train_test.Name.str.split("                 
 ",expand=True).iloc[:,1]                                                       
 train_test['Name_key']=train_test['SecondName']+train_test['Room']             
                                                                                

────────────────────────────────────────────────────────────────────────────────

              ##### 4.8.Let's apply SimpleImputer + OneHotEncoder.              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 num_cols =                                                                     
 ['ShoppingMall','FoodCourt','RoomService','Spa','VRDeck','Expenses','Age']     
 cat_cols = ['CryoSleep','Cabin_1','Cabin_3','VIP','HomePlanet','Destination']  
 transported=['Transported']                                                    
 #notvg =                                                                       
 ['HomePlanet','VIP','ShoppingMall','FoodCourt','Age','Cabin_2','Destination']  
 train_test = train_test[num_cols+cat_cols+transported].copy()                  
                                                                                
 num_imp = SimpleImputer(strategy='mean')                                       
 cat_imp = SimpleImputer(strategy='most_frequent')                              
 ohe = OneHotEncoder (handle_unknown='ignore',sparse = False)                   
                                                                                
                                                                                
 train_test[num_cols] =                                                         
 pd.DataFrame(num_imp.fit_transform(train_test[num_cols]),columns=num_cols)     
 train_test[cat_cols] =                                                         
 pd.DataFrame(cat_imp.fit_transform(train_test[cat_cols]),columns=cat_cols)     
 temp_train = pd.DataFrame(ohe.fit_transform(train_test[cat_cols]),columns=     
 ohe.get_feature_names_out())                                                   
 train_test = train_test.drop(cat_cols,axis=1)                                  
 train_test = pd.concat([train_test,temp_train],axis=1)                         
                                                                                

────────────────────────────────────────────────────────────────────────────────

                   ##### 4.9.Let's split train and test set.                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train = train_test[train_test['Transported'].notnull()].copy()                 
 train.Transported =train.Transported.astype('int')                             
 test = train_test[train_test['Transported'].isnull()].drop("Transported",axis= 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X = train.drop('Transported',axis=1)                                           
 y = train.Transported                                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X,y = shuffle(X,y)                                                             
 X = X.reset_index(drop=True)                                                   
 y = y.reset_index(drop=True)                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(get_score(xgb.XGBClassifier(**params_XGB_best),X,y).mean())              
                                                                                

────────────────────────────────────────────────────────────────────────────────

                      ##### 4.10.Let's deal with outliers.                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 features_isolation                                                             
 =['ShoppingMall','FoodCourt','RoomService','Spa','VRDeck','Age']               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from sklearn.ensemble import IsolationForest                                   
                                                                                
 isf = IsolationForest(n_jobs=-1,                                               
 random_state=1,n_estimators=100,contamination=0.003)                           
 isf.fit(X[features_isolation], y)                                              
                                                                                
 rows = pd.DataFrame(isf.predict(X[features_isolation]),columns=['feature'])    
 rows_ind = rows[rows.feature == 1]                                             
 results = pd.DataFrame()                                                       
 results['results'] = - isf.score_samples(X[features_isolation])                
 results['chance'] = isf.decision_function(X[features_isolation])               
 results['yes/no'] = isf.predict(X[features_isolation])                         
                                                                                
 new = results[results['yes/no']==1]                                            
 old= results[results['yes/no']==-1]                                            
 fig,ax = plt.subplots(figsize=(20,10))                                         
 ax.set_facecolor('black')                                                      
 ax = plt.scatter(new.index,new.results,marker=".")                             
 ax = plt.scatter(old.index,old.results,color='white',marker="*")               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X_1 = X.iloc[rows_ind.index].reset_index(drop=True)                            
 y_1 = y.iloc[rows_ind.index].reset_index(drop=True)                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(get_score(xgb.XGBClassifier(**params_XGB_best),X_1,y_1).mean())          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #############################################################                  
 ### I found no evidence for the use of IsolationForest.   ###                  
 ### The result got worse with different contamination.    ###                  
 #############################################################                  
                                                                                

────────────────────────────────────────────────────────────────────────────────

                       ##### 4.11.Permutation Importance.                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig = plt.subplots(figsize=(30,10))                                            
 sns.heatmap(train.corr(),annot=True)                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 params_XGB_best= {'lambda': 3.0610042624477543,                                
              'alpha': 4.581902571574289,                                       
              'colsample_bytree': 0.9241969052729379,                           
              'subsample': 0.9527591724824661,                                  
              'learning_rate': 0.06672065863100594,                             
              'n_estimators': 725, #initial value is 651                        
              'max_depth': 5,                                                   
              'min_child_weight': 1,                                            
              'num_parallel_tree': 1}                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 perm = PermutationImportance(xgb.XGBClassifier(**params_XGB_best),             
 random_state=1,n_iter =10,cv=5).fit(X, y)                                      
 eli5.show_weights(perm, feature_names = X.columns.tolist(),top=50)             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 drop_list=['ShoppingMall','Age','CryoSleep_True','HomePlanet_Earth','HomePlane 
 Europa',                                                                       
 'VIP_True','HomePlanet_Mars','Destination_PSO J318.5-22','VIP_False',          
 'Destination_55 Cancri e','FoodCourt','Destination_TRAPPIST-1e']               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X=X.drop(drop_list,axis=1)                                                     
 test=test.drop(drop_list,axis=1)                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(get_score(xgb.XGBClassifier(**params_XGB_best),X,y).mean())              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from imblearn.over_sampling import SMOTE                                       
 smote = SMOTE(sampling_strategy=1, n_jobs=-1)                                  
 X_sm, y_sm = smote.fit_resample(X, y)                                          
 X = X_sm                                                                       
 y = y_sm                                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 y.value_counts()                                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────

                                ##### 5.Modeling                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 ###############################                                                
 # Let's define some settings ##                                                
 ###############################                                                
 optuna_study = "ON"                                                            
 LGBM_study = "OFF"                                                             
 XGB_study = "OFF"                                                              
 CAT_study = "OFF"                                                              
                                                                                
                                                                                
 gpu_switch = "OFF"                                                             
 skf = StratifiedKFold(n_splits=10)                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────

                                ##### 5.1. LGBM                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 if gpu_switch == "ON":                                                         
     method = "gpu"                                                             
 else:                                                                          
     method = "cpu"                                                             
                                                                                
 def objective(trial):                                                          
     param = {                                                                  
      'objective': 'binary',                                                    
      'device': method,                                                         
      'metric': 'binary_logloss',                                               
      'verbosity': -1,                                                          
      'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),                
      'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),                
      'learning_rate': trial.suggest_float('learning_rate', 0.0001,0.1),        
      'num_leaves': trial.suggest_int('num_leaves', 2, 512),                    
      'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),    
      'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),    
      'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),                  
      'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),      
      'n_estimators' : trial.suggest_int('n_estimators', 100, 20000),           
      'subsample':None,                                                         
      'subsample_freq':None,                                                    
      'reg_alpha':None,                                                         
      'colsample_bytree':None,                                                  
      'reg_lambda':None,                                                        
      'early_stopping_round':trial.suggest_int('early_stopping_round', 200, 200 
             }                                                                  
     results=[]                                                                 
     n_iterations=[]                                                            
     for train_index, test_index in skf.split(X, y):                            
         train_X, valid_X = X.iloc[train_index], X.iloc[test_index]             
         train_y, valid_y = y.iloc[train_index], yi.loc[test_index]             
         model = lgb.LGBMClassifier(**param).fit(train_X,train_y,               
                                             eval_set=[(valid_X,valid_y)],      
                                             callbacks=[lgb.log_evaluation(peri 
 =0, show_stdv=False)]                                                          
                                              )                                 
         n_iteration = model.best_iteration_                                    
         n_iterations.append(n_iteration)                                       
         result = accuracy_score(valid_y,(model.predict(valid_X)))              
         results.append(result)                                                 
         i=int(sum(n_iterations)/len(n_iterations))                             
     print("Average n_ite=" + str(i))                                           
     print("% of scatter =" + str(np.std(n_iterations)/i))                      
     n=sum(results)/len(results)                                                
     print(n)                                                                   
     print("FIIINISH__________________________________\n")                      
                                                                                
     return n                                                                   
                                                                                
 if optuna_study == LGBM_study:                                                 
     study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),       
                                 direction='maximize')                          
     study.optimize(objective, n_trials=10000)                                  
     print('Best trial:', study.best_trial.params)                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Trial 1468 finished with value: 0.8084681392022881                            
 params_LGBM= {'lambda_l1': 6.183557865875619,                                  
               'lambda_l2': 0.011470762415538793,                               
               'learning_rate': 0.08693013162698361,                            
               'num_leaves': 330,                                               
               'feature_fraction': 0.6583455294128855,                          
               'bagging_fraction': 0.8666491286986552,                          
               'bagging_freq': 1,                                               
               'min_child_samples': 17,                                         
               'n_estimators': 739}                                             
 #Average n_ite=739                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────

                                 ##### 5.2. XGB                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 if gpu_switch == "ON":                                                         
     method = "gpu_hist"                                                        
 else:                                                                          
     method = "hist"                                                            
                                                                                
 def objective(trial):                                                          
     print("START__________________________________")                           
     param = {                                                                  
         'tree_method':method,                                                  
         'objective': 'binary:logistic',                                        
         'eval_metric': 'logloss',                                              
         'lambda': trial.suggest_float('lambda', 0, 10.0),                      
         'alpha': trial.suggest_float('alpha', 0, 10.0),                        
         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1,1.0),  
         'subsample': trial.suggest_float('subsample', 0.2,1.0),                
         'learning_rate': trial.suggest_float('learning_rate', 0.0001,0.1),     
         'n_estimators': trial.suggest_int('n_estimators', 100,3000),           
         'max_depth': trial.suggest_categorical('max_depth',                    
 [2,3,4,5,6,7,8,9,10]),                                                         
         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),      
         'num_parallel_tree': trial.suggest_int('num_parallel_tree',1,1),       
         #'early_stopping_rounds':200,                                          
         }                                                                      
                                                                                
     results=[]                                                                 
     n_iterations=[]                                                            
     for train_index, test_index in skf.split(X, y):                            
         train_X, valid_X = X.iloc[train_index], X.iloc[test_index]             
         train_y, valid_y = y.iloc[train_index], y.iloc[test_index]             
         model = xgb.XGBClassifier(**param).fit(train_X,train_y,                
                                       eval_set=[(valid_X,valid_y)],            
                                       verbose=0                                
                                      )                                         
         #n_iteration = model.get_booster().best_iteration                      
         #n_iterations.append(n_iteration)                                      
         result = accuracy_score(valid_y,(model.predict(valid_X)))              
         results.append(result)                                                 
         #i=int(sum(n_iterations)/len(n_iterations))                            
    # print("Average n_ite=" + str(i))                                          
    # print("% of scatter =" + str(np.std(n_iterations)/i))                     
     n=sum(results)/len(results)                                                
     print (n)                                                                  
     print("FIIINISH__________________________________\n")                      
     return n                                                                   
                                                                                
 if optuna_study == XGB_study:                                                  
     study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),       
                                 direction='maximize')                          
     study.optimize(objective, n_trials=1000)                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Trial 82 finished with value: 0.8107728163567985                              
 params_XGB_best= {'lambda': 3.0610042624477543,                                
              'alpha': 4.581902571574289,                                       
              'colsample_bytree': 0.9241969052729379,                           
              'subsample': 0.9527591724824661,                                  
              'learning_rate': 0.06672065863100594,                             
              'n_estimators': 850, #initial value is 651                        
              'max_depth': 5,                                                   
              'min_child_weight': 1,                                            
              'num_parallel_tree': 1}                                           
 #Average n_ite=661                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────

                              ##### 6. Submission                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 pred_XGB_best = (xgb.XGBClassifier(**params_XGB_best).fit(X,y)).predict(test)  
 sample['Transported'] = pred_XGB_best                                          
 sample['Transported']=sample['Transported']>0.5                                
 sample.to_csv('XGB_best.csv', index=False)                                     
                                                                                
