Reading kernel: lazer999/spaceship-titanic-top-6-for-beginners
Cells: 68 (45 code, 23 markdown) | 10220 chars


                        Spaceship Titanic Made Easy 🚢👽                        

The goals of this notebook is to make this data and this competition easy for   
you.                                                                            

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
Description:                                                                    

Welcome to the year 2912, where your data science skills are needed to solve a  
cosmic mystery. We've received a transmission from four lightyears away and     
things aren't looking good.                                                     

The Spaceship Titanic was an interstellar passenger liner launched a month ago. 
With almost 13,000 passengers on board, the vessel set out on its maiden voyage 
transporting emigrants from our solar system to three newly habitable exoplanets
orbiting nearby stars.                                                          

While rounding Alpha Centauri en route to its first destination—the torrid 55   
Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden  
within a dust cloud. Sadly, it met a similar fate as its namesake from 1000     
years before. Though the ship stayed intact, almost half of the passengers were 
transported to an alternate dimension!                                          

To help rescue crews and retrieve the lost passengers, you are challenged to    
predict which passengers were transported by the anomaly using records recovered
from the spaceship’s damaged computer system.                                   

Help save them and change history!                                              

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                           1.Importing Libraries 😀                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
                                                                                
 import numpy as np                                                             
 import pandas as pd                                                            
 import matplotlib.pyplot as plt                                                
 %matplotlib inline                                                             
 import seaborn as sns                                                          
 sns.set(style='darkgrid', font_scale=2)                                        
 import warnings                                                                
 warnings.filterwarnings('ignore')                                              
                                                                                
 # Sklearn                                                                      
 from sklearn.model_selection import train_test_split                           
 from sklearn.metrics import accuracy_score                                     
 from sklearn.preprocessing import LabelEncoder                                 
                                                                                
 # Models                                                                       
 from xgboost import XGBClassifier                                              
 from catboost import CatBoostClassifier                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                            2- Loading the Data 📅                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train = pd.read_csv('../input/spaceship-titanic/train.csv')                 
 df_test = pd.read_csv('../input/spaceship-titanic/test.csv')                   
                                                                                
 df_train.head()                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
Columns Description                                                             

 • PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp 
   where gggg indicates a group the passenger is travelling with and pp is their
   number within the group. People in a group are often family members, but not 
   always.                                                                      
 • HomePlanet - The planet the passenger departed from, typically their planet  
   of permanent residence.                                                      
 • CryoSleep - Indicates whether the passenger elected to be put into suspended 
   animation for the duration of the voyage. Passengers in cryosleep are        
   confined to their cabins.                                                    
 • Cabin - The cabin number where the passenger is staying. Takes the form      
   deck/num/side, where side can be either P for Port or S for Starboard.       
 • Destination - The planet the passenger will be debarking to.                 
 • Age - The age of the passenger.                                              
 • VIP - Whether the passenger has paid for special VIP service during the      
   voyage.                                                                      
 • RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has 
   billed at each of the Spaceship Titanic's many luxury amenities.             
 • Name - The first and last names of the passenger.                            
 • Transported - Whether the passenger was transported to another dimension.    
   This is the target, the column you are trying to predict.                    

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                             3- Let's Explore 👓                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 r1,c1 = df_train.shape                                                         
 print('The training data has {} rows and {} columns'.format(r1,c1))            
 r2,c2 = df_test.shape                                                          
 print('The validation data has {} rows and {} columns'.format(r2,c2))          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train.info()                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train.describe()                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_test.describe()                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                             3.B Missing values 🤔                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # To see the quantity of null vaues in all the columns.                        
 # c1 stands for the number of columns in the training data.                    
                                                                                
                                                                                
 print('MISSING VALUES IN TRAINING DATASET:')                                   
 print(df_train.isna().sum().nlargest(c1))                                      
 print('')                                                                      
 print('MISSING VALUES IN VALIDATION DATASET:')                                 
 print(df_test.isna().sum().nlargest(c2))                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train.set_index('PassengerId',inplace=True)                                 
 df_test.set_index('PassengerId',inplace=True)                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                            3.C Null Replacement 🤩                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] =          
 df_train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)  
 df_test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] =           
 df_test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)   
                                                                                
 df_train['Age'] =df_train['Age'].fillna(df_train['Age'].median())              
 df_test['Age'] =df_test['Age'].fillna(df_test['Age'].median())                 
                                                                                
 df_train['VIP'] =df_train['VIP'].fillna(False)                                 
 df_test['VIP'] =df_test['VIP'].fillna(False)                                   
                                                                                
 df_train['HomePlanet'] =df_train['HomePlanet'].fillna('Mars')                  
 df_test['HomePlanet'] =df_test['HomePlanet'].fillna('Mars')                    
                                                                                
 df_train['Destination']=df_train['Destination'].fillna("PSO J318.5-22")        
 df_test['Destination']=df_test['Destination'].fillna("PSO J318.5-22")          
                                                                                
 df_train['CryoSleep'] =df_train['CryoSleep'].fillna(False)                     
 df_test['CryoSleep'] =df_test['CryoSleep'].fillna(False)                       
                                                                                
 df_train['Cabin'] =df_train['Cabin'].fillna('T/0/P')                           
 df_test['Cabin'] =df_test['Cabin'].fillna('T/0/P')                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      4.Exploration and Visualization 🤠                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.figure(figsize=(15,18))                                                    
 sns.heatmap(df_train.corr(), annot=True);                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.pie(df_train.Transported.value_counts(), shadow=True, explode=[.1,.1],     
 autopct='%.1f%%')                                                              
 plt.title('Transported ', size=18)                                             
 plt.legend(['False', 'True'], loc='best', fontsize=12)                         
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_train.Transported);                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_train.HomePlanet,hue=df_train.Transported);                   
 # Dude, Europa is gone                                                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_train.VIP,hue=df_train.Transported);                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_train.CryoSleep,hue=df_train.Transported);                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
The people in CryoSleep are majorly Transported                                 

Do not sleep during travel alright ☠                                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_train.Destination,hue=df_train.Transported)                   
 plt.xticks(rotation=90);                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.boxplot(y=df_train.Age,x=df_train.Transported);                            
 #Age is not affecting much. But I have a plan XD                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                          4.B Splitting Cabin Column ⚔                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Cabin - The cabin number where the passenger is staying. Takes the form      
 deck/num/side, where side can be either P for Port or S for Starboard.         
 df_train[['Deck','Num','Side']] = df_train.Cabin.str.split('/',expand=True)    
 df_test[['Deck','Num','Side']] = df_test.Cabin.str.split('/',expand=True)      
                                                                                

────────────────────────────────────────────────────────────────────────────────
Let's look into them                                                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_train.Deck,hue=df_train.Transported);                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.figure(figsize=(10,5))                                                     
 sns.histplot(data=df_train, x='Num', hue='Transported',bins=14);               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_train.Side,hue=df_train.Transported);                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(df_test.Side);                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                           5.Feature Engineering 🥱                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train['total_spent']= df_train['RoomService']+ df_train['FoodCourt']+       
 df_train['ShoppingMall']+ df_train['Spa']+ df_train['VRDeck']                  
 df_test['total_spent']=df_test['RoomService']+df_test['FoodCourt']+df_test['Sh 
 pingMall']+df_test['Spa']+df_test['VRDeck']                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train['AgeGroup'] = 0                                                       
 for i in range(6):                                                             
     df_train.loc[(df_train.Age >= 10*i) & (df_train.Age < 10*(i + 1)),         
 'AgeGroup'] = i                                                                
 # Same for test data                                                           
 df_test['AgeGroup'] = 0                                                        
 for i in range(6):                                                             
     df_test.loc[(df_test.Age >= 10*i) & (df_test.Age < 10*(i + 1)), 'AgeGroup' 
 = i                                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sns.countplot(y=df_train['AgeGroup'],hue=df_train['Transported']);             
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      6. Pre processing for Modeling 🤖                       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                  6.A Encoding                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from sklearn.preprocessing import LabelEncoder                                 
                                                                                
 categorical_cols=                                                              
 ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side','Num']             
 for i in categorical_cols:                                                     
     print(i)                                                                   
     le=LabelEncoder()                                                          
     arr=np.concatenate((df_train[i], df_test[i])).astype(str)                  
     le.fit(arr)                                                                
     df_train[i]=le.transform(df_train[i].astype(str))                          
     df_test[i]=le.transform(df_test[i].astype(str))                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train.head()                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                              6.B Dropping Columns                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train= df_train.drop(['Name','Cabin'],axis=1)                               
 df_test= df_test.drop(['Name','Cabin'],axis=1)                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.figure(figsize=(30,10))                                                    
 sns.heatmap(df_train.corr(), annot=True);                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_train['Transported']=df_train['Transported'].replace({True:1,False:0})      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X=df_train.drop('Transported',axis=1)                                          
 y = df_train['Transported']                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X.columns                                                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                             6.C Splitting Columns                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25,        
 random_state=0)                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                             Modeling - Finally 😎                              

────────────────────────────────────────────────────────────────────────────────
Cat Boost 😾                                                                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
                                                                                
 from catboost import CatBoostClassifier                                        
 model=CatBoostClassifier(iterations=1500,                                      
                          eval_metric='Accuracy',                               
                         verbose=0)                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 model.fit(X_train,y_train)                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 pred_y=model.predict(X_val)                                                    
                                                                                
 pred=model.predict(X_train)                                                    
                                                                                
 print(accuracy_score(y_train.values,pred))                                     
 print(accuracy_score(y_val.values,pred_y))                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from sklearn.model_selection import GridSearchCV                               
 gcv=GridSearchCV(CatBoostClassifier(),param_grid={'iterations':                
 range(200,2000,200), 'eval_metric': ['Accuracy'],'verbose':[0]},cv=3)          
 gcv.fit(X_train,y_train)                                                       
 pred_y=gcv.predict(X_val)                                                      
                                                                                
 pred=gcv.predict(X_train)                                                      
                                                                                
 print(accuracy_score(y_train.values,pred))                                     
 print(accuracy_score(y_val.values,pred_y))                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
Gradient Boosting                                                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from sklearn.ensemble import GradientBoostingClassifier                        
 gb=GradientBoostingClassifier(random_state=1,n_estimators=250,learning_rate=0. 
 ,max_depth=3)                                                                  
 gb.fit(X_train,y_train)                                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 pred_y=gb.predict(X_val)                                                       
 pred=gb.predict(X_train)                                                       
                                                                                
 print(accuracy_score(y_train.values,pred))                                     
 print(accuracy_score(y_val.values,pred_y))                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                 Cat wins!!! 😼                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # lets re fit the model on the entire data                                     
 gcv.fit(X,y)                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #for i,z in zip(X.columns,gcv.get_feature_importance()):                       
     #print('importance of',i,'is',z)                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                       7. Prediction and Submission 😎                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 y_pred = gcv.predict(df_test)                                                  
                                                                                
 sub=pd.DataFrame({'Transported':y_pred.astype(bool)},index=df_test.index)      
                                                                                
 sub.head()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sub.to_csv('submission')                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────


                              Thanks for reading:)                              



                       Upvote! and Leave some suggestions                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 140/24458100                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
                                                                                
                                                                                
