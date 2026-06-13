Reading kernel: odins0n/spaceship-titanic-eda-27-different-models
Cells: 97 (32 code, 65 markdown) | 24420 chars

Created by Sanskar Hasija                                                       

🚀Spaceship Titanic -📊EDA + 27 different models📈                              

24 February 2022                                                                

────────────────────────────────────────────────────────────────────────────────


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              🚀SPACESHIP TITANIC -📊EDA + 27 DIFFERENT MODELS📈              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛




           If you find this notebook useful, support with an upvote👍           

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              Table of Contents                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛



                                                                                

 • 1. Introduction                                                              
 • 2. Imports                                                                   
 • 3. Data Loading and Preperation                                              
    • 3.1 Exploring Train Data                                                  
    • 3.2 Exploring Test Data                                                   
    • 3.3 Submission File                                                       
 • 4. EDA                                                                       
    • 4.1 Overview of Data                                                      
    • 4.2 Null Value Distribution                                               
    • 4.3 Continuos and Categorical Data Distribution                           
    • 4.4 Feature Distribution of Continous Features                            
    • 4.5 Feature Distribution of Categorical Features                          
    • 4.6 Target Distribution                                                   
    • 4.7 Correlation Matrix                                                    
 • 5. Data Pre-Processing                                                       
 • 6. Modeling                                                                  
    • 6.1 27 Different Classifiers ( LAZY PREDICT )                             
    • 6.2 LGBM Classifier                                                       
 • 7. Submission                                                                

────────────────────────────────────────────────────────────────────────────────


                                                                                





┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                 Introduction                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
🌆 header.png?t=2022-02-11-21-53-06                                             

────────────────────────────────────────────────────────────────────────────────
The competition is organised by Kaggle and is in the GettingStarted Prediction  
Competition series.                                                             

In this competition, you are supposed to predict predict which passengers were  
transported by the anomaly using records recovered from the spaceship’s damaged 
computer system.                                                                

Submissions are evaluated on Classification Accuracy.                           

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                





┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                   Imports                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────


                            Installing LazyPredict :                            

────────────────────────────────────────────────────────────────────────────────
Check the offical Documentation of LazyPredict here :                           
https://lazypredict.readthedocs.io                                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 from IPython.display import clear_output                                       
 !pip3 install -U lazypredict                                                   
 !pip3 install -U pandas #Upgrading pandas                                      
                                                                                
 clear_output()                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 import numpy as np                                                             
 import pandas as pd                                                            
 import seaborn as sns                                                          
 import plotly.express as px                                                    
 import matplotlib.pyplot as plt                                                
 import plotly.graph_objects as go                                              
 from plotly.subplots import make_subplots                                      
                                                                                
                                                                                
                                                                                
 from sklearn.impute import SimpleImputer                                       
 from sklearn.metrics import accuracy_score                                     
 from sklearn.preprocessing import LabelEncoder                                 
 from sklearn.model_selection import StratifiedKFold, train_test_split          
                                                                                
                                                                                
 from lightgbm import LGBMClassifier                                            
 import lazypredict                                                             
 from lazypredict.Supervised import LazyClassifier                              
                                                                                
 import time                                                                    
 import warnings                                                                
 warnings.filterwarnings('ignore')                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                





┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                         Data Loading and Preparation                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train = pd.read_csv("../input/spaceship-titanic/train.csv")                    
 test = pd.read_csv("../input/spaceship-titanic/test.csv")                      
 submission = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")   
                                                                                
 RANDOM_STATE = 12                                                              
 FOLDS = 5                                                                      
 STRATEGY = 'median'                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────



                             Column Descriptions  :                             

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


                                                                                




                             Exploring Train Data :                             

────────────────────────────────────────────────────────────────────────────────















































 •  There are total of 14 columns and 8693 rows in train data.                  
 •  Train data contains 119378 observation with 2324  missing values.           
 •  All 12 feature columns have missing values in them with CryoSleep having    
   highest missing values (217)                                                 
 •  Transported is the target variable which is only available in the train     
   dataset.                                                                     


────────────────────────────────────────────────────────────────────────────────


                           Quick view of Train Data :                           

────────────────────────────────────────────────────────────────────────────────
Below are the first 5 rows of train dataset:                                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train.head()                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f'\033[94mNumber of rows in train data: {train.shape[0]}')               
 print(f'\033[94mNumber of columns in train data: {train.shape[1]}')            
 print(f'\033[94mNumber of values in train data: {train.count().sum()}')        
 print(f'\033[94mNumber missing values in train data: {sum(train.isna().sum())} 
                                                                                

────────────────────────────────────────────────────────────────────────────────


                          Column Wise missing values :                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f'\033[94m')                                                             
 print(train.isna().sum().sort_values(ascending = False))                       
                                                                                

────────────────────────────────────────────────────────────────────────────────


                       Basic statistics of training data :                      

────────────────────────────────────────────────────────────────────────────────
Below is the basic statistics for each variables which contain information on   
count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and
maximum.                                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train.describe()                                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────


                                                                                




                               Exploring Test Data                              

────────────────────────────────────────────────────────────────────────────────




































 •  There are total of 13 columns and 4277 rows in test data.                   
 •  Train data contains 54484 observation with 1117  missing values.            
 •  All 12 feature columns have missing values in them with FoodCourt having    
   highest missing values (106)                                                 


────────────────────────────────────────────────────────────────────────────────


                             Quick view of Test Data                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 test.head()                                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f'\033[94mNumber of rows in test data: {test.shape[0]}')                 
 print(f'\033[94mNumber of columns in test data: {test.shape[1]}')              
 print(f'\033[94mNumber of values in train data: {test.count().sum()}')         
 print(f'\033[94mNo of rows with missing values  in test data:                  
 {sum(test.isna().sum())}')                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────


                           Column Wise missing values                           

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f'\033[94m')                                                             
 print((test.isna().sum().sort_values(ascending = False)))                      
                                                                                

────────────────────────────────────────────────────────────────────────────────


                          Basic statistics of test data                         

────────────────────────────────────────────────────────────────────────────────
Below is the basic statistics for each variables which contain information on   
count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and
maximum.                                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 test.describe()                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────


                                                                                




                                 Submission File                                

────────────────────────────────────────────────────────────────────────────────


                          Quick view of Submission File                         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 submission.head()                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                





┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                      EDA                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────


                                                                                




                                Overview of Data                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train.drop(["PassengerId"] , axis = 1 , inplace = True)                        
 test.drop(["PassengerId"] , axis = 1 , inplace = True)                         
 TARGET = 'Transported'                                                         
 FEATURES = [col for col in train.columns if col != TARGET]                     
 RANDOM_STATE = 12                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train.iloc[:, :-1].describe().T.sort_values(by='std' , ascending = False)\     
                      .style.background_gradient(cmap='GnBu')\                  
                      .bar(subset=["max"], color='#BB0000')\                    
                      .bar(subset=["mean",], color='green')                     
                                                                                

────────────────────────────────────────────────────────────────────────────────


                                                                                




                             Null Value Distribution                            

────────────────────────────────────────────────────────────────────────────────









































 •  The maximum of missing value in an row is 3 and the lowest is no missing    
   value                                                                        
 •  Interestingly, the missing value distribution (row basis) is quite the same 
   between train and test dataset.                                              
 •  There are around 76% of the observations (row basis) that has no missing    
   values.                                                                      
 •  Rest 24% of the observations (row basis) that have 1 to 3  missing values   


────────────────────────────────────────────────────────────────────────────────


                                                                                



                      Column wise Null Value Distribution                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 test_null = pd.DataFrame(test.isna().sum())                                    
 test_null = test_null.sort_values(by = 0 ,ascending = False)                   
 train_null = pd.DataFrame(train.isna().sum())                                  
 train_null = train_null.sort_values(by = 0 ,ascending = False)[:-1]            
                                                                                
                                                                                
 fig = make_subplots(rows=1,                                                    
                     cols=2,                                                    
                     column_titles = ["Train Data", "Test Data"] ,              
                     x_title="Missing Values")                                  
                                                                                
 fig.add_trace(go.Bar(x=train_null[0],                                          
                      y=train_null.index,                                       
                      orientation="h",                                          
                     marker=dict(color=[n for n in range(12)],                  
                                 line_color='rgb(0,0,0)' ,                      
                                 line_width = 2,                                
                                 coloraxis="coloraxis")),                       
               1, 1)                                                            
 fig.add_trace(go.Bar(x=test_null[0],                                           
                      y=test_null.index,                                        
                      orientation="h",                                          
                     marker=dict(color=[n for n in range(12)],                  
                                 line_color='rgb(0,0,0)',                       
                                 line_width = 2,                                
                                 coloraxis="coloraxis")),                       
               1, 2)                                                            
                                                                                
 fig.update_layout(showlegend=False, title_text="Column wise Null Value         
 Distribution", title_x=0.5)                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────


                                                                                



                        Row wise Null Value Distribution                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 missing_train_row = train.isna().sum(axis=1)                                   
 missing_train_row =                                                            
 pd.DataFrame(missing_train_row.value_counts()/train.shape[0]).reset_index()    
 missing_test_row = test.isna().sum(axis=1)                                     
 missing_test_row =                                                             
 pd.DataFrame(missing_test_row.value_counts()/test.shape[0]).reset_index()      
 missing_train_row.columns = ['no', 'count']                                    
 missing_test_row.columns = ['no', 'count']                                     
 missing_train_row["count"] = missing_train_row["count"]*100                    
 missing_test_row["count"] = missing_test_row["count"]*100                      
                                                                                
                                                                                
 fig = make_subplots(rows=1,                                                    
                     cols=2,                                                    
                     column_titles = ["Train Data", "Test Data"] ,              
                     x_title="Missing Values",)                                 
                                                                                
 fig.add_trace(go.Bar(x=missing_train_row["no"],                                
                      y=missing_train_row["count"]  ,                           
                     marker=dict(color=[n for n in range(4)],                   
                                 line_color='rgb(0,0,0)' ,                      
                                 line_width = 3                                 
                                 ,coloraxis="coloraxis")),                      
               1, 1)                                                            
 fig.add_trace(go.Bar(x= missing_test_row["no"],                                
                      y=missing_test_row["count"],                              
                     marker=dict(color=[n for n in range(4)],                   
                                 line_color='rgb(0,0,0)',                       
                                 line_width = 3 ,                               
                                 coloraxis="coloraxis")),                       
               1, 2)                                                            
 fig.update_layout(showlegend=False, title_text="Row wise Null Value            
 Distribution", title_x=0.5)                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────


                     Dealing with missing value (reference)                     

Some references on how to deal with missing value:                              

 • Missing Values by Alexis Cook                                                
 • Data Cleaning Challenge: Handling missing values by Rachael Tatman           
 • A Guide to Handling Missing values in Python  by Parul Pandey                

Some models that have capability to handle missing value by default are:        

 • XGBoost: https://xgboost.readthedocs.io/en/latest/faq.html                   
 • LightGBM: https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html     
 • Catboost:                                                                    
   https://catboost.ai/docs/concepts/algorithm-missing-values-processing.html   

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                




                  Continuos and Categorical Data Distribution                   

────────────────────────────────────────────────────────────────────────────────


















































 •  Out of 12 features 6 features are continous, 2 features are text data and 4 
   features are categorical.                                                    
 • HomePlanet and Destination have 3 differnt unique values.                    
 • CryoSleep and VIP are bool features                                          


────────────────────────────────────────────────────────────────────────────────
                                                                                
 df = pd.concat([train[FEATURES], test[FEATURES]], axis=0)                      
 text_features = ["Cabin", "Name"]                                              
 cat_features = [col for col in FEATURES if df[col].nunique() < 25 and col not  
 text_features ]                                                                
 cont_features = [col for col in FEATURES if df[col].nunique() >= 25 and col no 
 in text_features ]                                                             
                                                                                
 del df                                                                         
 print(f'\033[94mTotal number of features: {len(FEATURES)}')                    
 print(f'\033[94mNumber of categorical features: {len(cat_features)}')          
 print(f'\033[94mNumber of continuos features: {len(cont_features)}')           
 print(f'\033[94mNumber of text features: {len(text_features)}')                
                                                                                
 labels=['Categorical', 'Continuos', "Text"]                                    
 values= [len(cat_features), len(cont_features), len(text_features)]            
 colors = ['#DE3163', '#58D68D']                                                
                                                                                
 fig = go.Figure(data=[go.Pie(                                                  
     labels=labels,                                                             
     values=values, pull=[0.1, 0, 0 ],                                          
     marker=dict(colors=colors,                                                 
                 line=dict(color='#000000',                                     
                           width=2))                                            
 )])                                                                            
 fig.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                




                   Feature Distribution of Continous Features                   

────────────────────────────────────────────────────────────────────────────────


                               Distribution of Age                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_age = train.copy()                                                       
 test_age = test.copy()                                                         
 train_age["type"] = "Train"                                                    
 test_age["type"] = "Test"                                                      
 ageDf = pd.concat([train_age, test_age])                                       
 fig = px.histogram(data_frame = ageDf,                                         
                    x="Age",                                                    
                    color= "type",                                              
                    color_discrete_sequence =  ['#58D68D','#DE3163'],           
                    marginal="box",                                             
                    nbins= 100,                                                 
                     template="plotly_white"                                    
                 )                                                              
 fig.update_layout(title = "Distribution of Age" , title_x = 0.5)               
 fig.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                




                   Feature Distribution of Categorical Features                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 if len(cat_features) == 0 :                                                    
     print("No Categorical features")                                           
 else:                                                                          
     ncols = 2                                                                  
     nrows = 2                                                                  
                                                                                
     fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))                   
     for r in range(nrows):                                                     
         for c in range(ncols):                                                 
             col = cat_features[r*ncols+c]                                      
             sns.countplot(train[col],ax = axes[r,c] ,palette = "viridis",      
 label='Train data')                                                            
             sns.countplot(test[col],ax = axes[r,c] ,palette = "magma",         
 label='Test data')                                                             
             axes[r,c].legend()                                                 
             axes[r,c].set_ylabel('')                                           
             axes[r,c].set_xlabel(col, fontsize=20)                             
             axes[r,c].tick_params(labelsize=10, width=0.5)                     
             axes[r,c].xaxis.offsetText.set_fontsize(4)                         
             axes[r,c].yaxis.offsetText.set_fontsize(4)                         
     plt.show()                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                




                               Target Distribution                              

────────────────────────────────────────────────────────────────────────────────















 • There are two target values - 0 and 1.                                       
 • Both the target values are almost equally distributed.                       


────────────────────────────────────────────────────────────────────────────────
                                                                                
 target_df = pd.DataFrame(train[TARGET].value_counts()).reset_index()           
 target_df.columns = [TARGET, 'count']                                          
 fig = px.bar(data_frame =target_df,                                            
              x = TARGET,                                                       
              y = 'count'                                                       
             )                                                                  
 fig.update_traces(marker_color =['#58D68D','#DE3163'],                         
                   marker_line_color='rgb(0,0,0)',                              
                   marker_line_width=2,)                                        
 fig.update_layout(title = "Target Distribution",                               
                   template = "plotly_white",                                   
                   title_x = 0.5)                                               
 print("\033[94mPercentage of Transported = 0: {:.2f}                           
 %".format(target_df["count"][0] *100 / train.shape[0]))                        
 print("\033[94mPercentage of Transported = 1: {:.2f}                           
 %".format(target_df["count"][1]* 100 / train.shape[0]))                        
 fig.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                




                                Correlation matrix                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig = px.imshow(train.corr() ,text_auto=True, aspect="auto" ,                  
 color_continuous_scale = "viridis")                                            
 fig.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                





┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                             Data Pre-Processing                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────


                             Imputing Missing Values                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"            
 ,"RoomService"]                                                                
 imputer = SimpleImputer(strategy=STRATEGY )                                    
 imputer.fit(train[imputer_cols])                                               
 train[imputer_cols] = imputer.transform(train[imputer_cols])                   
 test[imputer_cols] = imputer.transform(test[imputer_cols])                     
 train["HomePlanet"].fillna('Z', inplace=True)                                  
 test["HomePlanet"].fillna('Z', inplace=True)                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────


                         Encoding Categorical Features                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 label_cols = ["HomePlanet", "CryoSleep","Cabin", "Destination" ,"VIP"]         
 def label_encoder(train,test,columns):                                         
     for col in columns:                                                        
         train[col] = train[col].astype(str)                                    
         test[col] = test[col].astype(str)                                      
         train[col] = LabelEncoder().fit_transform(train[col])                  
         test[col] =  LabelEncoder().fit_transform(test[col])                   
     return train, test                                                         
                                                                                
 train ,test = label_encoder(train,test ,label_cols)                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train.drop(["Name" ,"Cabin"] , axis = 1 ,inplace = True)                       
 test.drop(["Name" ,"Cabin"] , axis = 1 ,inplace = True)                        
 X = train.drop(TARGET , axis =1 )                                              
 y = train[TARGET]                                                              
 X_train , X_test , y_train , y_test = train_test_split(X ,                     
                                                        y,                      
                                                        random_state = 12 ,     
                                                        test_size =0.33)        
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                





┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                   Modeling                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────


                                                                                




                   27 Different Classifiers ( LAZY PREDICT ) :                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 clf = LazyClassifier(verbose=0,                                                
                      ignore_warnings=True,                                     
                      custom_metric=None,                                       
                      predictions=False,                                        
                      random_state=12,                                          
                      classifiers='all')                                        
                                                                                
 models, predictions = clf.fit(X_train , X_test , y_train , y_test)             
 clear_output()                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────


                                 TOP 15 Models                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 models[:15]                                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────


                              Visualizing Results                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 line = px.line(data_frame= models ,y =["Accuracy"] , markers = True)           
 line.update_xaxes(title="Model",                                               
               rangeslider_visible = False)                                     
 line.update_yaxes(title = "Accuracy")                                          
 line.update_traces(line_color="red")                                           
 line.update_layout(showlegend = True,                                          
     title = {                                                                  
         'text': 'Accuracy vs Model',                                           
         'y':0.94,                                                              
         'x':0.5,                                                               
         'xanchor': 'center',                                                   
         'yanchor': 'top'})                                                     
                                                                                
 line.show()                                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 line = px.line(data_frame= models ,y =["ROC AUC" , "F1 Score"] , markers = Tru 
 line.update_xaxes(title="Model",                                               
               rangeslider_visible = False)                                     
 line.update_yaxes(title = "ROC AUC SCORE")                                     
 line.update_layout(showlegend = True,                                          
     title = {                                                                  
         'text': 'ROC AUC and F1 Score vs Model',                               
         'y':0.94,                                                              
         'x':0.5,                                                               
         'xanchor': 'center',                                                   
         'yanchor': 'top'})                                                     
                                                                                
 line.show()                                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 line = px.line(data_frame= models ,y =["Time Taken"] , markers = True)         
 line.update_xaxes(title="Model",                                               
               rangeslider_visible = False)                                     
 line.update_yaxes(title = "Time(s)")                                           
 line.update_traces(line_color="purple")                                        
 line.update_layout(showlegend = True,                                          
     title = {                                                                  
         'text': 'TIME TAKEN vs Model',                                         
         'y':0.94,                                                              
         'x':0.5,                                                               
         'xanchor': 'center',                                                   
         'yanchor': 'top'})                                                     
                                                                                
 line.show()                                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────


                                                                                




                           LGBM Classifier(5 FOLDS)  :                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 lgb_params = {                                                                 
     'objective' : 'binary',                                                    
     'n_estimators' :50,                                                        
     'learning_rate' : 0.08                                                     
 }                                                                              
                                                                                
 lgb_predictions = 0                                                            
 lgb_scores = []                                                                
 lgb_fimp = []                                                                  
 LGBM_FEATURES = list(train.columns)[:-1]                                       
 skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE) 
 for fold, (train_idx, valid_idx) in enumerate(skf.split(train[LGBM_FEATURES],  
 train[TARGET])):                                                               
     print(f'\033[94m')                                                         
     print(10*"=", f"Fold={fold+1}", 10*"=")                                    
     start_time = time.time()                                                   
                                                                                
     X_train, X_valid = train.iloc[train_idx][LGBM_FEATURES],                   
 train.iloc[valid_idx][LGBM_FEATURES]                                           
     y_train , y_valid = train[TARGET].iloc[train_idx] ,                        
 train[TARGET].iloc[valid_idx]                                                  
                                                                                
     model = LGBMClassifier(**lgb_params)                                       
     model.fit(X_train, y_train,verbose=0)                                      
                                                                                
     preds_valid = model.predict(X_valid)                                       
     acc = accuracy_score(y_valid,  preds_valid)                                
     lgb_scores.append(acc)                                                     
     run_time = time.time() - start_time                                        
                                                                                
     fim = pd.DataFrame(index=LGBM_FEATURES,                                    
                  data=model.feature_importances_,                              
                  columns=[f'{fold}_importance'])                               
     lgb_fimp.append(fim)                                                       
                                                                                
     print(f"Fold={fold+1}, Accuracy score: {acc:.2f}%, Run Time:               
 {run_time:.2f}s")                                                              
     test_preds = model.predict(test[LGBM_FEATURES])                            
     lgb_predictions += test_preds/FOLDS                                        
 print("")                                                                      
 print("Mean Accuracy :", np.mean(lgb_scores))                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────


                               Feature Importance                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 lgbm_fis_df = pd.concat(lgb_fimp, axis=1).head(15)                             
 lgbm_fis_df.sort_values('1_importance').plot(kind='barh', figsize=(15, 10),    
                                        title='Feature Importance Across Folds' 
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────


                                                                                





┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Submission                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────


                          LGBM Classifier Submission :                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 submission[TARGET] = lgb_predictions.astype("bool")                            
 submission.to_csv("submission.csv",index=False)                                
 submission.head()                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────


⬆️Back to Table of Contents ⬆️                                                    

────────────────────────────────────────────────────────────────────────────────



                            Thank you for reading🙂                             



      If you have any feedback or find anything wrong, please let me know!      
