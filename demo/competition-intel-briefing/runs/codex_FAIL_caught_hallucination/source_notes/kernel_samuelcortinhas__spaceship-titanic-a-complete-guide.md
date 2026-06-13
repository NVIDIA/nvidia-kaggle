Reading kernel: samuelcortinhas/spaceship-titanic-a-complete-guide
Cells: 203 (82 code, 121 markdown) | 53502 chars

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                 Introduction                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
Welcome to this comprehensive guide on binary classification with the Spaceship 
Titanic dataset. The objective is to predict whether a passenger was transported
to an alternate dimension during the Spaceship Titanic's collision with a       
spacetime anomaly.                                                              

We will cover:                                                                  

 • Exploratory Data Analysis                                                    
 • Feature Engineering                                                          
 • Data Cleaning                                                                
 • Encoding, Scaling and Preprocessing                                          
 • Training Machine Learning Models                                             
 • Cross Validation and Ensembling Predictions                                  

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Libraries                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Core                                                                         
 import numpy as np                                                             
 import pandas as pd                                                            
 import matplotlib.pyplot as plt                                                
 %matplotlib inline                                                             
 import seaborn as sns                                                          
 sns.set(style='darkgrid', font_scale=1.4)                                      
 from imblearn.over_sampling import SMOTE                                       
 import itertools                                                               
 import warnings                                                                
 warnings.filterwarnings('ignore')                                              
 import plotly.express as px                                                    
 import time                                                                    
                                                                                
 # Sklearn                                                                      
 from sklearn.model_selection import train_test_split, GridSearchCV,            
 RandomizedSearchCV, StratifiedKFold                                            
 from sklearn.metrics import accuracy_score, confusion_matrix, recall_score,    
 precision_score, f1_score                                                      
 from sklearn.metrics import roc_auc_score, plot_confusion_matrix,              
 plot_roc_curve, roc_curve                                                      
 from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, 
 LabelEncoder                                                                   
 from sklearn.feature_selection import mutual_info_classif                      
 from sklearn.decomposition import PCA                                          
 from sklearn.compose import ColumnTransformer                                  
 from sklearn.pipeline import Pipeline                                          
 from sklearn.impute import SimpleImputer                                       
 import eli5                                                                    
 from eli5.sklearn import PermutationImportance                                 
 from sklearn.utils import resample                                             
                                                                                
 # Models                                                                       
 from sklearn.linear_model import LinearRegression, LogisticRegression          
 from sklearn.neighbors import KNeighborsClassifier                             
 from sklearn.svm import SVC                                                    
 from sklearn.tree import DecisionTreeClassifier                                
 from sklearn.ensemble import RandomForestClassifier                            
 from xgboost import XGBClassifier                                              
 from lightgbm import LGBMClassifier                                            
 from catboost import CatBoostClassifier                                        
 from sklearn.naive_bayes import GaussianNB                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                     Data                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
Load data                                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Save to df                                                                   
 train = pd.read_csv('../input/spaceship-titanic/train.csv')                    
 test = pd.read_csv('../input/spaceship-titanic/test.csv')                      
                                                                                
 # Shape and preview                                                            
 print('Train set shape:', train.shape)                                         
 print('Test set shape:', test.shape)                                           
 train.head()                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Feature descriptions:                                                           

▌  • PassengerId - A unique Id for each passenger. Each Id takes the form     
▌    gggg_pp where gggg indicates a group the passenger is travelling with and
▌    pp is their number within the group. People in a group are often family  
▌    members, but not always.                                                 
▌  • HomePlanet - The planet the passenger departed from, typically their     
▌    planet of permanent residence.                                           
▌  • CryoSleep - Indicates whether the passenger elected to be put into       
▌    suspended animation for the duration of the voyage. Passengers in        
▌    cryosleep are confined to their cabins.                                  
▌  • Cabin - The cabin number where the passenger is staying. Takes the form  
▌    deck/num/side, where side can be either P for Port or S for Starboard.   
▌  • Destination - The planet the passenger will be debarking to.             
▌  • Age - The age of the passenger.                                          
▌  • VIP - Whether the passenger has paid for special VIP service during the  
▌    voyage.                                                                  
▌  • RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger 
▌    has billed at each of the Spaceship Titanic's many luxury amenities.     
▌  • Name - The first and last names of the passenger.                        
▌  • Transported - Whether the passenger was transported to another dimension.
▌    This is the target, the column you are trying to predict.                

────────────────────────────────────────────────────────────────────────────────
Missing values                                                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print('TRAIN SET MISSING VALUES:')                                             
 print(train.isna().sum())                                                      
 print('')                                                                      
 print('TEST SET MISSING VALUES:')                                              
 print(test.isna().sum())                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
Almost every feature has missing values! How we deal with these values will be  
very important.                                                                 

────────────────────────────────────────────────────────────────────────────────
Duplicates                                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f'Duplicates in train set: {train.duplicated().sum()},                   
 ({np.round(100*train.duplicated().sum()/len(train),1)}%)')                     
 print('')                                                                      
 print(f'Duplicates in test set: {test.duplicated().sum()},                     
 ({np.round(100*test.duplicated().sum()/len(test),1)}%)')                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
Cardinality of features                                                         

────────────────────────────────────────────────────────────────────────────────
There are 6 continuous features, 4 categorical features (excluding the target)  
and 3 descriptive/qualitative features.                                         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train.nunique()                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
Data types                                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train.dtypes                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
We will need to transform the data to be numeric (int64 or float64) so that we  
can train machine learning models. These models (in general) don't work on text.

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                     EDA                                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
Let us explore the dataset to gain insights.                                    

────────────────────────────────────────────────────────────────────────────────
Target distribution                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Figure size                                                                  
 plt.figure(figsize=(6,6))                                                      
                                                                                
 # Pie plot                                                                     
 train['Transported'].value_counts().plot.pie(explode=[0.1,0.1],                
 autopct='%1.1f%%', shadow=True, textprops={'fontsize':16}).set_title("Target   
 distribution")                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
The target is highly balanced, so we luckily don't have to consider techniques  
like under/over-sampling.                                                       

────────────────────────────────────────────────────────────────────────────────
Continuous features                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Figure size                                                                  
 plt.figure(figsize=(10,4))                                                     
                                                                                
 # Histogram                                                                    
 sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)     
                                                                                
 # Aesthetics                                                                   
 plt.title('Age distribution')                                                  
 plt.xlabel('Age (years)')                                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
Notes:                                                                          

 • 0-18 year olds were more likely to be transported than not.                  
 • 18-25 year olds were less likely to be transported than not.                 
 • Over 25 year olds were about equally likely to be transported than not.      

Insight:                                                                        

 • Create a new feature that indicates whether the passanger is a child,        
   adolescent or adult.                                                         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Expenditure features                                                         
 exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']        
                                                                                
 # Plot expenditure features                                                    
 fig=plt.figure(figsize=(10,20))                                                
 for i, var_name in enumerate(exp_feats):                                       
     # Left plot                                                                
     ax=fig.add_subplot(5,2,2*i+1)                                              
     sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False,          
 hue='Transported')                                                             
     ax.set_title(var_name)                                                     
                                                                                
     # Right plot (truncated)                                                   
     ax=fig.add_subplot(5,2,2*i+2)                                              
     sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True,           
 hue='Transported')                                                             
     plt.ylim([0,100])                                                          
     ax.set_title(var_name)                                                     
 fig.tight_layout()  # Improves appearance a bit                                
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
Notes:                                                                          

 • Most people don't spend any money (as we can see on the left).               
 • The distribution of spending decays exponentially (as we can see on the      
   right).                                                                      
 • There are a small number of outliers.                                        
 • People who were transported tended to spend less.                            
 • RoomService, Spa and VRDeck have different distributions to FoodCourt and    
   ShoppingMall - we can think of this as luxury vs essential amenities.        

Insight:                                                                        

 • Create a new feature that tracks the total expenditure across all 5          
   amenities.                                                                   
 • Create a binary feature to indicate if the person has not spent anything.    
   (i.e. total expenditure is 0).                                               
 • Take the log transform to reduce skew.                                       

────────────────────────────────────────────────────────────────────────────────
Categorical features                                                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Categorical features                                                         
 cat_feats=['HomePlanet', 'CryoSleep', 'Destination', 'VIP']                    
                                                                                
 # Plot categorical features                                                    
 fig=plt.figure(figsize=(10,16))                                                
 for i, var_name in enumerate(cat_feats):                                       
     ax=fig.add_subplot(4,1,i+1)                                                
     sns.countplot(data=train, x=var_name, axes=ax, hue='Transported')          
     ax.set_title(var_name)                                                     
 fig.tight_layout()  # Improves appearance a bit                                
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
Notes:                                                                          

 • VIP does not appear to be a useful feature; the target split is more or less 
   equal.                                                                       
 • CryoSleep appears the be a very useful feature in contrast.                  

Insights:                                                                       

 • We might consider dropping the VIP column to prevent overfitting.            

────────────────────────────────────────────────────────────────────────────────
Qualitative features                                                            

────────────────────────────────────────────────────────────────────────────────
We can't plot this data (yet). We need to transform it into more useful         
features.                                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Qualitative features                                                         
 qual_feats=['PassengerId', 'Cabin' ,'Name']                                    
                                                                                
 # Preview qualitative features                                                 
 train[qual_feats].head()                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
Notes:                                                                          

 • PassengerId takes the form gggg_pp where gggg indicates a group the passenger
   is travelling with and pp is their number within the group.                  
 • Cabin takes the form deck/num/side, where side can be either P for Port or S 
   for Starboard.                                                               

Insights:                                                                       

 • We can extract the group and group size from the PassengerId feature.        
 • We can extract the deck, number and side from the cabin feature.             
 • We could extract the surname from the name feature to identify families.     

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                             Feature engineering                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
The philosophy to feature engineering is simple. Better features make better    
models.                                                                         

────────────────────────────────────────────────────────────────────────────────
Age status                                                                      

────────────────────────────────────────────────────────────────────────────────
Bin age feature into groups. This will be helpful for filling missing values    
like expenditure according to age.                                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # New features - training set                                                  
 train['Age_group']=np.nan                                                      
 train.loc[train['Age']<=12,'Age_group']='Age_0-12'                             
 train.loc[(train['Age']>12) & (train['Age']<18),'Age_group']='Age_13-17'       
 train.loc[(train['Age']>=18) & (train['Age']<=25),'Age_group']='Age_18-25'     
 train.loc[(train['Age']>25) & (train['Age']<=30),'Age_group']='Age_26-30'      
 train.loc[(train['Age']>30) & (train['Age']<=50),'Age_group']='Age_31-50'      
 train.loc[train['Age']>50,'Age_group']='Age_51+'                               
                                                                                
 # New features - test set                                                      
 test['Age_group']=np.nan                                                       
 test.loc[test['Age']<=12,'Age_group']='Age_0-12'                               
 test.loc[(test['Age']>12) & (test['Age']<18),'Age_group']='Age_13-17'          
 test.loc[(test['Age']>=18) & (test['Age']<=25),'Age_group']='Age_18-25'        
 test.loc[(test['Age']>25) & (test['Age']<=30),'Age_group']='Age_26-30'         
 test.loc[(test['Age']>30) & (test['Age']<=50),'Age_group']='Age_31-50'         
 test.loc[test['Age']>50,'Age_group']='Age_51+'                                 
                                                                                
 # Plot distribution of new features                                            
 plt.figure(figsize=(10,4))                                                     
 g=sns.countplot(data=train, x='Age_group', hue='Transported',                  
 order=['Age_0-12','Age_13-17','Age_18-25','Age_26-30','Age_31-50','Age_51+'])  
 plt.title('Age group distribution')                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
Expenditure                                                                     

────────────────────────────────────────────────────────────────────────────────
Calculate total expenditure and identify passengers with no expenditure.        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # New features - training set                                                  
 train['Expenditure']=train[exp_feats].sum(axis=1)                              
 train['No_spending']=(train['Expenditure']==0).astype(int)                     
                                                                                
 # New features - test set                                                      
 test['Expenditure']=test[exp_feats].sum(axis=1)                                
 test['No_spending']=(test['Expenditure']==0).astype(int)                       
                                                                                
 # Plot distribution of new features                                            
 fig=plt.figure(figsize=(12,4))                                                 
 plt.subplot(1,2,1)                                                             
 sns.histplot(data=train, x='Expenditure', hue='Transported', bins=200)         
 plt.title('Total expenditure (truncated)')                                     
 plt.ylim([0,200])                                                              
 plt.xlim([0,20000])                                                            
                                                                                
 plt.subplot(1,2,2)                                                             
 sns.countplot(data=train, x='No_spending', hue='Transported')                  
 plt.title('No spending indicator')                                             
 fig.tight_layout()                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
Passenger group                                                                 

────────────────────────────────────────────────────────────────────────────────
Extract passenger group and group size from PassengerId.                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # New feature - Group                                                          
 train['Group'] = train['PassengerId'].apply(lambda x:                          
 x.split('_')[0]).astype(int)                                                   
 test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(in 
                                                                                
 # New feature - Group size                                                     
 train['Group_size']=train['Group'].map(lambda x: pd.concat([train['Group'],    
 test['Group']]).value_counts()[x])                                             
 test['Group_size']=test['Group'].map(lambda x: pd.concat([train['Group'],      
 test['Group']]).value_counts()[x])                                             
                                                                                
 # Plot distribution of new features                                            
 plt.figure(figsize=(20,4))                                                     
 plt.subplot(1,2,1)                                                             
 sns.histplot(data=train, x='Group', hue='Transported', binwidth=1)             
 plt.title('Group')                                                             
                                                                                
 plt.subplot(1,2,2)                                                             
 sns.countplot(data=train, x='Group_size', hue='Transported')                   
 plt.title('Group size')                                                        
 fig.tight_layout()                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
We can't really use the Group feature in our models because it has too big of a 
cardinality (6217) and would explode the number of dimensions with one-hot      
encoding.                                                                       

The Group size on the other hand should be a useful feature. In fact, we can    
compress the feature further by creating a 'Solo' column that tracks whether    
someone is travelling on their own or not. The figure on the right shows that   
group size=1 is less likely to be transported than group size>1.                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # New feature                                                                  
 train['Solo']=(train['Group_size']==1).astype(int)                             
 test['Solo']=(test['Group_size']==1).astype(int)                               
                                                                                
 # New feature distribution                                                     
 plt.figure(figsize=(10,4))                                                     
 sns.countplot(data=train, x='Solo', hue='Transported')                         
 plt.title('Passenger travelling solo or not')                                  
 plt.ylim([0,3000])                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
Cabin location                                                                  

────────────────────────────────────────────────────────────────────────────────
Extract deck, number and side from cabin feature.                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Replace NaN's with outliers for now (so we can split feature)                
 train['Cabin'].fillna('Z/9999/Z', inplace=True)                                
 test['Cabin'].fillna('Z/9999/Z', inplace=True)                                 
                                                                                
 # New features - training set                                                  
 train['Cabin_deck'] = train['Cabin'].apply(lambda x: x.split('/')[0])          
 train['Cabin_number'] = train['Cabin'].apply(lambda x:                         
 x.split('/')[1]).astype(int)                                                   
 train['Cabin_side'] = train['Cabin'].apply(lambda x: x.split('/')[2])          
                                                                                
 # New features - test set                                                      
 test['Cabin_deck'] = test['Cabin'].apply(lambda x: x.split('/')[0])            
 test['Cabin_number'] = test['Cabin'].apply(lambda x:                           
 x.split('/')[1]).astype(int)                                                   
 test['Cabin_side'] = test['Cabin'].apply(lambda x: x.split('/')[2])            
                                                                                
 # Put Nan's back in (we will fill these later)                                 
 train.loc[train['Cabin_deck']=='Z', 'Cabin_deck']=np.nan                       
 train.loc[train['Cabin_number']==9999, 'Cabin_number']=np.nan                  
 train.loc[train['Cabin_side']=='Z', 'Cabin_side']=np.nan                       
 test.loc[test['Cabin_deck']=='Z', 'Cabin_deck']=np.nan                         
 test.loc[test['Cabin_number']==9999, 'Cabin_number']=np.nan                    
 test.loc[test['Cabin_side']=='Z', 'Cabin_side']=np.nan                         
                                                                                
 # Drop Cabin (we don't need it anymore)                                        
 train.drop('Cabin', axis=1, inplace=True)                                      
 test.drop('Cabin', axis=1, inplace=True)                                       
                                                                                
 # Plot distribution of new features                                            
 fig=plt.figure(figsize=(10,12))                                                
 plt.subplot(3,1,1)                                                             
 sns.countplot(data=train, x='Cabin_deck', hue='Transported',                   
 order=['A','B','C','D','E','F','G','T'])                                       
 plt.title('Cabin deck')                                                        
                                                                                
 plt.subplot(3,1,2)                                                             
 sns.histplot(data=train, x='Cabin_number', hue='Transported',binwidth=20)      
 plt.vlines(300, ymin=0, ymax=200, color='black')                               
 plt.vlines(600, ymin=0, ymax=200, color='black')                               
 plt.vlines(900, ymin=0, ymax=200, color='black')                               
 plt.vlines(1200, ymin=0, ymax=200, color='black')                              
 plt.vlines(1500, ymin=0, ymax=200, color='black')                              
 plt.vlines(1800, ymin=0, ymax=200, color='black')                              
 plt.title('Cabin number')                                                      
 plt.xlim([0,2000])                                                             
                                                                                
 plt.subplot(3,1,3)                                                             
 sns.countplot(data=train, x='Cabin_side', hue='Transported')                   
 plt.title('Cabin side')                                                        
 fig.tight_layout()                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
This is interesting! It appears that Cabin_number is grouped into chunks of 300 
cabins. This means we can compress this feature into a categorical one, which   
indicates which chunk each passenger is in.                                     

Other notes: The cabin deck 'T' seems to be an outlier (there are only 5        
samples).                                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # New features - training set                                                  
 train['Cabin_region1']=(train['Cabin_number']<300).astype(int)   # one-hot     
 encoding                                                                       
 train['Cabin_region2']=((train['Cabin_number']>=300) &                         
 (train['Cabin_number']<600)).astype(int)                                       
 train['Cabin_region3']=((train['Cabin_number']>=600) &                         
 (train['Cabin_number']<900)).astype(int)                                       
 train['Cabin_region4']=((train['Cabin_number']>=900) &                         
 (train['Cabin_number']<1200)).astype(int)                                      
 train['Cabin_region5']=((train['Cabin_number']>=1200) &                        
 (train['Cabin_number']<1500)).astype(int)                                      
 train['Cabin_region6']=((train['Cabin_number']>=1500) &                        
 (train['Cabin_number']<1800)).astype(int)                                      
 train['Cabin_region7']=(train['Cabin_number']>=1800).astype(int)               
                                                                                
 # New features - test set                                                      
 test['Cabin_region1']=(test['Cabin_number']<300).astype(int)   # one-hot       
 encoding                                                                       
 test['Cabin_region2']=((test['Cabin_number']>=300) &                           
 (test['Cabin_number']<600)).astype(int)                                        
 test['Cabin_region3']=((test['Cabin_number']>=600) &                           
 (test['Cabin_number']<900)).astype(int)                                        
 test['Cabin_region4']=((test['Cabin_number']>=900) &                           
 (test['Cabin_number']<1200)).astype(int)                                       
 test['Cabin_region5']=((test['Cabin_number']>=1200) &                          
 (test['Cabin_number']<1500)).astype(int)                                       
 test['Cabin_region6']=((test['Cabin_number']>=1500) &                          
 (test['Cabin_number']<1800)).astype(int)                                       
 test['Cabin_region7']=(test['Cabin_number']>=1800).astype(int)                 
                                                                                
 # Plot distribution of new features                                            
 plt.figure(figsize=(10,4))                                                     
 train['Cabin_regions_plot']=(train['Cabin_region1']+2*train['Cabin_region2']+3 
 rain['Cabin_region3']+4*train['Cabin_region4']+5*train['Cabin_region5']+6*trai 
 'Cabin_region6']+7*train['Cabin_region7']).astype(int)                         
 sns.countplot(data=train, x='Cabin_regions_plot', hue='Transported')           
 plt.title('Cabin regions')                                                     
 train.drop('Cabin_regions_plot', axis=1, inplace=True)                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
Last name                                                                       

────────────────────────────────────────────────────────────────────────────────
Calculate family size from last name.                                           

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Replace NaN's with outliers for now (so we can split feature)                
 train['Name'].fillna('Unknown Unknown', inplace=True)                          
 test['Name'].fillna('Unknown Unknown', inplace=True)                           
                                                                                
 # New feature - Surname                                                        
 train['Surname']=train['Name'].str.split().str[-1]                             
 test['Surname']=test['Name'].str.split().str[-1]                               
                                                                                
 # New feature - Family size                                                    
 train['Family_size']=train['Surname'].map(lambda x:                            
 pd.concat([train['Surname'],test['Surname']]).value_counts()[x])               
 test['Family_size']=test['Surname'].map(lambda x:                              
 pd.concat([train['Surname'],test['Surname']]).value_counts()[x])               
                                                                                
 # Put Nan's back in (we will fill these later)                                 
 train.loc[train['Surname']=='Unknown','Surname']=np.nan                        
 train.loc[train['Family_size']>100,'Family_size']=np.nan                       
 test.loc[test['Surname']=='Unknown','Surname']=np.nan                          
 test.loc[test['Family_size']>100,'Family_size']=np.nan                         
                                                                                
 # Drop name (we don't need it anymore)                                         
 train.drop('Name', axis=1, inplace=True)                                       
 test.drop('Name', axis=1, inplace=True)                                        
                                                                                
 # New feature distribution                                                     
 plt.figure(figsize=(12,4))                                                     
 sns.countplot(data=train, x='Family_size', hue='Transported')                  
 plt.title('Family size')                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                Missing values                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
Combine train and test                                                          

────────────────────────────────────────────────────────────────────────────────
This will make it easier to fill missing values. We will split it back later.   

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Labels and features                                                          
 y=train['Transported'].copy().astype(int)                                      
 X=train.drop('Transported', axis=1).copy()                                     
                                                                                
 # Concatenate dataframes                                                       
 data=pd.concat([X, test], axis=0).reset_index(drop=True)                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
Explore missing values                                                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Columns with missing values                                                  
 na_cols=data.columns[data.isna().any()].tolist()                               
                                                                                
 # Missing values summary                                                       
 mv=pd.DataFrame(data[na_cols].isna().sum(), columns=['Number_missing'])        
 mv['Percentage_missing']=np.round(100*mv['Number_missing']/len(data),2)        
 mv                                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Heatmap of missing values                                                    
 plt.figure(figsize=(12,6))                                                     
 sns.heatmap(train[na_cols].isna().T, cmap='summer')                            
 plt.title('Heatmap of missing values')                                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
Missing values make up about 2% of the data, which is a relatively small amount.
For the most part, they don't seem to be happening at the same time (except the 
features made from splitting Cabin and Name), but let's inspect closer.         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Countplot of number of missing values by passenger                           
 train['na_count']=train.isna().sum(axis=1)                                     
 plt.figure(figsize=(10,4))                                                     
 sns.countplot(data=train, x='na_count', hue='Transported')                     
 plt.title('Number of missing entries by passenger')                            
 train.drop('na_count', axis=1, inplace=True)                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Notes:                                                                          

 • Missing values are independent of the target and for the most part are       
   isolated.                                                                    
 • Even though only 2% of the data is missing, about 25% of all passengers have 
   at least 1 missing value.                                                    
 • PassengerId is the only (original) feature to not have any missing values.   

Insight:                                                                        

 • Since most of the missing values are isolated it makes sense to try to fill  
   these in as opposed to just dropping rows.                                   
 • If there is a relationship between PassengerId and other features we can fill
   missing values according to this column.                                     

────────────────────────────────────────────────────────────────────────────────
Strategy                                                                        

────────────────────────────────────────────────────────────────────────────────
The easiest way to deal with missing values is to just use the median for       
continuous features and the mode for categorical features (see version 20 of    
this notebook). This will work 'well enough' but if we want to maximise the     
accuracy of our models then we need to look for patterns within the missing     
data. The way to do this is by looking at the joint distribution of features,   
e.g. do passengers from the same group tend to come from the same family? There 
are obviously many combinations so we will just summarise the useful trends I   
and others have found.                                                          

────────────────────────────────────────────────────────────────────────────────
HomePlanet and Group                                                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution of Group and HomePlanet                                   
 GHP_gb=data.groupby(['Group','HomePlanet'])['HomePlanet'].size().unstack().fil 
 a(0)                                                                           
 GHP_gb.head()                                                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Countplot of unique values                                                   
 sns.countplot((GHP_gb>0).sum(axis=1))                                          
 plt.title('Number of unique home planets per group')                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
Great! This shows that everyone in the same group comes from the same home      
planet. So we can fill the missing HomePlanet values according to the group. (At
least the ones where the group size is bigger than 1.)                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 HP_bef=data['HomePlanet'].isna().sum()                                         
                                                                                
 # Passengers with missing HomePlanet and in a group with known HomePlanet      
 GHP_index=data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Gr 
 p']).isin(GHP_gb.index)].index                                                 
                                                                                
 # Fill corresponding missing values                                            
 data.loc[GHP_index,'HomePlanet']=data.iloc[GHP_index,:]['Group'].map(lambda x: 
 GHP_gb.idxmax(axis=1)[x])                                                      
                                                                                
 # Print number of missing values left                                          
 print('#HomePlanet missing values before:',HP_bef)                             
 print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
We managed to fill 131 values with 100% confidence but we are not finished yet. 

────────────────────────────────────────────────────────────────────────────────
HomePlanet and CabinDeck                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution of CabinDeck and HomePlanet                               
 CDHP_gb=data.groupby(['Cabin_deck','HomePlanet'])['HomePlanet'].size().unstack 
 .fillna(0)                                                                     
                                                                                
 # Heatmap of missing values                                                    
 plt.figure(figsize=(10,4))                                                     
 sns.heatmap(CDHP_gb.T, annot=True, fmt='g', cmap='coolwarm')                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Notes:                                                                          

 • Passengers on decks A, B, C or T came from Europa.                           
 • Passengers on deck G came from Earth.                                        
 • Passengers on decks D, E or F came from multiple planets.                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 HP_bef=data['HomePlanet'].isna().sum()                                         
                                                                                
 # Decks A, B, C or T came from Europa                                          
 data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck'].isin(['A', 'B', 'C' 
 'T'])), 'HomePlanet']='Europa'                                                 
                                                                                
 # Deck G came from Earth                                                       
 data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck']=='G'),              
 'HomePlanet']='Earth'                                                          
                                                                                
 # Print number of missing values left                                          
 print('#HomePlanet missing values before:',HP_bef)                             
 print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
HomePlanet and Surname                                                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution of Surname and HomePlanet                                 
 SHP_gb=data.groupby(['Surname','HomePlanet'])['HomePlanet'].size().unstack().f 
 lna(0)                                                                         
                                                                                
 # Countplot of unique values                                                   
 plt.figure(figsize=(10,4))                                                     
 sns.countplot((SHP_gb>0).sum(axis=1))                                          
 plt.title('Number of unique planets per surname')                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
Fantastic! Everyone with the same surname comes from the same home planet.      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 HP_bef=data['HomePlanet'].isna().sum()                                         
                                                                                
 # Passengers with missing HomePlanet and in a family with known HomePlanet     
 SHP_index=data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Su 
 ame']).isin(SHP_gb.index)].index                                               
                                                                                
 # Fill corresponding missing values                                            
 data.loc[SHP_index,'HomePlanet']=data.iloc[SHP_index,:]['Surname'].map(lambda  
 SHP_gb.idxmax(axis=1)[x])                                                      
                                                                                
 # Print number of missing values left                                          
 print('#HomePlanet missing values before:',HP_bef)                             
 print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Only 10 HomePlanet missing values left - let's look at them                  
 data[data['HomePlanet'].isna()][['PassengerId','HomePlanet','Destination']]    
                                                                                

────────────────────────────────────────────────────────────────────────────────
Everyone left is heading towards TRAPPIST-1e. So let's look at the joint        
distribution of HomePlanet and Destination.                                     

────────────────────────────────────────────────────────────────────────────────
HomePlanet and Destination                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution of HomePlanet and Destination                             
 HPD_gb=data.groupby(['HomePlanet','Destination'])['Destination'].size().unstac 
 ).fillna(0)                                                                    
                                                                                
 # Heatmap of missing values                                                    
 plt.figure(figsize=(10,4))                                                     
 sns.heatmap(HPD_gb.T, annot=True, fmt='g', cmap='coolwarm')                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
Most people heading towards TRAPPIST-1e came from Earth so it makes sense to    
guess they came from there. But remember from earlier, no one on deck D came    
from Earth so we need to filter these out.                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 HP_bef=data['HomePlanet'].isna().sum()                                         
                                                                                
 # Fill remaining HomePlanet missing values with Earth (if not on deck D) or Ma 
 (if on Deck D)                                                                 
 data.loc[(data['HomePlanet'].isna()) & ~(data['Cabin_deck']=='D'),             
 'HomePlanet']='Earth'                                                          
 data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck']=='D'),              
 'HomePlanet']='Mars'                                                           
                                                                                
 # Print number of missing values left                                          
 print('#HomePlanet missing values before:',HP_bef)                             
 print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
Awesome! We're done with HomePlanet.                                            

────────────────────────────────────────────────────────────────────────────────
Destination                                                                     

────────────────────────────────────────────────────────────────────────────────
Since the majority (68%) of passengers are heading towards TRAPPIST-1e (see EDA 
section), we'll just impute this value (i.e. the mode). A better rule hasn't    
been found at this stage.                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 D_bef=data['Destination'].isna().sum()                                         
                                                                                
 # Fill missing Destination values with mode                                    
 data.loc[(data['Destination'].isna()), 'Destination']='TRAPPIST-1e'            
                                                                                
 # Print number of missing values left                                          
 print('#Destination missing values before:',D_bef)                             
 print('#Destination missing values after:',data['Destination'].isna().sum())   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Surname and group                                                               

────────────────────────────────────────────────────────────────────────────────
The reason we are filling missing surnames is because we will use surnames later
to fill missing values of other features. It also means we can improve the      
accuracy of the family size feature.                                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution of Group and Surname                                      
 GSN_gb=data[data['Group_size']>1].groupby(['Group','Surname'])['Surname'].size 
 .unstack().fillna(0)                                                           
                                                                                
 # Countplot of unique values                                                   
 plt.figure(figsize=(10,4))                                                     
 sns.countplot((GSN_gb>0).sum(axis=1))                                          
 plt.title('Number of unique surnames by group')                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
The majority (83%) of groups contain only 1 family. So let's fill missing       
surnames according to the majority surname in that group.                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 SN_bef=data['Surname'].isna().sum()                                            
                                                                                
 # Passengers with missing Surname and in a group with known majority Surname   
 GSN_index=data[data['Surname'].isna()][(data[data['Surname'].isna()]['Group']) 
 sin(GSN_gb.index)].index                                                       
                                                                                
 # Fill corresponding missing values                                            
 data.loc[GSN_index,'Surname']=data.iloc[GSN_index,:]['Group'].map(lambda x:    
 GSN_gb.idxmax(axis=1)[x])                                                      
                                                                                
 # Print number of missing values left                                          
 print('#Surname missing values before:',SN_bef)                                
 print('#Surname missing values after:',data['Surname'].isna().sum())           
                                                                                

────────────────────────────────────────────────────────────────────────────────
That is the best we can do. We don't have to get rid of all of these missing    
values because we will end up dropping the surname feature anyway. However, we  
can update the family size feature.                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Replace NaN's with outliers (so we can use map)                              
 data['Surname'].fillna('Unknown', inplace=True)                                
                                                                                
 # Update family size feature                                                   
 data['Family_size']=data['Surname'].map(lambda x:                              
 data['Surname'].value_counts()[x])                                             
                                                                                
 # Put NaN's back in place of outliers                                          
 data.loc[data['Surname']=='Unknown','Surname']=np.nan                          
                                                                                
 # Say unknown surname means no family                                          
 data.loc[data['Family_size']>100,'Family_size']=0                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
CabinSide and Group                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution of Group and Cabin features                               
 GCD_gb=data[data['Group_size']>1].groupby(['Group','Cabin_deck'])['Cabin_deck' 
 size().unstack().fillna(0)                                                     
 GCN_gb=data[data['Group_size']>1].groupby(['Group','Cabin_number'])['Cabin_num 
 r'].size().unstack().fillna(0)                                                 
 GCS_gb=data[data['Group_size']>1].groupby(['Group','Cabin_side'])['Cabin_side' 
 size().unstack().fillna(0)                                                     
                                                                                
 # Countplots                                                                   
 fig=plt.figure(figsize=(16,4))                                                 
 plt.subplot(1,3,1)                                                             
 sns.countplot((GCD_gb>0).sum(axis=1))                                          
 plt.title('#Unique cabin decks per group')                                     
                                                                                
 plt.subplot(1,3,2)                                                             
 sns.countplot((GCN_gb>0).sum(axis=1))                                          
 plt.title('#Unique cabin numbers per group')                                   
                                                                                
 plt.subplot(1,3,3)                                                             
 sns.countplot((GCS_gb>0).sum(axis=1))                                          
 plt.title('#Unique cabin sides per group')                                     
 fig.tight_layout()                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
Another rule! Everyone in the same group is also on the same cabin side. For    
cabin deck and cabin number there is also a fairly good (but not perfect)       
correlation with group.                                                         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 CS_bef=data['Cabin_side'].isna().sum()                                         
                                                                                
 # Passengers with missing Cabin side and in a group with known Cabin side      
 GCS_index=data[data['Cabin_side'].isna()][(data[data['Cabin_side'].isna()]['Gr 
 p']).isin(GCS_gb.index)].index                                                 
                                                                                
 # Fill corresponding missing values                                            
 data.loc[GCS_index,'Cabin_side']=data.iloc[GCS_index,:]['Group'].map(lambda x: 
 GCS_gb.idxmax(axis=1)[x])                                                      
                                                                                
 # Print number of missing values left                                          
 print('#Cabin_side missing values before:',CS_bef)                             
 print('#Cabin_side missing values after:',data['Cabin_side'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
CabinSide and Surname                                                           

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution of Surname and Cabin side                                 
 SCS_gb=data[data['Group_size']>1].groupby(['Surname','Cabin_side'])['Cabin_sid 
 ].size().unstack().fillna(0)                                                   
                                                                                
 # Ratio of sides                                                               
 SCS_gb['Ratio']=SCS_gb['P']/(SCS_gb['P']+SCS_gb['S'])                          
                                                                                
 # Histogram of ratio                                                           
 plt.figure(figsize=(10,4))                                                     
 sns.histplot(SCS_gb['Ratio'], kde=True, binwidth=0.05)                         
 plt.title('Ratio of cabin side by surname')                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Print proportion                                                             
 print('Percentage of families all on the same cabin side:',                    
 100*np.round((SCS_gb['Ratio'].isin([0,1])).sum()/len(SCS_gb),3),'%')           
                                                                                
 # Another view of the same information                                         
 SCS_gb.head()                                                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
This shows that families tend to be on the same cabin side (and 77% of families 
are entirely on the same side).                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 CS_bef=data['Cabin_side'].isna().sum()                                         
                                                                                
 # Drop ratio column                                                            
 SCS_gb.drop('Ratio', axis=1, inplace=True)                                     
                                                                                
 # Passengers with missing Cabin side and in a family with known Cabin side     
 SCS_index=data[data['Cabin_side'].isna()][(data[data['Cabin_side'].isna()]['Su 
 ame']).isin(SCS_gb.index)].index                                               
                                                                                
 # Fill corresponding missing values                                            
 data.loc[SCS_index,'Cabin_side']=data.iloc[SCS_index,:]['Surname'].map(lambda  
 SCS_gb.idxmax(axis=1)[x])                                                      
                                                                                
 # Drop surname (we don't need it anymore)                                      
 data.drop('Surname', axis=1, inplace=True)                                     
                                                                                
 # Print number of missing values left                                          
 print('#Cabin_side missing values before:',CS_bef)                             
 print('#Cabin_side missing values after:',data['Cabin_side'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
The remaining missing values will be replaced with an outlier. This is because  
we really don't know which one of the two (balanced) sides we should assign.    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Value counts                                                                 
 data['Cabin_side'].value_counts()                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 CS_bef=data['Cabin_side'].isna().sum()                                         
                                                                                
 # Fill remaining missing values with outlier                                   
 data.loc[data['Cabin_side'].isna(),'Cabin_side']='Z'                           
                                                                                
 # Print number of missing values left                                          
 print('#Cabin_side missing values before:',CS_bef)                             
 print('#Cabin_side missing values after:',data['Cabin_side'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
CabinDeck and Group                                                             

────────────────────────────────────────────────────────────────────────────────
Remember (from above) that groups tend to be on the same cabin deck.            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 CD_bef=data['Cabin_deck'].isna().sum()                                         
                                                                                
 # Passengers with missing Cabin deck and in a group with known majority Cabin  
 deck                                                                           
 GCD_index=data[data['Cabin_deck'].isna()][(data[data['Cabin_deck'].isna()]['Gr 
 p']).isin(GCD_gb.index)].index                                                 
                                                                                
 # Fill corresponding missing values                                            
 data.loc[GCD_index,'Cabin_deck']=data.iloc[GCD_index,:]['Group'].map(lambda x: 
 GCD_gb.idxmax(axis=1)[x])                                                      
                                                                                
 # Print number of missing values left                                          
 print('#Cabin_deck missing values before:',CD_bef)                             
 print('#Cabin_deck missing values after:',data['Cabin_deck'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
CabinDeck and HomePlanet                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution                                                           
 data.groupby(['HomePlanet','Destination','Solo','Cabin_deck'])['Cabin_deck'].s 
 e().unstack().fillna(0)                                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
Notes:                                                                          

 • Passengers from Mars are most likely in deck F.                              
 • Passengers from Europa are (more or less) most likely in deck C if travelling
   solo and deck B otherwise.                                                   
 • Passengers from Earth are (more or less) most likely in deck G.              

────────────────────────────────────────────────────────────────────────────────
We will fill in missing values according to where the mode appears in these     
subgroups.                                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 CD_bef=data['Cabin_deck'].isna().sum()                                         
                                                                                
 # Fill missing values using the mode                                           
 na_rows_CD=data.loc[data['Cabin_deck'].isna(),'Cabin_deck'].index              
 data.loc[data['Cabin_deck'].isna(),'Cabin_deck']=data.groupby(['HomePlanet','D 
 tination','Solo'])['Cabin_deck'].transform(lambda x:                           
 x.fillna(pd.Series.mode(x)[0]))[na_rows_CD]                                    
                                                                                
 # Print number of missing values left                                          
 print('#Cabin_deck missing values before:',CD_bef)                             
 print('#Cabin_deck missing values after:',data['Cabin_deck'].isna().sum())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
CabinNumber and CabinDeck                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Scatterplot                                                                  
 plt.figure(figsize=(10,4))                                                     
 sns.scatterplot(x=data['Cabin_number'], y=data['Group'],                       
 c=LabelEncoder().fit_transform(data.loc[~data['Cabin_number'].isna(),'Cabin_de 
 ']), cmap='tab10')                                                             
 plt.title('Cabin_number vs group coloured by group')                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
There is an interesting pattern here. The cabin_number and group_number share a 
linear relationship on a deck by deck basis. We can therefore extrapolate the   
missing cabin numbers using linear regression on a deck by deck basis to get an 
approximate cabin number.                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 CN_bef=data['Cabin_number'].isna().sum()                                       
                                                                                
 # Extrapolate linear relationship on a deck by deck basis                      
 for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:                               
     # Features and labels                                                      
     X_CN=data.loc[~(data['Cabin_number'].isna()) &                             
 (data['Cabin_deck']==deck),'Group']                                            
     y_CN=data.loc[~(data['Cabin_number'].isna()) &                             
 (data['Cabin_deck']==deck),'Cabin_number']                                     
     X_test_CN=data.loc[(data['Cabin_number'].isna()) &                         
 (data['Cabin_deck']==deck),'Group']                                            
                                                                                
     # Linear regression                                                        
     model_CN=LinearRegression()                                                
     model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)                             
     preds_CN=model_CN.predict(X_test_CN.values.reshape(-1, 1))                 
                                                                                
     # Fill missing values with predictions                                     
     data.loc[(data['Cabin_number'].isna()) &                                   
 (data['Cabin_deck']==deck),'Cabin_number']=preds_CN.astype(int)                
                                                                                
 # Print number of missing values left                                          
 print('#Cabin_number missing values before:',CN_bef)                           
 print('#Cabin_number missing values after:',data['Cabin_number'].isna().sum()) 
                                                                                

────────────────────────────────────────────────────────────────────────────────
Let's update the cabin regions with the new data.                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # One-hot encode cabin regions                                                 
 data['Cabin_region1']=(data['Cabin_number']<300).astype(int)                   
 data['Cabin_region2']=((data['Cabin_number']>=300) &                           
 (data['Cabin_number']<600)).astype(int)                                        
 data['Cabin_region3']=((data['Cabin_number']>=600) &                           
 (data['Cabin_number']<900)).astype(int)                                        
 data['Cabin_region4']=((data['Cabin_number']>=900) &                           
 (data['Cabin_number']<1200)).astype(int)                                       
 data['Cabin_region5']=((data['Cabin_number']>=1200) &                          
 (data['Cabin_number']<1500)).astype(int)                                       
 data['Cabin_region6']=((data['Cabin_number']>=1500) &                          
 (data['Cabin_number']<1800)).astype(int)                                       
 data['Cabin_region7']=(data['Cabin_number']>=1800).astype(int)                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
VIP                                                                             

────────────────────────────────────────────────────────────────────────────────
VIP is a highly unbalanced binary feature so we will just impute the mode.      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 data['VIP'].value_counts()                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 V_bef=data['VIP'].isna().sum()                                                 
                                                                                
 # Fill missing values with mode                                                
 data.loc[data['VIP'].isna(),'VIP']=False                                       
                                                                                
 # Print number of missing values left                                          
 print('#VIP missing values before:',V_bef)                                     
 print('#VIP missing values after:',data['VIP'].isna().sum())                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Age                                                                             

────────────────────────────────────────────────────────────────────────────────
Age varies across many features like HomePlanet, group size, expenditure and    
cabin deck, so we will impute missing values according to the median of these   
subgroups.                                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution                                                           
 data.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].median() 
 nstack().fillna(0)                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 A_bef=data[exp_feats].isna().sum().sum()                                       
                                                                                
 # Fill missing values using the median                                         
 na_rows_A=data.loc[data['Age'].isna(),'Age'].index                             
 data.loc[data['Age'].isna(),'Age']=data.groupby(['HomePlanet','No_spending','S 
 o','Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[na_rows_A]  
                                                                                
 # Print number of missing values left                                          
 print('#Age missing values before:',A_bef)                                     
 print('#Age missing values after:',data['Age'].isna().sum())                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Let's update the age_group feature using the new data.                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Update age group feature                                                     
 data.loc[data['Age']<=12,'Age_group']='Age_0-12'                               
 data.loc[(data['Age']>12) & (data['Age']<18),'Age_group']='Age_13-17'          
 data.loc[(data['Age']>=18) & (data['Age']<=25),'Age_group']='Age_18-25'        
 data.loc[(data['Age']>25) & (data['Age']<=30),'Age_group']='Age_26-30'         
 data.loc[(data['Age']>30) & (data['Age']<=50),'Age_group']='Age_31-50'         
 data.loc[data['Age']>50,'Age_group']='Age_51+'                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
CryoSleep                                                                       

────────────────────────────────────────────────────────────────────────────────
The best way to predict if a passenger is in CryoSleep or not is to see if they 
spent anything.                                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution                                                           
 data.groupby(['No_spending','CryoSleep'])['CryoSleep'].size().unstack().fillna 
 )                                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 CSL_bef=data['CryoSleep'].isna().sum()                                         
                                                                                
 # Fill missing values using the mode                                           
 na_rows_CSL=data.loc[data['CryoSleep'].isna(),'CryoSleep'].index               
 data.loc[data['CryoSleep'].isna(),'CryoSleep']=data.groupby(['No_spending'])[' 
 yoSleep'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CSL]     
                                                                                
 # Print number of missing values left                                          
 print('#CryoSleep missing values before:',CSL_bef)                             
 print('#CryoSleep missing values after:',data['CryoSleep'].isna().sum())       
                                                                                

────────────────────────────────────────────────────────────────────────────────
Expenditure and CryoSleep                                                       

────────────────────────────────────────────────────────────────────────────────
This one makes a lot of sense. We don't expect people in CryoSleep to be able to
spend anything.                                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print('Maximum expenditure of passengers in                                    
 CryoSleep:',data.loc[data['CryoSleep']==True,exp_feats].sum(axis=1).max())     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 E_bef=data[exp_feats].isna().sum().sum()                                       
                                                                                
 # CryoSleep has no expenditure                                                 
 for col in exp_feats:                                                          
     data.loc[(data[col].isna()) & (data['CryoSleep']==True), col]=0            
                                                                                
 # Print number of missing values left                                          
 print('#Expenditure missing values before:',E_bef)                             
 print('#Expenditure missing values after:',data[exp_feats].isna().sum().sum()) 
                                                                                

────────────────────────────────────────────────────────────────────────────────
Expenditure and others                                                          

────────────────────────────────────────────────────────────────────────────────
Expenditure varies across many features but we will only impute missing values  
using HomePlanet, Solo and Age group to prevent overfitting. We will also use   
the mean instead of the median because a large proportion of passengers don't   
spend anything and median usually comes out as 0. Note how under 12's don't     
spend anything.                                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Joint distribution                                                           
 data.groupby(['HomePlanet','Solo','Age_group'])['Expenditure'].mean().unstack( 
 fillna(0)                                                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Missing values before                                                        
 E_bef=data[exp_feats].isna().sum().sum()                                       
                                                                                
 # Fill remaining missing values using the median                               
 for col in exp_feats:                                                          
     na_rows=data.loc[data[col].isna(),col].index                               
     data.loc[data[col].isna(),col]=data.groupby(['HomePlanet','Solo','Age_grou 
 ])[col].transform(lambda x: x.fillna(x.mean()))[na_rows]                       
                                                                                
 # Print number of missing values left                                          
 print('#Expenditure missing values before:',E_bef)                             
 print('#Expenditure missing values after:',data[exp_feats].isna().sum().sum()) 
                                                                                

────────────────────────────────────────────────────────────────────────────────
Finally, we can update the expenditure and no_spending features with these new  
data points.                                                                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Update expenditure and no_spending                                           
 data['Expenditure']=data[exp_feats].sum(axis=1)                                
 data['No_spending']=(data['Expenditure']==0).astype(int)                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 data.isna().sum()                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
No missing values left! It was a lot of effort but it should improve the        
accuracy of our models.                                                         

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                Preprocessing                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
Split data back into train and test sets                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Train and test                                                               
 X=data[data['PassengerId'].isin(train['PassengerId'].values)].copy()           
 X_test=data[data['PassengerId'].isin(test['PassengerId'].values)].copy()       
                                                                                

────────────────────────────────────────────────────────────────────────────────
Drop unwanted features                                                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Drop qualitative/redundant/collinear/high cardinality features               
 X.drop(['PassengerId', 'Group', 'Group_size', 'Age_group', 'Cabin_number'],    
 axis=1, inplace=True)                                                          
 X_test.drop(['PassengerId', 'Group', 'Group_size', 'Age_group', 'Cabin_number' 
 axis=1, inplace=True)                                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
Log transform                                                                   

────────────────────────────────────────────────────────────────────────────────
The logarithm transform is used to decrease skew in distributions, especially   
with large outliers. It can make it easier for algorithms to 'learn' the correct
relationships. We will apply it to the expenditure features as these are heavily
skewed by outliers.                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Plot log transform results                                                   
 fig=plt.figure(figsize=(12,20))                                                
 for i, col in                                                                  
 enumerate(['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Expenditur 
 ]):                                                                            
     plt.subplot(6,2,2*i+1)                                                     
     sns.histplot(X[col], binwidth=100)                                         
     plt.ylim([0,200])                                                          
     plt.title(f'{col} (original)')                                             
                                                                                
     plt.subplot(6,2,2*i+2)                                                     
     sns.histplot(np.log(1+X[col]), color='C1')                                 
     plt.ylim([0,200])                                                          
     plt.title(f'{col} (log-transform)')                                        
                                                                                
 fig.tight_layout()                                                             
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Apply log transform                                                          
 for col in                                                                     
 ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Expenditure']:       
     X[col]=np.log(1+X[col])                                                    
     X_test[col]=np.log(1+X_test[col])                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
Encoding and scaling                                                            

────────────────────────────────────────────────────────────────────────────────
We will use column transformers to be more professional. It's also good         
practice.                                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Indentify numerical and categorical columns                                  
 numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64',  
 'float64']]                                                                    
 categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object" 
                                                                                
 # Scale numerical data to have mean=0 and variance=1                           
 numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])         
                                                                                
 # One-hot encode categorical data                                              
 categorical_transformer = Pipeline(steps=[('onehot',                           
 OneHotEncoder(drop='if_binary', handle_unknown='ignore',sparse=False))])       
                                                                                
 # Combine preprocessing                                                        
 ct = ColumnTransformer(                                                        
     transformers=[                                                             
         ('num', numerical_transformer, numerical_cols),                        
         ('cat', categorical_transformer, categorical_cols)],                   
         remainder='passthrough')                                               
                                                                                
 # Apply preprocessing                                                          
 X = ct.fit_transform(X)                                                        
 X_test = ct.transform(X_test)                                                  
                                                                                
 # Print new shape                                                              
 print('Training set shape:', X.shape)                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
PCA                                                                             

────────────────────────────────────────────────────────────────────────────────
Just for fun, let's look at the transformed data in PCA space. This gives a low 
dimensional representation of the data, which preserves local and global        
structure.                                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 pca = PCA(n_components=3)                                                      
 components = pca.fit_transform(X)                                              
                                                                                
 total_var = pca.explained_variance_ratio_.sum() * 100                          
                                                                                
 fig = px.scatter_3d(                                                           
     components, x=0, y=1, z=2, color=y, size=0.1*np.ones(len(X)), opacity = 1, 
     title=f'Total Explained Variance: {total_var:.2f}%',                       
     labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},                            
     width=800, height=500                                                      
 )                                                                              
 fig.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Explained variance (how important each additional principal component is)    
 pca = PCA().fit(X)                                                             
 fig, ax = plt.subplots(figsize=(10,4))                                         
 xi = np.arange(1, 1+X.shape[1], step=1)                                        
 yi = np.cumsum(pca.explained_variance_ratio_)                                  
 plt.plot(xi, yi, marker='o', linestyle='--', color='b')                        
                                                                                
 # Aesthetics                                                                   
 plt.ylim(0.0,1.1)                                                              
 plt.xlabel('Number of Components')                                             
 plt.xticks(np.arange(1, 1+X.shape[1], step=2))                                 
 plt.ylabel('Cumulative variance (%)')                                          
 plt.title('Explained variance by each component')                              
 plt.axhline(y=1, color='r', linestyle='-')                                     
 plt.text(0.5, 0.85, '100% cut-off threshold', color = 'red')                   
 ax.grid(axis='x')                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
Create a validation set                                                         

────────────────────────────────────────────────────────────────────────────────
We will use this to choose which model(s) to use.                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Train-validation split                                                       
 X_train, X_valid, y_train, y_valid =                                           
 train_test_split(X,y,stratify=y,train_size=0.8,test_size=0.2,random_state=0)   
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                               Model selection                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
To briefly mention the algorithms we will use,                                  

Logistic Regression: Unlike linear regression which uses Least Squares, this    
model uses Maximum Likelihood Estimation to fit a sigmoid-curve on the target   
variable distribution. The sigmoid/logistic curve is commonly used when the data
is questions had binary output.                                                 

K-Nearest Neighbors (KNN): KNN works by selecting the majority class of the     
k-nearest neighbours, where the metric used is usually Euclidean distance. It is
a simple and effective algorithm but can be sensitive by many factors, e.g. the 
value of k, the preprocessing done to the data and the metric used.             

Support Vector Machine (SVM): SVM finds the optimal hyperplane that seperates   
the data in the feature space. Predictions are made by looking at which side of 
the hyperplane the test point lies on. Ordinary SVM assumes the data is linearly
separable, which is not always the case. A kernel trick can be used when this   
assumption fails to transform the data into a higher dimensional space where it 
is linearly seperable. SVM is a popular algorithm because it is computationally 
effecient and produces very good results.                                       

Random Forest (RF): RF is a reliable ensemble of decision trees, which can be   
used for regression or classification problems. Here, the individual trees are  
built via bagging (i.e. aggregation of bootstraps which are nothing but multiple
train datasets created via sampling with replacement) and split using fewer     
features. The resulting diverse forest of uncorrelated trees exhibits reduced   
variance; therefore, is more robust towards change in data and carries its      
prediction accuracy to new data. It works well with both continuous &           
categorical data.                                                               

Extreme Gradient Boosting (XGBoost): XGBoost is similar to RF in that it is made
up of an ensemble of decision-trees. The difference arises in how those trees as
derived; XGboost uses extreme gradient boosting when optimising its objective   
function. It often produces the best results but is relatively slow compared to 
other gradient boosting algorithms.                                             

Light Gradient Boosting Machine (LGBM): LGBM works essentially the same as      
XGBoost but with a lighter boosting technique. It usually produces similar      
results to XGBoost but is significantly faster.                                 

Categorical Boosting (CatBoost): CatBoost is an open source algorithm based on  
gradient boosted decision trees. It supports numerical, categorical and text    
features. It works well with heterogeneous data and even relatively small data. 
Informally, it tries to take the best of both worlds from XGBoost and LGBM.     

Naive Bayes (NB): Naive Bayes learns how to classify samples by using Bayes'    
Theorem. It uses prior information to 'update' the probability of an event by   
incoorporateing this information according to Bayes' law. The algorithm is quite
fast but a downside is that it assumes the input features are independent, which
is not always the case.                                                         

────────────────────────────────────────────────────────────────────────────────
We will train these models and evaluate them on the validation set to then      
choose which ones to carry through to the next stage (cross validation).        

────────────────────────────────────────────────────────────────────────────────
Define classifiers                                                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Classifiers                                                                  
 classifiers = {                                                                
     "LogisticRegression" : LogisticRegression(random_state=0),                 
     "KNN" : KNeighborsClassifier(),                                            
     "SVC" : SVC(random_state=0, probability=True),                             
     "RandomForest" : RandomForestClassifier(random_state=0),                   
     #"XGBoost" : XGBClassifier(random_state=0, use_label_encoder=False,        
 eval_metric='logloss'), # XGBoost takes too long                               
     "LGBM" : LGBMClassifier(random_state=0),                                   
     "CatBoost" : CatBoostClassifier(random_state=0, verbose=False),            
     "NaiveBayes": GaussianNB()                                                 
 }                                                                              
                                                                                
 # Grids for grid search                                                        
 LR_grid = {'penalty': ['l1','l2'],                                             
            'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],                               
            'max_iter': [50, 100, 150]}                                         
                                                                                
 KNN_grid = {'n_neighbors': [3, 5, 7, 9],                                       
             'p': [1, 2]}                                                       
                                                                                
 SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],                              
             'kernel': ['linear', 'rbf'],                                       
             'gamma': ['scale', 'auto']}                                        
                                                                                
 RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],                      
         'max_depth': [4, 6, 8, 10, 12]}                                        
                                                                                
 boosted_grid = {'n_estimators': [50, 100, 150, 200],                           
         'max_depth': [4, 8, 12],                                               
         'learning_rate': [0.05, 0.1, 0.15]}                                    
                                                                                
 NB_grid={'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}                           
                                                                                
 # Dictionary of all grids                                                      
 grid = {                                                                       
     "LogisticRegression" : LR_grid,                                            
     "KNN" : KNN_grid,                                                          
     "SVC" : SVC_grid,                                                          
     "RandomForest" : RF_grid,                                                  
     "XGBoost" : boosted_grid,                                                  
     "LGBM" : boosted_grid,                                                     
     "CatBoost" : boosted_grid,                                                 
     "NaiveBayes": NB_grid                                                      
 }                                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
Train and evaluate models                                                       

────────────────────────────────────────────────────────────────────────────────
Train models with grid search (but no cross validation so it doesn't take too   
long) to get a rough idea of which are the best models for this dataset.        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 i=0                                                                            
 clf_best_params=classifiers.copy()                                             
 valid_scores=pd.DataFrame({'Classifer':classifiers.keys(), 'Validation         
 accuracy': np.zeros(len(classifiers)), 'Training time':                        
 np.zeros(len(classifiers))})                                                   
 for key, classifier in classifiers.items():                                    
     start = time.time()                                                        
     clf = GridSearchCV(estimator=classifier, param_grid=grid[key], n_jobs=-1,  
 cv=None)                                                                       
                                                                                
     # Train and score                                                          
     clf.fit(X_train, y_train)                                                  
     valid_scores.iloc[i,1]=clf.score(X_valid, y_valid)                         
                                                                                
     # Save trained model                                                       
     clf_best_params[key]=clf.best_params_                                      
                                                                                
     # Print iteration and training time                                        
     stop = time.time()                                                         
     valid_scores.iloc[i,2]=np.round((stop - start)/60, 2)                      
                                                                                
     print('Model:', key)                                                       
     print('Training time (mins):', valid_scores.iloc[i,2])                     
     print('')                                                                  
     i+=1                                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Show results                                                                 
 valid_scores                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Motivated by this, we will take LGBM and CatBoost to the final stage of         
modelling.                                                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Show best parameters from grid search                                        
 clf_best_params                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Modelling                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
We can finally train our best model on the whole training set using cross       
validation and ensembling predictions together to produce the most confident    
predictions.                                                                    

────────────────────────────────────────────────────────────────────────────────
Define best models                                                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Classifiers                                                                  
 best_classifiers = {                                                           
     "LGBM" : LGBMClassifier(**clf_best_params["LGBM"], random_state=0),        
     "CatBoost" : CatBoostClassifier(**clf_best_params["CatBoost"],             
 verbose=False, random_state=0),                                                
 }                                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
Cross validation and ensembling predictions                                     

────────────────────────────────────────────────────────────────────────────────
Predictions are ensembled together using soft voting. This averages the         
predicted probabilies to produce the most confident predictions.                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Number of folds in cross validation                                          
 FOLDS=10                                                                       
                                                                                
 preds=np.zeros(len(X_test))                                                    
 for key, classifier in best_classifiers.items():                               
     start = time.time()                                                        
                                                                                
     # 10-fold cross validation                                                 
     cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)         
                                                                                
     score=0                                                                    
     for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):               
         # Get training and validation sets                                     
         X_train, X_valid = X[train_idx], X[val_idx]                            
         y_train, y_valid = y[train_idx], y[val_idx]                            
                                                                                
         # Train model                                                          
         clf = classifier                                                       
         clf.fit(X_train, y_train)                                              
                                                                                
         # Make predictions and measure accuracy                                
         preds += clf.predict_proba(X_test)[:,1]                                
         score += clf.score(X_valid, y_valid)                                   
                                                                                
     # Average accuracy                                                         
     score=score/FOLDS                                                          
                                                                                
     # Stop timer                                                               
     stop = time.time()                                                         
                                                                                
     # Print accuracy and time                                                  
     print('Model:', key)                                                       
     print('Average validation accuracy:', np.round(100*score,2))               
     print('Training time (mins):', np.round((stop - start)/60,2))              
     print('')                                                                  
                                                                                
 # Ensemble predictions                                                         
 preds=preds/(FOLDS*len(best_classifiers))                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Submission                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
Let's look at the distribution of the predicted probabilities.                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.figure(figsize=(10,4))                                                     
 sns.histplot(preds, binwidth=0.01, kde=True)                                   
 plt.title('Predicted probabilities')                                           
 plt.xlabel('Probability')                                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
It is interesting to see that the models are either very confident or very      
unconfident but not much in between.                                            

────────────────────────────────────────────────────────────────────────────────
Post processing                                                                 

────────────────────────────────────────────────────────────────────────────────
Finally, we need to convert each predicted probability into one of the two      
classes (transported or not). The simplest way is to round each probability to  
the nearest integer (0 for False or 1 for True). However, assuming the train and
test sets have similar distributions, we can tune the classification threshold  
to obtain a similar proportion of transported/not transported in our predictions
as in the train set. Remember that the proportion of transported passengers in  
the train set was 50.4%.                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Proportion (in test set) we get from rounding                                
 print(np.round(100*np.round(preds).sum()/len(preds),2))                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
Our models seem to (potentially) overestimate the number of transported         
passengers in the test set. Let's try to bring that proportion down a bit.      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Proportion of predicted positive (transported) classes                       
 def preds_prop(preds_arr, thresh):                                             
     pred_classes=(preds_arr>=thresh).astype(int)                               
     return pred_classes.sum()/len(pred_classes)                                
                                                                                
 # Plot proportions across a range of thresholds                                
 def plot_preds_prop(preds_arr):                                                
     # Array of thresholds                                                      
     T_array=np.arange(0,1,0.001)                                               
                                                                                
     # Calculate proportions                                                    
     prop=np.zeros(len(T_array))                                                
     for i, T in enumerate(T_array):                                            
         prop[i]=preds_prop(preds_arr, T)                                       
                                                                                
     # Plot proportions                                                         
     plt.figure(figsize=(10,4))                                                 
     plt.plot(T_array, prop)                                                    
     target_prop=0.519         # Experiment with this value                     
     plt.axhline(y=target_prop, color='r', linestyle='--')                      
     plt.text(-0.02,0.45,f'Target proportion: {target_prop}', fontsize=14)      
     plt.title('Predicted target distribution vs threshold')                    
     plt.xlabel('Threshold')                                                    
     plt.ylabel('Proportion')                                                   
                                                                                
     # Find optimal threshold (the one that leads to the proportion being close 
 to target_prop)                                                                
     T_opt=T_array[np.abs(prop-target_prop).argmin()]                           
     print('Optimal threshold:', T_opt)                                         
     return T_opt                                                               
                                                                                
 T_opt=plot_preds_prop(preds)                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Classify test set using optimal threshold                                    
 preds_tuned=(preds>=T_opt).astype(int)                                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
Submit predictions                                                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Sample submission (to get right format)                                      
 sub=pd.read_csv('../input/spaceship-titanic/sample_submission.csv')            
                                                                                
 # Add predictions                                                              
 sub['Transported']=preds_tuned                                                 
                                                                                
 # Replace 0 to False and 1 to True                                             
 sub=sub.replace({0:False, 1:True})                                             
                                                                                
 # Prediction distribution                                                      
 plt.figure(figsize=(6,6))                                                      
 sub['Transported'].value_counts().plot.pie(explode=[0.1,0.1], autopct='%1.1f%% 
 shadow=True, textprops={'fontsize':16}).set_title("Prediction distribution")   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Output to csv                                                                
 sub.to_csv('submission.csv', index=False)                                      
                                                                                
