Reading kernel: fernaandodantas/lgbm-0-8066-score-top-7-solution
Cells: 66 (42 code, 24 markdown) | 13327 chars

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Libraries                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Space titanic competition 0.80664 Score                                       
                                                                                
                                                                                
 import pandas as pd                                                            
 import numpy as np                                                             
 import matplotlib.pyplot as plt                                                
 import seaborn as sns                                                          
 from sklearn.model_selection import                                            
 train_test_split,GridSearchCV,cross_val_score,StratifiedKFold                  
 import xgboost as xgb                                                          
 import category_encoders as ce                                                 
 from sklearn.impute import SimpleImputer                                       
 from sklearn.pipeline import Pipeline                                          
 from sklearn.pipeline import make_pipeline                                     
 from sklearn.compose import ColumnTransformer                                  
 from sklearn.metrics import accuracy_score, roc_auc_score, precision_score,    
 recall_score,f1_score, confusion_matrix                                        
 from category_encoders import TargetEncoder                                    
 from sklearn.impute import SimpleImputer,KNNImputer                            
 from lightgbm import LGBMClassifier                                            
                                                                                
 import warnings                                                                
                                                                                
 warnings.filterwarnings("ignore")                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                 Let’s Start                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
As you can see, after I loaded the dataset, I removed both the “Passenger Id”   
and “Name” columns. They are not going to provide any useful or important       
information to the prediction. Someone’s name or Id does not change the         
probability of being Transported.                                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")                  
                                                                                
 #After loading the dataset, I dropped the "PassengerId" and the "Name" column. 
                                                                                
 #But why? Knowing a passengers name or Id will not provide any usefull         
 information to the model.                                                      
                                                                                
 #Someone named "James" or with a certain Id number will not have a greater or  
 lower chance of surviving.                                                     
                                                                                
 #So, I just removed both columns.                                              
                                                                                
 df.drop(columns=["PassengerId","Name"],inplace=True)                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
Now, we are going to discuss a fundamental step I came across after trying to   
improve my score a thousand times. This step relies on exploring the “Cabin”    
column. Notice that the rows on the “Cabin” column follow a specific pattern.   
Something like this: “A/5/S”, “C/1/S”, “F/7/P”. And I decided to investigate it.
So, to make things simple I split the rows of the “Cabin” into three columns    
based on both slashes (”/”) of the rows. For example, the “A/5/S” row would be  
transformed into three new columns: The first one is named “cabin_code”         
referring to the character behind the first slash (A). The second one named     
“id_cabin” refers to the character behind the second slash (5). The third one   
named “cabin_sector” refers to the character after the second slash (S). And we 
end up with three new columns.                                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Splitting                                                                     
                                                                                
 df[["cabin_code","id_cabin","cabin_sector"]] = df["Cabin"].str.split("/", n=2, 
 expand=True)                                                                   
                                                                                
 df.head(4)                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
First of all, I noticed that “cabin_code” only has 8 different characters which 
means that the cabins are, somehow, divided into 8 sections.                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 cabinss = df.cabin_code.value_counts(1).sort_index()                           
 cabinss                                                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
Also, I asked myself if passengers from a specific section had a higher chance  
of being transported or if this statement was not true. With the plot below we  
can conclude that passengers from the B and C sections have a greater chance of 
surviving and passengers from the E section have a lower chance of surviving.   

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.figure(figsize=(12, 4.5))                                                  
 _= sns.countplot(data=df, x="cabin_code", hue="Transported", palette="coolwarm 
                                                                                

────────────────────────────────────────────────────────────────────────────────
I did the same thing with the “cabin_sector” column and also noticed that there 
was a difference between the sectors. Passengers from the P sector have a lower 
chance of being transported, while in the S sector, the opposite happens.       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.figure(figsize=(10, 4.5))                                                  
 _= sns.countplot(data=df, x="cabin_sector", hue="Transported",                 
 palette="coolwarm")                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
This means that this exploration of the original “Cabin” column is worth it     
since new insights are being added to the model.                                

Now, we can finally delete the “Cabin” column. It will not provide any useful   
information for the model anymore. We have already extracted everything useful  
from it.                                                                        

I also removed the “cabin_id” and the column that I had created. As I said      
before, the Id will not interfere with the model’s predictive ability.          

So used: df.drop(columns=[“Cabin”,”id_cabin”], inplace=True) to drop both       
columns                                                                         

Before splitting our data, the “Transported” column must be in a binary format. 
As you can see, I switched “True” for 1 and “False” for 0.                      

Binary transformation: df[“Transported”] = df[“Transported”].map({True:1,       
False:0})                                                                       

I also removed every row that had missing values in the “cabin_code” column.    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 pop_id_cabin = df.pop("id_cabin")                                              
 df.insert(3, 'id_cabin', pop_id_cabin)  # Insert column 'C' at the beginning   
                                                                                
 pop_id_cabin = df.pop("cabin_sector")                                          
 df.insert(3, 'cabin_sector', pop_id_cabin)  # Insert column 'C' at the beg     
                                                                                
 pop_id_cabin = df.pop("cabin_code")                                            
 df.insert(3, 'cabin_code', pop_id_cabin)  # Insert column 'C' at the beg       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
                                                                                
 #BINARY TRANSFORMATION                                                         
 df["Transported"] = df["Transported"].map({True:1, False:0})                   
                                                                                
                                                                                
 #DROPPING COLUMNS                                                              
 df.drop(columns=["Cabin","id_cabin"], inplace=True)                            
                                                                                
                                                                                
 #DROPPING NULLS                                                                
 df.dropna(subset=["cabin_code"], inplace=True)                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
Now, we can finally split the data and proceed to develop our model.            

After splitting in train and test, I separated the test data into two           
categories: numerical and categorical. Why is that? We are going to perform     
different operations depending on the type of the variable. Categorical data    
must be encoded since most models are not able to understand categorical values 
and it must be converted to numerical values. Also, we are going to apply       
different techniques to fill the null values in our dataset, but I will talk    
more about it later on.                                                         

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                Splitting Data                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Define X and y                                                                
                                                                                
 X = df.iloc[:,0:12]                                                            
 y = df["Transported"]                                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Data splitting                                                                
                                                                                
 X_train, X_test, y_train, y_test =                                             
 train_test_split(X,y,random_state=42,test_size=0.25)                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Separate categorical and numerical                                            
                                                                                
 cat_feat = np.array([coluna for coluna in X_train.columns if                   
 X_train[coluna].dtype.name == 'object'])                                       
                                                                                
 num_feat = np.array([coluna for coluna in X_train.columns if coluna not in     
 cat_feat])                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
We can now create our pipeline. There are going to be two pipelines: one is     
going to handle the categorical data and the other one is going to handle       
numerical data. The missing values of the categorical data will be filled with  
the most frequent value (mode) and after the Target Encoder will be applied to  
transform categorical variables into numerical variables. The numerical data    
missing values will be filled with a strategy called K-nearest neighbors, which 
uses the Euclidean distance between the data points to find the best number to  
fill the missing values. If don’t know how this Pipeline technique works, I     
recommend you check my article about Pipelines.                                 
("https://medium.com/@fernandao.lacerda.dantas/boost-your-pipelines-with-columnt
ransformer-b2c009db096f")                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Categorical and numerical pipelines                                           
                                                                                
                                                                                
 cat_pipe = Pipeline([("imputer_cat",                                           
 SimpleImputer(strategy="most_frequent")),("encoder", ce.TargetEncoder()),      
                     ])                                                         
                                                                                
 num_pipe = Pipeline([("imputer_num", KNNImputer(n_neighbors=3))])              
                                                                                

────────────────────────────────────────────────────────────────────────────────
And with column transformer, we can attach both transformations to one variable 
that I named “transformer”. Observe that we also have to specify the type of    
data to which the pipeline will be applied to: “cat_pipe” will be applied to    
“cat_feat” and “num_pipe” will be applied to “num_feat”, meaning the categorical
pipeline will take care of the categorical data and the numerical pipeline will 
take care of the numerical data.                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Using ColumnTransformer                                                       
                                                                                
                                                                                
 transformer = ColumnTransformer([("num_trans", num_pipe, num_feat),            
                             ("cat_trans", cat_pipe, cat_feat)])                
                                                                                

────────────────────────────────────────────────────────────────────────────────
After finishing the pipelines, we have to apply the transformations to our data.
We use “fit.transform” in the “X_train” data to make the model “learn” the      
transformations and apply “transform” in the “X_test” data.                     

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # ".fit_transform" in train data"                                              
                                                                                
 # ".transform" in test data"                                                   
                                                                                
 X_train_transformed = transformer.fit_transform(X_train, y_train)              
 X_test_transformed = transformer.transform(X_test)                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
In the next step, we are going to perform a Stratified Cross-Validation to      
select the best tree-based model we will use. We are going to try three models: 
LGBMClassifier, XGBoost and XGBoost (booster=”gblinear”). And based on the      
accuracy, the cross-validation will give the mean and the standard deviation of 
the performance of each model.                                                  

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                               Cross Validation                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 models = []                                                                    
                                                                                
 models.append(("xgb",xgb.XGBClassifier()))                                     
 models.append(("xgbgblinear",xgb.XGBClassifier(booster="gblinear")))           
 models.append(("LGBM",LGBMClassifier(verbose=-1)))                             
                                                                                
                                                                                
 print(models)                                                                  
                                                                                
 results = dict()                                                               
                                                                                
 for  name, model in models:                                                    
     skf = StratifiedKFold(n_splits = 5, random_state=None)                     
     cv_results = cross_val_score(model,X_train_transformed,y_train,cv=skf,     
 scoring="accuracy")                                                            
     results[name]= (cv_results.mean(), cv_results.std())                       
                                                                                
 print("name     results.mean     results.std")                                 
                                                                                
 for key,value in results.items():                                              
     print(key,value)                                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
We can see that LGBMClassifier had the best performance, therefore it will be   
the model used.                                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 lgbmc = LGBMClassifier()                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
However, the model itself will not be enough to provide great accuracy. Thus, we
now have to perform what is called hyperparameter tuning to make the model more 
precise.                                                                        

We set the parameters we want to test and using sklearn’s GridSearchCV we will  
obtain the best hyperparameters. GridSearchCV will test the parameters that we  
want and will show us the combination that has the best performance.            

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                            Hyperparameter Tunning                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
                                                                                
                                                                                
 lgbm_params = {"n_estimators":[100,200,300],                                   
                "learning_rate":[0.01,0.05,0.1,0.3],                            
                "num_leaves":[20,50,80,100],                                    
               "verbose":[-1]}                                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 grid_search = GridSearchCV (estimator = lgbmc,                                 
                             param_grid = lgbm_params,                          
                             n_jobs=-1,                                         
                             cv = 5,                                            
                             scoring="accuracy",                                
                            error_score='raise')                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 grid_result = grid_search.fit(X_train_transformed, y_train);                   
                                                                                
                                                                                
 final_model = lgbmc.set_params(**grid_result.best_params_)                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
With the best model optimized, we can finally train our model and obtain our    
predictions.                                                                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
                                                                                
 #training the model                                                            
 final_model.fit(X_train_transformed, y_train)                                  
                                                                                
                                                                                
 #predictions                                                                   
 y_pred = final_model.predict(X_test_transformed)                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
After obtaining our predictions, we have to test our model using metrics such as
recall, precision, f1 score and accuracy. The data frame below shows us some of 
those metrics and we can conclude that the model is having a great performance. 

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                    Scores                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 precision = precision_score(y_pred, y_test)                                    
 accuracy = accuracy_score(y_pred, y_test)                                      
 recall = precision_score(y_pred, y_test)                                       
 f1 = f1_score(y_pred, y_test)                                                  
                                                                                
 score = []                                                                     
 score.append(("precision", precision))                                         
 score.append(("accuracy",accuracy))                                            
 score.append(("recall",recall))                                                
 score.append(("f1",f1))                                                        
                                                                                
 score= pd.DataFrame(score)                                                     
 score.rename(columns={0: "Metric", 1:"Result"}, inplace=True)                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 #Metrics obtained                                                              
                                                                                
 display(score)                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f"Precision: {score.iloc[0,1]:.4f}")                                     
 print(f"Accuracy: {score.iloc[1,1]:.4f}")                                      
 print(f"F1_Score: {score.iloc[2,1]:.4f}")                                      
 print(f"Recall: {score.iloc[3,1]:.4f}")                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────

 • Precision: 0.8470                                                            
 • Accuracy: 0.8112                                                             
 • F1_Score: 0.8470                                                             
 • Recall: 0.8163                                                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 confusions_matrix = confusion_matrix(y_pred, y_test)                           
                                                                                
 from sklearn.metrics import ConfusionMatrixDisplay                             
                                                                                
 conf_disp = ConfusionMatrixDisplay(confusion_matrix=confusions_matrix)         
                                                                                
 conf_disp.plot()                                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Submission                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 testecsv = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 testecsv.head(6)                                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 testecsv[["cabin_code","id_cabin","cabin_sector"]] =                           
 testecsv["Cabin"].str.split("/", n=2, expand=True)                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 testecsv.drop(columns=["id_cabin"], inplace=True)                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X_teste = testecsv.drop(columns=["PassengerId","Cabin","Name"])                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 testecsv.info()                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 model = final_model                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X.dropna(subset=["cabin_code"], inplace=True)                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 X = transformer.fit_transform(X,y)                                             
 X_teste = transformer.transform(X_teste)                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 model.fit(X, y)                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 y_predz = model.predict(X_teste)                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 subimisspace = pd.Series(index = testecsv["PassengerId"].values, data = y_pred 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 subimisspace = subimisspace.reset_index()                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 subimisspace = pd.DataFrame(subimisspace)                                      
 subimisspace                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 subimisspace[0]=subimisspace[0].map({1:"True", 0:"False"})                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 subimisspace.rename(columns = {"index":"PassengerId", 0:"Transported"},        
 inplace=True)                                                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 subimisspace.to_csv("testy.csv", index=False)                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 subimisspace                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
The strategies that I mention in this article led me to a 0.80664 score in the  
competition and I am sure you can improve my model to achieve an even higher    
score with your knowledge!                                                      

If you enjoyed this article, don’t forget to support me or hit me with a follow!

See you in the next article!                                                    

-Fernando Dantas                                                                
