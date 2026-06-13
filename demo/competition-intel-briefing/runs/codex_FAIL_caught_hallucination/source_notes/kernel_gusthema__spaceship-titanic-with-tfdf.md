Reading kernel: gusthema/spaceship-titanic-with-tfdf
Cells: 61 (29 code, 32 markdown) | 11902 chars

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Spaceship Titanic Dataset with TensorFlow Decision Forests          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
This notebook walks you through how to train a baseline Random Forest model     
using TensorFlow Decision Forests on the Spaceship Titanic dataset made         
available for this competition.                                                 

Roughly, the code will look as follows:                                         

                                                                                
 import tensorflow_decision_forests as tfdf                                     
 import pandas as pd                                                            
                                                                                
 dataset = pd.read_csv("project/dataset.csv")                                   
 tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")  
                                                                                
 model = tfdf.keras.RandomForestModel()                                         
 model.fit(tf_dataset)                                                          
                                                                                
 print(model.summary())                                                         
                                                                                

Decision Forests are a family of tree-based models including Random Forests and 
Gradient Boosted Trees. They are the best place to start when working with      
tabular data, and will often outperform (or provide a strong baseline) before   
you begin experimenting with neural networks.                                   

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              Import the library                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 import tensorflow as tf                                                        
 import tensorflow_decision_forests as tfdf                                     
 import pandas as pd                                                            
 import numpy as np                                                             
 import seaborn as sns                                                          
 import matplotlib.pyplot as plt                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print("TensorFlow v" + tf.__version__)                                         
 print("TensorFlow Decision Forests v" + tfdf.__version__)                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                               Load the Dataset                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Load a dataset into a Pandas Dataframe                                       
 dataset_df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')          
 print("Full train dataset shape is {}".format(dataset_df.shape))               
                                                                                

────────────────────────────────────────────────────────────────────────────────
The data is composed of 14 columns and 8693 entries. We can see all 14          
dimensions of our dataset by printing out the first 5 entries using the         
following code:                                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Display the first 5 examples                                                 
 dataset_df.head(5)                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
There are 12 feature columns. Using these features your model has to predict    
whether the passenger is rescued or not indicated by the column Transported.    

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃             Let us quickly do a basic exploration of the dataset             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df.describe()                                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df.info()                                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   Bar chart for label column: Transported                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plot_df = dataset_df.Transported.value_counts()                                
 plot_df.plot(kind="bar")                                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                         Numerical data distribution                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Let us plot all the numerical columns and their value counts:                   

────────────────────────────────────────────────────────────────────────────────
                                                                                
 fig, ax = plt.subplots(5,1,  figsize=(10, 10))                                 
 plt.subplots_adjust(top = 2)                                                   
                                                                                
 sns.histplot(dataset_df['Age'], color='b', bins=50, ax=ax[0]);                 
 sns.histplot(dataset_df['FoodCourt'], color='b', bins=50, ax=ax[1]);           
 sns.histplot(dataset_df['ShoppingMall'], color='b', bins=50, ax=ax[2]);        
 sns.histplot(dataset_df['Spa'], color='b', bins=50, ax=ax[3]);                 
 sns.histplot(dataset_df['VRDeck'], color='b', bins=50, ax=ax[4]);              
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                             Prepare the dataset                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
We will drop both PassengerId and Name columns as they are not necessary for    
model training.                                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1)                  
 dataset_df.head(5)                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
We will check for the missing values using the following code:                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df.isnull().sum().sort_values(ascending=False)                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
This dataset contains a mix of numeric, categorical and missing features. TF-DF 
supports all these feature types natively, and no preprocessing is required.    

But this datatset also has boolean fields with missing values. TF-DF doesn't    
support boolean fields yet. So we need to convert those fields into int. To     
account for the missing values in the boolean fields, we will replace them with 
zero.                                                                           

In this notebook, we will replace null value entries with zero for numerical    
columns as well and only let TF-DF handle the missing values in categorical     
columns.                                                                        

Note: You can choose to let TF-DF handle missing values in numerical columns if 
need be.                                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] 
 dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa',            
 'VRDeck']].fillna(value=0)                                                     
 dataset_df.isnull().sum().sort_values(ascending=False)                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
Since, TF-DF cannot handle boolean columns, we will have to adjust the labels in
column Transported to convert them into the integer format that TF-DF expects.  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 label = "Transported"                                                          
 dataset_df[label] = dataset_df[label].astype(int)                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
We will also convert the boolean fields CryoSleep and VIP to int.               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df['VIP'] = dataset_df['VIP'].astype(int)                              
 dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
The value of column Cabin is a string with the format Deck/Cabin_num/Side. Here 
we will split the Cabin column and create 3 new columns Deck, Cabin_num and     
Side, since it will be easier to train the model on those individual data.      

Run the following command to split the column Cabin into columns Deck, Cabin_num
and Side                                                                        

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", 
 expand=True)                                                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
Remove original Cabin column from the dataset since it's not needed anymore.    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 try:                                                                           
     dataset_df = dataset_df.drop('Cabin', axis=1)                              
 except KeyError:                                                               
     print("Field does not exist")                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
Let us display the first 5 examples from the prepared dataset.                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 dataset_df.head(5)                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
Now let us split the dataset into training and testing datasets:                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def split_dataset(dataset, test_ratio=0.20):                                   
   test_indices = np.random.rand(len(dataset)) < test_ratio                     
   return dataset[~test_indices], dataset[test_indices]                         
                                                                                
 train_ds_pd, valid_ds_pd = split_dataset(dataset_df)                           
 print("{} examples in training, {} examples in testing.".format(               
     len(train_ds_pd), len(valid_ds_pd)))                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
There's one more step required before we can train the model. We need to convert
the datatset from Pandas format (pd.DataFrame) into TensorFlow Datasets format  
(tf.data.Dataset).                                                              

TensorFlow Datasets is a high performance data loading library which is helpful 
when training neural networks with accelerators like GPUs and TPUs.             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)     
 valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)     
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                Select a Model                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

There are several tree-based models for you to choose from.                     

 • RandomForestModel                                                            
 • GradientBoostedTreesModel                                                    
 • CartModel                                                                    
 • DistributedGradientBoostedTreesModel                                         

To start, we'll work with a Random Forest. This is the most well-known of the   
Decision Forest training algorithms.                                            

A Random Forest is a collection of decision trees, each trained independently on
a random subset of the training dataset (sampled with replacement). The         
algorithm is unique in that it is robust to overfitting, and easy to use.       

We can list the all the available models in TensorFlow Decision Forests using   
the following code:                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 tfdf.keras.get_all_models()                                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                             Configure the model                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

TensorFlow Decision Forests provides good defaults for you (e.g. the top ranking
hyperparameters on our benchmarks, slightly modified to run in reasonable time).
If you would like to configure the learning algorithm, you will find many       
options you can explore to get the highest possible accuracy.                   

You can select a template and/or set parameters as follows:                     

rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")    

Read more here.                                                                 

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                            Create a Random Forest                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Today, we will use the defaults to create the Random Forest Model. By default   
the model is set to train for a classification task.                            

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                                                                                
 rf = tfdf.keras.RandomForestModel()                                            
 rf.compile(metrics=["accuracy"]) # Optional, you can use this to include a lis 
 of eval metrics                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                               Train the model                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

We will train the model using a one-liner.                                      

Note: you may see a warning about Autograph. You can safely ignore this, it will
be fixed in the next release.                                                   

────────────────────────────────────────────────────────────────────────────────
                                                                                
 rf.fit(x=train_ds)                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                             Visualize the model                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

One benefit of tree-based models is that we can easily visualize them. The      
default number of trees used in the Random Forests is 300. We can select a tree 
to display below.                                                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)            
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Evaluate the model on the Out of bag (OOB) data and the validation dataset  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Before training the dataset we have manually seperated 20% of the dataset for   
validation named as valid_ds.                                                   

We can also use Out of bag (OOB) score to validate our RandomForestModel. To    
train a Random Forest Model, a set of random samples from training set are      
choosen by the algorithm and the rest of the samples are used to finetune the   
model.The subset of data that is not chosen is known as Out of bag data (OOB).  
OOB score is computed on the OOB data.                                          

Read more about OOB data here.                                                  

The training logs show the accuracy evaluated on the out-of-bag dataset         
according to the number of trees in the model. Let us plot this.                

Note: Larger values are better for this hyperparameter.                         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 import matplotlib.pyplot as plt                                                
 logs = rf.make_inspector().training_logs()                                     
 plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in  
 logs])                                                                         
 plt.xlabel("Number of trees")                                                  
 plt.ylabel("Accuracy (out-of-bag)")                                            
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
We can also see some general stats on the OOB dataset:                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 inspector = rf.make_inspector()                                                
 inspector.evaluation()                                                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
Now, let us run an evaluation using the validation dataset.                     

────────────────────────────────────────────────────────────────────────────────
                                                                                
 evaluation = rf.evaluate(x=valid_ds,return_dict=True)                          
                                                                                
 for name, value in evaluation.items():                                         
   print(f"{name}: {value:.4f}")                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                             Variable importances                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Variable importances generally indicate how much a feature contributes to the   
model predictions or quality. There are several ways to identify important      
features using TensorFlow Decision Forests. Let us list the available Variable  
Importances for Decision Trees:                                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f"Available variable importances:")                                      
 for importance in inspector.variable_importances().keys():                     
   print("\t", importance)                                                      
                                                                                

────────────────────────────────────────────────────────────────────────────────
As an example, let us display the important features for the Variable Importance
NUM_AS_ROOT.                                                                    

The larger the importance score for NUM_AS_ROOT, the more impact it has on the  
outcome of the model.                                                           

By default, the list is sorted from the most important to the least. From the   
output you can infer that the feature at the top of the list is used as the root
node in most number of trees in the random forest than any other feature.       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Each line is: (feature name, (index of the feature), importance score)       
 inspector.variable_importances()["NUM_AS_ROOT"]                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Submission                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Load the test dataset                                                        
 test_df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')              
 submission_id = test_df.PassengerId                                            
                                                                                
 # Replace NaN values with zero                                                 
 test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)  
                                                                                
 # Creating New Features - Deck, Cabin_num and Side from the column Cabin and   
 remove Cabin                                                                   
 test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/",       
 expand=True)                                                                   
 test_df = test_df.drop('Cabin', axis=1)                                        
                                                                                
 # Convert boolean to 1's and 0's                                               
 test_df['VIP'] = test_df['VIP'].astype(int)                                    
 test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)                        
                                                                                
 # Convert pd dataframe to tf dataset                                           
 test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)                       
                                                                                
 # Get the predictions for testdata                                             
 predictions = rf.predict(test_ds)                                              
 n_predictions = (predictions > 0.5).astype(bool)                               
 output = pd.DataFrame({'PassengerId': submission_id,                           
                        'Transported': n_predictions.squeeze()})                
                                                                                
 output.head()                                                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 sample_submission_df =                                                         
 pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')           
 sample_submission_df['Transported'] = n_predictions                            
 sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)     
 sample_submission_df.head()                                                    
                                                                                
