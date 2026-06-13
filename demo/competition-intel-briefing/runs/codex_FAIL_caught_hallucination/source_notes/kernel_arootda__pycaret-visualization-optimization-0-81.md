Reading kernel: arootda/pycaret-visualization-optimization-0-81
Cells: 93 (52 code, 41 markdown) | 28408 chars


────────────────────────────────────────────────────────────────────────────────
🌆 image.png                                                                    

────────────────────────────────────────────────────────────────────────────────
Hello, I'm a student studying machine learning hard.                            

We have to solve the problem of the Titanic spacecraft in 2912. Many passengers 
were moved to a different dimension in a situation where they collided with a   
space-time anomalies. We have to track down the missing person.                 

This competition deals with space Titanic-themed categorization problems. When I
looked into the data, I thought it was very similar to the Titanic competition  
that I knew well. When I approached a problem, I thought it was important to    
design a model quickly and easily, compare it even if it wasn't perfect, and    
produce results and draw insights. Previously, I mainly used the optuna library,
but recently there seem to be a lot of Pycaret users, so I thought I'd try using
this library. I hope to study with people who are new to pycaret or who are     
curious about how to optimize through pycaret. I decided to focus on the        
following in this competition.                                                  

 1 EDA: I tried to explore in detail by variable. In particular, I thought about
   what variables would help me solve the classification problem and decided    
   what kind of processing is needed.                                           
 2 FE: I tried feature engineering based on the results of the last EDA.        
 3 Modeling: Demonstrates the effective design and optimization of models using 
   pycaret. We compared various optimization libraries and methods and tried to 
   achieve the best performance using ensemble techniques.                      

Through this process, I got a score of 0.81. If you move this laptop as it is,  
you will get a score of 0.8097. Currently, the resulting hyperparameters are    
fixed, so if you want to optimize them again, you need to check the annotations 
in the modeling section.                                                        

Information on the pycaret can be obtained from the following sites.            
https://pycaret.org/                                                            

🌆 image.png                                                                    

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              📌 Import Modules                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                    Modules                                     

────────────────────────────────────────────────────────────────────────────────
                                                                                
 !pip install pycaret                                                           
 from pycaret.classification import *                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 import numpy as np                                                             
 import pandas as pd                                                            
 import matplotlib.pyplot as plt                                                
 import seaborn as sns                                                          
 import warnings                                                                
 warnings.filterwarnings('ignore')                                              
                                                                                
 from sklearn.impute import SimpleImputer                                       
 from sklearn.metrics import accuracy_score                                     
 from sklearn.preprocessing import LabelEncoder                                 
 from sklearn.model_selection import StratifiedKFold, train_test_split          
                                                                                
 import plotly.express as px                                                    
 from plotly.subplots import make_subplots                                      
 import plotly.graph_objects as go                                              
                                                                                
 PALETTE=['lightcoral', 'lightskyblue', 'gold', 'sandybrown', 'navajowhite',    
         'khaki', 'lightslategrey', 'turquoise', 'rosybrown', 'thistle', 'pink' 
 sns.set_palette(PALETTE)                                                       
 BACKCOLOR = '#f6f5f5'                                                          
                                                                                
 from IPython.core.display import HTML                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                                  User Modules                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def multi_table(table_list):                                                   
     return HTML(                                                               
         f"<table><tr> {''.join(['<td>' + table._repr_html_() + '</td>' for tab 
 in table_list])} </tr></table>")                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def cat_dist(data, var, hue, msg_show=True):                                   
     total_cnt = data[var].count()                                              
     f, ax = plt.subplots(1, 2, figsize=(25, 8))                                
     hues = [None, hue]                                                         
     titles = [f"{var}'s distribution", f"{var}'s distribution by {hue}"]       
                                                                                
     for i in range(2):                                                         
         sns.countplot(data[var], edgecolor='black', hue=hues[i], linewidth=1,  
 ax=ax[i], data=data)                                                           
         ax[i].set_xlabel(var, weight='bold', size=13)                          
         ax[i].set_ylabel('Count', weight='bold', size=13)                      
         ax[i].set_facecolor(BACKCOLOR)                                         
         ax[i].spines[['top', 'right']].set_visible(False)                      
         ax[i].set_title(titles[i], size=15, weight='bold')                     
         for patch in ax[i].patches:                                            
             x, height, width = patch.get_x(), patch.get_height(),              
 patch.get_width()                                                              
             if msg_show:                                                       
                 ax[i].text(x + width / 2, height + 3, f'{height} \n({height /  
 total_cnt * 100:2.2f}%)', va='center', ha='center', size=12, bbox={'facecolor' 
 'white', 'boxstyle': 'round'})                                                 
     plt.show()                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 def continuous_dist(data, x, y):                                               
     f, ax = plt.subplots(1, 4, figsize=(35, 10))                               
     sns.histplot(data=train, x=y, hue=x, ax=ax[0], element='step')             
     sns.violinplot(x=data[x], y=data[y], ax=ax[1], edgecolor='black',          
 linewidth=1)                                                                   
     sns.boxplot(x=data[x], y=data[y], ax=ax[2])                                
     sns.stripplot(x=data[x], y=data[y], ax=ax[3])                              
     for i in range(4):                                                         
         ax[i].spines[['top','right']].set_visible(False)                       
         ax[i].set_xlabel(x, weight='bold', size=20)                            
         ax[i].set_ylabel(y, weight='bold', size=20)                            
         ax[i].set_facecolor(BACKCOLOR)                                         
     f.suptitle(f"{y}'s distribution by {x}", weight='bold', size=25)           
     plt.show()                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                 📌 Read Data                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train = pd.read_csv("../input/spaceship-titanic/train.csv")                    
 test = pd.read_csv("../input/spaceship-titanic/test.csv")                      
 submission = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")   
                                                                                
 all_data = pd.concat([train, test], axis=0)                                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                    📌 EDA                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

────────────────────────────────────────────────────────────────────────────────

                                1. Check Data🔎                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 all_data.head(10).style.background_gradient()                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(f'\033[32mtrain size : {train.shape[0]} x {train.shape[1]}')             
 print(f'\033[32mtest size : {test.shape[0]} x {test.shape[1]}')                
 print(f'\033[32mtotal size : {all_data.shape[0]} x {all_data.shape[1]}')       
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                                2. Description🔎                                

When I start the EDA, I first check the description of each variable to see if  
there are any types and measurements.                                           

────────────────────────────────────────────────────────────────────────────────

                                                                                
  Variable                                Definition                            
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  PassengerId                             A unique Id for each passenger. Each  
                                          Id takes the form gggg_pp where gggg  
                                          indicates a group the passenger is    
                                          travelling with and pp is their       
                                          number within the group. People in a  
                                          group are often family members, but   
                                          not always.                           
  HomePlanet                              The planet the passenger departed     
                                          from, typically their planet of       
                                          permanent residence.                  
  CryoSleep                               Indicates whether the passenger       
                                          elected to be put into suspended      
                                          animation for the duration of the     
                                          voyage. Passengers in cryosleep are   
                                          confined to their cabins.HomePlanet   
  Cabin                                   The cabin number where the passenger  
                                          is staying. Takes the form            
                                          deck/num/side, where side can be      
                                          either P for Port or S for            
                                          Starboard.                            
  Destination                             The planet the passenger will be      
                                          debarking to.                         
  Age                                     The age of the passenger.             
  VIP                                     Whether the passenger has paid for    
                                          special VIP service during the        
                                          voyage.                               
  RoomService, FoodCourt, ShoppingMall,   Amount the passenger has billed at    
  Spa, VRDeck                             each of the Spaceship Titanic's many  
                                          luxury amenities.                     
  Name                                    The first and last names of the       
                                          passenger.                            
  Transported                             Whether the passenger was             
                                          transported to another dimension.     
                                          This is the target, the column you    
                                          are trying to predict.                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 all_data.columns                                                               
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 all_data.dtypes                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 all_data.info()                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 all_data.isnull().sum()                                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 multi_table([pd.DataFrame(all_data[i].value_counts()) for i in all_data.column 
 if i != 'Age'])                                                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
If you look at it so far, you can see exactly which variables are nominal and   
which are continuous. Additional basic statistics can be found for numeric      
variables. I checked the statistics of the training data, the test data, and the
total data.                                                                     

────────────────────────────────────────────────────────────────────────────────
                                                                                
 nominal_vars = ['HomePlanet', 'CryoSleep', 'Cabin', 'Desination', 'VIP', 'Name 
 continuous_vars = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',   
 'VRDeck']                                                                      
 target = 'Transported'                                                         
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train_st = train[continuous_vars].describe()                                   
 test_st = test[continuous_vars].describe()                                     
 all_st = all_data[continuous_vars].describe()                                  
 multi_table([all_st, train_st, test_st])                                       
                                                                                

────────────────────────────────────────────────────────────────────────────────

We have confirmed variable descriptions/data formats/missing values/categories, 
etc. I drew the following conclusion. The following conclusions can be used for 
future Feature Engineering.                                                     

PassengerId: This variable is a variable that is separated by _, such as        
0001_01. We can also decompose this variable to obtain a new one. I won't use it
for analysis as it is because of the high oil price.                            
Home Planet: Nominal variable separated from Earth/Europa/Mars. Earth is very   
important. 288 missing values exist and must be processed. We need to check the 
percentage of passenger space movement by the Home Planet.                      
CryoSleep: This variable is binary data. The proportion of False is high, and   
310 missing values exist. It goes through a Home Planet-like process.           
Cabin: This variable has Deck and Side recorded relative to /. Therefore, I     
think you can separate strings to obtain new variables. There is still a missing
value.                                                                          
Destination: Categorized into three types. You should check the correlation with
variables such as Home Planet and Cabin.                                        
Age: The distribution must be determined by numerical variables. In particular, 
it is important to determine the distribution of dependent variables.           
RoomService~VRDeck:Most of these variables are recorded at 0. These operations  
can be used to create new variables because they are all related to charges.    
This information should be used to handle missing values because of the large   
bias.                                                                           
Name: Passenger names do not seem to have a significant impact on analysis.     
Transported: The dependent variable required to resolve this problem. Binary    
variables.                                                                      


────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                               3. Missing Value🔎                               

Most variables had missing values. These are subject to preprocessing.          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 import missingno as msno                                                       
 msno.matrix(all_data)                                                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 msno.bar(all_data)                                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────

Most variables have missing values. We must solve this problem. The missing     
value processing methods are broadly divided into the following:                

 1 Remove missing values                                                        
 2 Replacement of missing values                                                

I prefer alternative methods to removal methods. This is because information    
loss can occur. A good way to replace it is to replace continuous variables with
mean, median, and categorical variables with most free values. Another method is
to understand why missing values of variables occur and replace them directly   
with specific values. For example, the missing value of the Shopping Mall is    
likely to be paid for, so I can replace it with zero.I will then proceed with   
this task in the preprocessing part.                                            


────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                               3. Detail Explore                                

Explore in detail for each variable.                                            

────────────────────────────────────────────────────────────────────────────────
                        Transported(Dependent, Nominal)                         

The target value is a binary label consisting of True and False. Therefore, it  
is necessary to make sure that it is balanced. I checked the number and         
percentage of each category using countplot. When I checked, I found that True  
was 50.36% and False was 49.64%.                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.subplots(figsize=(25, 10))                                                 
 plt.pie(train.Transported.value_counts(), shadow=True, explode=[.03,.03],      
 autopct='%1.1f%%', textprops={'fontsize': 20, 'color': 'white'})               
 plt.title('Transported Distribution', size=20)                                 
 plt.legend(['False', 'True'], loc='best', fontsize=12)                         
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                              HomePlanet (Nominal)                              

These variables are nominal variables, so we decided to check the distribution  
with countplot. To see if this variable can explain the dependent variable well,
we checked the distribution of the dependent variable again. I confirmed the    
following results.                                                              

 1 Among the Homeplanets, 54.19% of the earth's share is the highest.           
 2 The ratio of Europa to Mars is almost the same.                              
 3 Among Europa, the percentage of Transported is certainly high.               
 4 The false percentage of Transported in Earth is certainly high.              
 5 There is not much difference in Transported among Mars.                      

= > More than half of people belong to Earth. In addition, the difference       
between Earth and Europa's transported ratio is certain, so the sorting         
algorithm can work well. Therefore, you will be able to use the variable well to
solve this problem.                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 cat_dist(train, var='HomePlanet', hue='Transported')                           
 train.pivot_table(index="HomePlanet", values="Transported", aggfunc=['count',  
 'sum', 'mean']).style.background_gradient(vmin=0)                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                              CryoSleep (Nominal)                               

CryoSleep is a nominal variable. Therefore, you can view the distribution in    
Countplot. It can be seen that CryoSleep has a high false percentage of 64.17%. 
You should try to see how the CryoSleep value affects Transported.              

If CryoSleep is False, the Transported false ratio is approximately 68%. When   
True, the Transported ratio is approximately 80%. => Verified that the value of 
CryoSleep clearly distinguishes Transported. Therefore, just looking at a       
person's CryoSleep gives you a 50% chance of matching Transported to True or    
False. Classification algorithms will also take a big hint from looking at this 
variable.                                                                       

────────────────────────────────────────────────────────────────────────────────
                                                                                
 cat_dist(train, var='CryoSleep', hue='Transported')                            
 train.pivot_table(index="CryoSleep", values="Transported", aggfunc=['count',   
 'sum', 'mean']).style.background_gradient(vmin=0)                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                                 Cabin(Nominal)                                 

There are so many categories of Cabin that I thought it would be meaningless to 
check them individually. I was able to find something in common with Cabin's    
name, but I thought the first word and the last word were Cabin's type. That's  
why the distribution was confirmed based on them.                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 tmp = train.copy()                                                             
 tmp['Deck'] = train.Cabin.apply(lambda x:str(x)[:1])                           
 tmp['side'] = train.Cabin.apply(lambda x:str(x)[-1:])                          
                                                                                
 cat_dist(tmp, var='Deck', hue='Transported')                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 cat_dist(tmp, var='side', hue='Transported')                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                             Destination (Nominal)                              

────────────────────────────────────────────────────────────────────────────────
                                                                                
 cat_dist(train, var='Destination', hue='Transported')                          
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                                 VIP (Nominal)                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 cat_dist(train, var='VIP', hue='Transported')                                  
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                                Age (continuous)                                

Age is a continuous value, so we checked the distribution in four flows. I      
understood the following.                                                       

 1 Transported is not significantly affected by age.                            
 2 However, it has been confirmed that the proportion of transported in young   
   sections is high, and that this is not the case in people in their 20s.      
 3 Therefore, I thought that categorizing data into categorical data by age     
   would create more meaningful variables.                                      

────────────────────────────────────────────────────────────────────────────────
                                                                                
 continuous_dist(train, 'Transported', 'Age')                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 tmp = train.copy()                                                             
 tmp['AgeBin'] = 7                                                              
 for i in range(6):                                                             
     tmp.loc[(tmp.Age >= 10*i) & (tmp.Age < 10*(i + 1)), 'AgeBin'] = i          
 cat_dist(tmp, var='AgeBin', hue='Transported', msg_show=False)                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                           Other Continuous Variables                           

They have a one-sided distribution. Most people didn't pay for this service.    
These can be extracted from variables such as total consumption.                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 f, ax = plt.subplots(1, 5, figsize=(40, 7))                                    
 sns.distplot(train.RoomService, ax=ax[0])                                      
 sns.distplot(train.FoodCourt, ax=ax[1])                                        
 sns.distplot(train.ShoppingMall, ax=ax[2])                                     
 sns.distplot(train.Spa, ax=ax[3])                                              
 sns.distplot(train.VRDeck, ax=ax[4])                                           
 plt.show()                                                                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────

                             4. Multinomial Explore                             

You should also consider the relationship between two or more variable          
combinations and dependent variables. For example, if you look at Home Planet   
and CryoSleep together, you can better define the relationship between the      
dependent variables:                                                            

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Heatmap can visualize continuous values (or binary variables) in categories  
 and categories.                                                                
 plt.subplots(figsize=(10, 5))                                                  
 g = sns.heatmap(train.pivot_table(index='HomePlanet', columns='CryoSleep',     
 values='Transported'), annot=True, cmap="YlGnBu")                              
 g.set_title('Transported ratio by HomePlanet and CryoSleep', weight='bold',    
 size=15)                                                                       
 g.set_xlabel('CryoSleep', weight='bold', size=13)                              
 g.set_ylabel('HomePlanet', weight='bold', size=13)                             
 plt.show()                                                                     
                                                                                
 pd.crosstab([train.CryoSleep, train.Transported],                              
 train.HomePlanet,margins=True).style.background_gradient()                     
                                                                                

────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.subplots(figsize=(10, 5))                                                  
 g = sns.heatmap(train.pivot_table(index='HomePlanet', columns='Destination',   
 values='Transported'), annot=True, cmap="YlGnBu")                              
 g.set_title('Transported ratio by HomePlanet and Destination', weight='bold',  
 size=15)                                                                       
 g.set_xlabel('Destination', weight='bold', size=13)                            
 g.set_ylabel('HomePlanet', weight='bold', size=13)                             
 plt.show()                                                                     
                                                                                
 pd.crosstab([train.Destination, train.Transported],                            
 train.HomePlanet,margins=True).style.background_gradient()                     
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 plt.subplots(figsize=(10, 5))                                                  
 g = sns.heatmap(train.pivot_table(index='CryoSleep', columns='Destination',    
 values='Transported'), annot=True, cmap="YlGnBu")                              
 g.set_title('Transported ratio by CryoSleep and Destination', weight='bold',   
 size=15)                                                                       
 g.set_xlabel('Destination', weight='bold', size=13)                            
 g.set_ylabel('CryoSleep', weight='bold', size=13)                              
 plt.show()                                                                     
                                                                                
 pd.crosstab([train.CryoSleep, train.Transported],                              
 train.Destination,margins=True).style.background_gradient()                    
                                                                                

────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                               📌 Preprocessing                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Let's move on with the preprocessing. The preprocessing process is as follows:  

────────────────────────────────────────────────────────────────────────────────
🌆 fe.png                                                                       

────────────────────────────────────────────────────────────────────────────────
                            Imputing Missing values                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Replace categorical variables with specific values (False, None) or freeest  
 values.                                                                        
 all_data['CryoSleep'].fillna(False, inplace=True)                              
 all_data['Cabin'].fillna('None', inplace=True)                                 
 all_data['VIP'].fillna(all_data.VIP.mode()[0], inplace=True)                   
 all_data['HomePlanet'].fillna(all_data.HomePlanet.mode()[0], inplace=True)     
 all_data['Destination'].fillna(all_data.Destination.mode()[0], inplace=True)   
                                                                                
 # Replace continuous variables with specific values (0) or averages.           
 all_data['Age'].fillna(all_data.Age.mean(), inplace=True)                      
 all_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] =\     
 all_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',                   
 'VRDeck']].fillna(0)                                                           
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                           Create Derivative variable                           

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # As mentioned earlier, create a new variable by decomposing strings in Cabin  
 and PassengerId.                                                               
 all_data['Deck'] = all_data.Cabin.apply(lambda x:str(x)[:1])                   
 all_data['Side'] = all_data.Cabin.apply(lambda x:str(x)[-1:])                  
 all_data['PassengerGroup'] = all_data['PassengerId'].apply(lambda x:           
 x.split('_')[0])                                                               
 all_data['PassengerNo'] = all_data['PassengerId'].apply(lambda x:              
 x.split('_')[1])                                                               
                                                                                
 # Generate new variables based on the amount of money used for various service 
 all_data['TotalSpend'] = all_data['RoomService'] + all_data['FoodCourt'] +     
 all_data['ShoppingMall'] + all_data['Spa'] + all_data['VRDeck']                
 all_data['PctRoomService'] = all_data['RoomService']/all_data['TotalSpend']    
 all_data['PctFoodCourt'] = all_data['FoodCourt']/all_data['TotalSpend']        
 all_data['PctShoppingMall'] = all_data['ShoppingMall']/all_data['TotalSpend']  
 all_data['PctSpa'] = all_data['Spa']/all_data['TotalSpend']                    
 all_data['PctVRDeck'] = all_data['VRDeck']/all_data['TotalSpend']              
                                                                                
 # Create new variables by dividing age groups.                                 
 all_data['AgeBin'] = 7                                                         
 for i in range(6):                                                             
     all_data.loc[(all_data.Age >= 10*i) & (all_data.Age < 10*(i + 1)), 'AgeBin 
 = i                                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Replaces the missing value that occurred when generating the derived variabl 
 fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa',    
 'PctVRDeck']                                                                   
 all_data[fill_cols] = all_data[fill_cols].fillna(0)                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                 Drop Variables                                 

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # Remove unnecessary variables.                                                
 all_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)          
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
                                    Encoding                                    

🌆 image.png                                                                    
Typically, categorical variable encoding is divided into one-hot encoding and   
label encoding. The anchor is one-hot encoding for nominal variables and label  
encoding for ordered variables. However, it uses a tree-based boost algorithm to
perform label encoding simply.                                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 for col in all_data.columns[all_data.dtypes == object]:                        
     if col != 'Transported':                                                   
         le = LabelEncoder()                                                    
         all_data[col] = le.fit_transform(all_data[col])                        
                                                                                
 all_data['CryoSleep'] = all_data['CryoSleep'].astype('int')                    
 all_data['VIP'] = all_data['VIP'].astype('int')                                
                                                                                

────────────────────────────────────────────────────────────────────────────────
                               Split Train / Test                               

────────────────────────────────────────────────────────────────────────────────
                                                                                
 train, X_test = all_data.iloc[:train.shape[0]],                                
 all_data.iloc[train.shape[0]:].drop(['Transported'], axis=1)                   
 X_train, y_train = train.drop(['Transported'], axis=1), train['Transported']   
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                          📌 Modeling and Optimizing                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

🌆 image.png                                                                    
Previously, data was verified and preprocessed. Finally, I made the final data. 
I used pycaret to proceed with the test method and optimization of models       
suitable for this data. The first thing you need to do is set up your           
environment. You can use the setup function to proceed with the configuration of
your environment. The most important thing is to pass the data you create and   
the target variable. Other options are optional.                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 s = setup(data=train,                                                          
           session_id=7010,                                                     
           target='Transported',                                                
           train_size=0.99,                                                     
           fold_strategy='stratifiedkfold',                                     
           fold=5,                                                              
           fold_shuffle=True,                                                   
           silent=True,                                                         
           ignore_low_variance=True,                                            
           remove_multicollinearity = True,                                     
           normalize = True,                                                    
           normalize_method = 'robust',)                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
Compare_model allows you to compare the results of learning given data by model.
I checked the top four models.                                                  

────────────────────────────────────────────────────────────────────────────────
                                                                                
 top4 = compare_models(n_select=4)                                              
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 print(top4[0])                                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
The best model is Catboost. I have been optimizing Catboost models in various   
ways, and I have also evaluated the performance of models with the top four     
models as ensemble.                                                             

────────────────────────────────────────────────────────────────────────────────
                                                                                
 !pip install scikit-optimize                                                   
 !pip install tune-sklearn ray[tune]                                            
 import optuna                                                                  
 !pip install hpbandster ConfigSpace                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
The following model is the best model I obtained while tuning myself using the  
annotation code below. I hope you can refer to it. If you derive results after  
learning in this model, you can get a score of about 0.8097.                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 catboost_best = create_model('catboost', nan_mode= 'Min',                      
                          eval_metric='Logloss',                                
                          iterations=1000,                                      
                          sampling_frequency='PerTree',                         
                          leaf_estimation_method='Newton',                      
                          grow_policy='SymmetricTree',                          
                          penalties_coefficient=1,                              
                          boosting_type='Plain',                                
                          model_shrink_mode='Constant',                         
                          feature_border_type='GreedyLogSum',                   
                          l2_leaf_reg=3,                                        
                          random_strength=1,                                    
                          rsm=1,                                                
                          boost_from_average=False,                             
                          model_size_reg=0.5,                                   
                          subsample=0.800000011920929,                          
                          use_best_model=False,                                 
                          class_names=[0, 1],                                   
                          depth=6,                                              
                          posterior_sampling=False,                             
                          border_count=254,                                     
                          classes_count=0,                                      
                          auto_class_weights='None',                            
                          sparse_features_conflict_fraction=0,                  
                          leaf_estimation_backtracking='AnyImprovement',        
                          best_model_min_trees=1,                               
                          model_shrink_rate=0,                                  
                          min_data_in_leaf=1,                                   
                          loss_function='Logloss',                              
                          learning_rate=0.02582800015807152,                    
                          score_function='Cosine',                              
                          task_type='CPU',                                      
                          leaf_estimation_iterations=10,                        
                          bootstrap_type='MVS',                                 
                          max_leaves=64)                                        
                                                                                

────────────────────────────────────────────────────────────────────────────────
The following code is the process of optimizing the model to a variety of       
algorithms. I compared them all because they had different access methods and   
different results. I've locked it up as a chairman now.                         

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # catboost = tune_model(create_model('catboost'), choose_better = True, n_iter 
 20)                                                                            
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # catboost2 = tune_model(create_model('catboost'), optimize='Accuracy',        
 #                        search_library='scikit-optimize',                     
 search_algorithm='bayesian',                                                   
 #                        choose_better = True, n_iter = 20)                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # catboost3 = tune_model(create_model('catboost'), optimize='Accuracy',        
 #                        search_library='tune-sklearn',                        
 search_algorithm='bayesian',                                                   
 #                        choose_better = True, n_iter = 20)                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # catboost4 = tune_model(create_model('catboost'), optimize='Accuracy',        
 #                        search_library='tune-sklearn',                        
 search_algorithm='hyperopt',                                                   
 #                        choose_better = True, n_iter = 20)                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # catboost5 = tune_model(create_model('catboost'), optimize='Accuracy',        
 #                        search_library='tune-sklearn',                        
 search_algorithm='optuna',                                                     
 #                        choose_better = True, n_iter = 20)                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # catboost6 = tune_model(create_model('catboost'), optimize='Accuracy',        
 #                        search_library='optuna', search_algorithm='tpe',      
 #                        choose_better = True, n_iter = 20)                    
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                 📌 Ensemble                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

I've blended the top four algorithms. Performance has not improved, but we can  
try this. In some cases, this is often a better performance.                    

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # tuned_top4 = [tune_model(i) for i in top4]                                   
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 # blender_top4 = blend_models(estimator_list=tuned_top4)                       
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 df_pred = predict_model(catboost_best, X_test)                                 
 y_pred = df_pred.loc[:, ['Label']]                                             
                                                                                

────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                            📌 Interpreting Model                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Pycaret provides SHAP. This explains why the model derived the results in this  
way. The following figure visualizes the shape value of each variable based on  
the size of the value.                                                          

────────────────────────────────────────────────────────────────────────────────
                                                                                
 interpret_model(catboost_best)                                                 
                                                                                

────────────────────────────────────────────────────────────────────────────────
                                                                                
 submission['Transported'] = y_pred                                             
 submission.to_csv('submission.csv', index=False)                               
 submission                                                                     
                                                                                
