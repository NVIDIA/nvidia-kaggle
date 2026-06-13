╭─ Feature engineering results ────────────────────────────────────────────────╮
│ Author: Samuel Cortinhas | Votes: 55 | Comments: 0 | Created: 2022-04-03     │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/316768      │
╰──────────────────────────────────────────────────────────────────────────────╯

Here is a summary of the feature engineering I have tried so far and the results
I got on the test set. The baseline I used can be found in my notebook here. All
I did was add/drop 1 feature at a time and see how the final score changed.     

 • drop VIP feature - 0.02% increase                                            
 • create an expenditure feature that sums the total money spent across the 5   
   amenities - 0.5% increase                                                    
 • create expenditure feature and drop the 5 amenities (RoomService, ...,VRDeck)
   - 4.7% decrease                                                              
 • create a cabin deck feature (A,B,...,T) - 0.35% increase                     
 • create a cabin side feature (P/S) - 0.9% increase                            
 • create a group size feature from the PassengerId column - 0.1% increase      
 • create a no spending binary feature that is 1 if no money was spent and 0    
   otherwise - 0.2% increase                                                    
 • Create a solo feature that tracks if passenger was travelling on their own   
   (from PassengerId) - 0.2% increase                                           
 • Create cabin region features according to chunks of 300 passengers (see here 
   for details) - 0.2% increase                                                 
 • create a family size feature from last name - 0.0% change                    
 • create binary age features for 0-18 yo, 18-25 yo, 25+ yo - 0.02% increase    
 • same as above but drop the age column - 0.4% drop                            

Conclusions:                                                                    

 • create total expenditure, cabin deck and cabin side features - these are very
   helpful                                                                      
 • group size, solo, no_spending and cabin regions also help a bit              

Do you agree? What feature engineering have you tried and did it improve your   
score?                                                                          

── 15 Comments ──

Rohan Mudaliar ▲1 2022-04-19
pretty helpful                                                                  

Satoshi_S 2022-04-21
Great post. I didn't come up with cabin region features on my own. I will give  
it a try!!                                                                      

Chirag Desai 2022-04-28
Excellent analysis.. I also played with features but didn't measure the impact. 
This is good way of doing feature analysis. Taking one feature at a time and    
checking its impact on the score and then updating again. Did you do same with  
NaN's? Please share if you have imputed values and checked the score, it will be
very helpful also. Thanks.                                                      

Fakhri Robi Aulia ▲7 2022-06-01
according to my analysis. The Spending could be divided into two parts. The     
First one Service spending such as 'RoomService','Spa','VRDeck' and spending on 
shopping 'FoodCourt','ShoppingMall'. When I combined into all spending features 
the correlation was -0.2. When separate into service_spending feature and       
shopping_spending the correlation became -0.35 and 0.04 respectively.           

Muhammad Irfan Azam 2022-06-02
Dropping VIP feature reduced my score to 4.6%                                   

Noman Bukhari 2022-09-14
How do you know if your feature engineering might just be overfitting a model   
rather than actually predicting better?                                         

tka400 ▲1 2022-09-20
Data is devided by two parts - train and validation. If validation score much   
less then the train score - it is overfitting                                   

Jayant Yadav 2022-12-29
quite insightful!                                                               

Roland Eriksson 2022-12-30
From my analysis CryoSleep is the most important feature. The spending features 
seem important but confusing, I hesitate if I should create one single          
"billing/spending" feature or split into expenditure groups. I have considered  
creating a "solo" feature, but from your results I doubt that will have any     
great effect. Cabin side did not look useful to me, I may have to reconsider    
that.                                                                           

abdelbasset Basto 2023-04-18
perfect thanks alot                                                             

Ishan 2023-08-12
Thank you so much for this. I implement all this in my notebook. It's really    
useful.                                                                         

Bhunakit Chantaraseno 2023-09-21
Thanks. It's very useful.                                                       

Andre Moreira 2023-10-16
Thank you for the analysis. I tried the idea of creating a total expenditure    
column and dropping the other ammenities, I can confirm that the model's        
prediction gets worse.                                                          

Based on a correlation analysis between the target variable ("Y") and the other 
variables ("X"), I find interesting that with very few parameters ("CryoSleep", 
"RoomService", "Spa", "VRDeck") I get almost as good predicitons (Score: 0.7746)
as with about all parameters included (0.78746).                                

Jacob Sultan 2024-07-08
I've implemented an algorithm to fill most of the cabins correctly to eliminate 
this need mostly for cabin chunking ! Please let me know what you think :)      

https://www.kaggle.com/code/jacobsultan/how-to-impute-nearly-every-cabin-correct
ly/notebook                                                                     

Ayilhan 2025-06-10
quite interesting                                                               

