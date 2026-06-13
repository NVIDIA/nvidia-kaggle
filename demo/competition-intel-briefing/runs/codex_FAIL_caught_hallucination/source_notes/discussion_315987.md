╭─ Some rules to fill NaNs ────────────────────────────────────────────────────╮
│ Author: Vincent Debout | Votes: 99 | Comments: 0 | Created: 2022-03-30       │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/315987      │
╰──────────────────────────────────────────────────────────────────────────────╯

I denote "Children", persons with age <= 12 years old.                          

Here are some rules I identified to fill NaNs:                                  

 • link between HomePlanet and CabinDeck: Earth on E, F,G Mars on D, E, F Europa
   on A, B, C, D, E, T                                                          
 • persons in the same group all have the same CabinSide, possibly on different 
   CabinDeck                                                                    
 • in groups with several CabinDeck, people with no bill are necessarily on     
   CabinDeck: Earth on G Europa on B Mars on E or F                             
 • for Europa groups with several CabinDeck, people with no bill are on deck B  
 • CabinNum are in increasing order of PassengerIdGroup for each (CabinDeck,    
   CabinSide) => This is very useful but need a complex algorithm. All Cabins   
   are occupied !                                                               
 • People in a group have the same HomePlanet                                   
 • Surnames are found in a unique HomePlanet                                    
 • Children have no bill                                                        
 • CryoSleep have no bill                                                       
 • People not Children and not CryoSleep with no bill all have Destination      
   TRAPPIST-1e                                                                  
 • link between HomePlanet, VIP (also Age and CryoSleep): Earth no VIP Mars VIP 
   have Age >= 18 and no CryoSleep and never goes to "55 Cancri e" Europa VIP   
   have Age >= 25                                                               
 • Europa with Age < 25 never goes to "PSO J318.5-22"                           
 • Mars children are all transported                                            

Approximative rules (with some exceptions):                                     

 • Firstnames are found in a unique HomePlanet                                  
 • ToTal bill has minimum per HomePlanet and CabinDeck (Earth is around 300,    
   Mars 400 and Europa 1000), be careful when filling with median               

Please let me know if you identify additionnal rules                            

── 60 Comments ──

 2022-03-31

 2022-04-01

Kim Jongheon ▲6 2022-04-02
Children(<=12) are not VIP                                                      

DigitalJester ▲4 2022-04-03
Spending (FoodCourt, VR etc) is extremely right tailed, with mean and median    
returning extremely different results                                           

Spending varies considerably by                                                 

 • Deck                                                                         
 • VIP Status                                                                   
 • Homeplanet                                                                   
 • Destination                                                                  
 • Age                                                                          
 • Cryosleep                                                                    

The correlation across (most) spending categories is (probably) lower than you  
might expect.                                                                   

 2022-04-03

 ▲1 2022-04-03

Samuel Cortinhas ▲1 2022-04-04
Passengers with cabin number>=1509 are in deck F.                               

Samuel Cortinhas ▲3 2022-04-05
Passengers on CabinDeck T are not in CryoSleep.                                 

Samuel Cortinhas ▲3 2022-04-05
Families (same surname) tend to be on the same cabin side.                      

 2022-04-08

 2022-04-09

 ▲2 2022-04-12

 2022-04-18

Outatime ▲3 2022-04-22
Excellent observations. Properly filling NaNs before creating models is very    
important for this exercise.                                                    

Mikhail Okrochkov ▲3 2022-04-24
Everyone from Europa who was in CryoSleep over age 35 is transported.           

Benedict Zhang 2022-04-29
Generally, people in the same group have the same CabinDeck and the same        
CabinSide. In other words, Using PassengId might be sufficient to fill in all   
nulls about CabinDeck and CabinSide info. I think that'd be a great help to save
yourself from too many details to find about cabins.                            

Joseph El Gemayel ▲1 2022-05-04
Great findings!! I've tried to implement your ideas, appreciating your feedback 
on the notebook:                                                                
https://www.kaggle.com/code/josephelgemayel/spaceship-titanic-fillna            

ZhangLongIsMe 2022-05-04
Hi folks, I think I found the complicated algorithmn to fill Cabin NaNs, check  
out my notebook with helper1.                                                   

Sivadharan K 2022-07-01
Excellent analysis. Thank you for sharing.                                      

Sohail Ahmed 2022-07-01
very interesting observations. 👍                                               

Larsik ▲5 2022-07-18
These are very useful!                                                          

Another one that I have used is that almost all VIPs (≈95%) travel alone. Also, 
my accuracy improved a lot when I made some dummy variables w.r.t. spending.    
People who spend a lot in Spa, VRDeck and RoomService are much less likely too  
be transported, wheras people who spend in the other two categories are more    
likely to be transported.                                                       

BTW, is it considering cheating to hardcode certain relationships? Like the ones
mentioned here that Mars children and people from Europa in CryoSleep > 35 are  
always transported. Around 250 people in the test set fit this criteria so      
automatically setting these to True would probably improve accuracy a fair      
amount?                                                                         

Teresa Wu 2022-07-18
Thanks for your findings and sharings! it's very useful~                        

 2022-07-28

 2022-08-23

Amihua 2022-10-13
These are very useful! Could you please teach me how you can find it?😁         

 2022-11-15

Tanveer Singh 2022-11-26
That's a really great finding! Can you please tell us how you found these       
patterns?                                                                       

Nikhil Correia 2022-11-27
Excellent Observations, will try to incorporate these changes!                  

 2022-12-08

Eamonn Tweedy ▲2 2022-12-14
🌆 
inbox%2F12006962%2F1dcd69aa2b833315f641e249a501fe27%2Fspending_cat_log_plots.png
?generation=1671027607797474&alt=media The bill amounts in all categories are in
will help.  If we restrict to those with nonzero bills in the respective        
category and plot the bill amount on a log axis, the distributions of           
log(spending) are strikingly similar across categories and across training and  
test sets - see attached figure.                                                

Eamonn Tweedy ▲1 2022-12-14
Quick follow-up: if we bin the passengers by age, then each bin individually    
appears to have the above type of density curve by log(spending) in each        
category.  However the scale of the y-axis (density) varies by age bin and by   
spending category.                                                              

Eamonn Tweedy ▲2 2022-12-14
It may also be helpful to notice that within each spending category:            
approximately 34% of passengers don't spend, approximately 64% of passengers    
spend, and approximately 2% of passengers have missing spending.  I suspect it's
reasonable to fill missing spending this way:                                   

 1 Set all CryoSleep = True and Age < 13 passengers to spending = 0 in all      
   categories.                                                                  
 2 Fill other missing spending (for each category, about 1% of total passengers)
   any way you want, e.g. set to 0.                                             

Note: It appears that in each feature, approximately 2% of training set         
passengers have NaN and approximately 2% of test set passengers have NaN.       

Roland Eriksson 2022-12-18
Thank you all who contributed to this topic, very useful! Now I am inspired to  
try out some more sophisticated ways of filling the NaNs, only done simple      
imputation so far.                                                              

Ankit Modak 2022-12-19
Great information guys! I have filled NaN values just by using statistical      
techniques. Thankyou for the motivation of finding more logical and correct ways
to fill NaN values.😀                                                           

Eamonn Tweedy 2022-12-19
Thanks @vdebout for initiating this thread and thanks to others for             
contributions.  The ideas here have been very helpful to my learning! I have    
created a notebook which implements some of these policies and builds           
predictions using some GBDT models:                                             
https://www.kaggle.com/code/eamonntweedy/spaceship-titanic-tree-models          

Mahabub Alam 2022-12-26
Thank you for sharing the great information with us.                            

 2023-01-03

y333ty333t ▲4 2023-02-18
Did anyone test model performance with and without filling the NaNs using these 
rules?                                                                          

 2023-03-06

 ▲1 2023-03-08

Niranjan Akella 2023-03-17
According to a short analysis conducted by me and my partner @neeliyeseswisree  
we found that the 'Transported' rate is as follows:                             

{'B': 365, 'C': 269, 'G': 83, 'A': -2, 'T': -3, 'D': -64, 'E': -250, 'F': -336} 

i.e.; Groups from Cabin B are said to have higher frequency of getting          
transported than any other cabins and that of Cabin F have the least chances of 
not getting transported.                                                        

 2023-04-01

TomBombadil95 2023-04-05
Very useful!                                                                    

TensorFlower ▲2 2023-06-21
I wonder if filling them using those rules can increase the model's accuracy to 
greater than 0.82.                                                              

 2023-07-17

HJ 2023-08-09

▌ Mars children and people from Europa in CryoSleep > 35 are always           
▌ transported. Around 250 people in the test set fit this criteria so         
▌ automatically setting these to True would probably improve accuracy a fair  
▌ amount?                                                                     

Not really. It seems these are easily learned by a simple model (lightgbm).     

 2023-08-14

Shane Simon 2023-09-29
These insights are so awesome to have for imputing. As a newbie, I'm just       
curious... How do you identify these relationships? Do you graph out each       
variable?                                                                       

Shane Simon 2023-09-29
I'm also dying to know the answer, but in reality, I think they are just good at
doing EDA and can find trends easily                                            

Anand Joshua 2023-11-03
Thanks, I was wondering about best ways to fill NaNs                            

 2024-01-23

Jacob Sultan 2024-05-21
Thanks so much for this info! I used a lot of these rules to find the algorithms
to fill in the Cabin Nans without guesses!                                      

Have posted a link if you want to take a look :)                                

https://www.kaggle.com/code/jacobsultan/how-to-impute-nearly-every-cabin-correct
ly                                                                              

Mohammed Hossam 2024-07-24
could we use a model to predict the missing value, we take null as what we want 
to predict and others as train data and test data                               

We1chnon 2024-07-27
People who destination is TRAPPIST-1e and cabin deck is T don't use CryoSleep.  

 2024-08-06

James Antony Das 2024-08-11
That's an interesting observation!                                              

Sheikh Muhammad Abdullah 2024-08-31
Yes i try but No Improvement                                                    

 2024-09-23

GreyMatterGuru 2024-10-16
These are really helpful!! Thanks for sharing.                                  

Brayden Thomas 2025-05-01
How much difference does it make to final score to include these rules? Anyone  
tried a model before and after to see results?                                  

