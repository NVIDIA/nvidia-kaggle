╭─ How to Choose OneHotEncoding between LabelEncoding? I'll Answer!! (EDA Tip)─╮
│ Author: Wongi Park | Votes: 44 | Comments: 0 | Created: 2022-02-25           │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/309803      │
╰──────────────────────────────────────────────────────────────────────────────╯

We apply One-Hot Encoding when:                                                 

 1 The Categorical feature is not ordinal                                       
 2 The number of categorical features is less so one-hot encoding can be        
   effectively applied                                                          

We apply Label Encoding when:                                                   

 1 The categorical feature is ordinal (like Primary school, high school)        
 2 The number of categories is quite large as one-hot encoding can lead to high 
   memory consumption                                                           


One-Hot-Encoding Feature                                                        

 • HomePlanet                                                                   
 • CryoSleep                                                                    
 • Cabin (Side)                                                                 
 • Destination                                                                  
 • VIP                                                                          

Label Encoding Feature                                                          

 • Cabin(Deck)                                                                  
 • Name    (cause.2)   (many nuinque values)                                    


Total Bill = 'RoomService' +  'FoodCourt' + 'ShoppingMall '+ 'Spa' + 'VRDeck'   

 • We need to deal with NaN values Before adding them. (NaN values replace 0    
   beacuse,  It's most common in data.)                                         

If you helpful, Don't forget vote! Model tip :                                  
https://www.kaggle.com/c/spaceship-titanic/discussion/309535                    

Code : https://www.kaggle.com/kalelpark/spaceship-titanic-solutions-for-beginner
(Code will be Update!)                                                          

── 19 Comments ──

LucifierX ▲1 2022-02-26
Nice Explanation                                                                

Wongi Park 2022-02-26
👍👍 Great Thanks~~ Keep Going                                                  

Subham Surana ▲1 2022-02-26
Well explained @kalelpark                                                       

 2022-02-26

Wongi Park 2022-02-26
Thanks!!!👍👍👍                                                                 

CaruzzoC ▲1 2022-02-27
Hi,                                                                             

I agree with the point "When to apply One-Hot Encoding/Label Encoding". But I am
wondering what made you choose the use Label Encoding for the Cabin feature? Is 
there a clear hierarchy regarding the deck? If yes, do you have an EDA or a     
hierarchy to propose?                                                           

Thank you 😃                                                                    

Wongi Park ▲1 2022-02-27
When I Using Pycaret, model seems to think that the teck is a more important    
feature than the destination. Therefore I use labelencoding. (Check my Notebook 
and see interpret_model. I write them.)👍                                       

CaruzzoC 2022-02-27
Great! I will test it out and give some feedback on how it influenced my        
results!                                                                        

Mostafa Alaa ▲1 2022-02-27
(Note and thank) Hello Wongi Park for these tips really makes me learn something
new to me, I've a small note about this phrase "We need to deal with NaN values 
Before adding them. (NaN values replace 0 because, It's most common in data.)", 
at this case we would make it zero because this kind of missing data means he   
may don't buy anything from this category, not like other data we choose most   
frequent, thanks again, Wongi :) @kalelpark                                     

Wongi Park ▲1 2022-02-27
It's an honor for me to be of help!!. Thanks~~ Keep gowing👍👍                  

C4rl05/V ▲1 2022-02-28
Hello @kalelpark, thanks for sharing, super easy to understand and apply        

Wongi Park 2022-02-28
Thanks for reading!👍 I hope your model improve!                                

C4rl05/V ▲1 2022-02-28
Thanks, @kalelpark, yes I'm currently implementing the code.                    

Tushar Khoche ▲1 2022-03-14
Very simple explanation, thanks @WongiPark                                      

Orkun Aran ▲1 2022-04-23
Hi Wongi, thanks for the information. Quick question:                           

 • If we create total_bill column it will be correlated with roomservice ,      
   foodcourt, shoppingmall, spa and Vrdeck. I know this is a problem in         
   regression, but not so sure about classification? It is ok to keep total_bill
   here?                                                                        

Luke DUTHOIT 2022-07-20
Thanks @kalelpark for this suggestion of summing all expenses into one variable 
! It helped me a lot !                                                          

Moreover, I used Phi-K correlation coefficients between all potential features  
and Transported, and show that 'Total Bill' is not the most correlated, but its 
coefficient is still higher than the sum of coefficients of 'FoodCourt' +       
''ShoppingMall ". Those last two features can thus being check out of the list  
of features used to build our X matrix.                                         

abdelbasset Basto 2023-04-18
thank you                                                                       

Omikumar Bhavinkumar Makadia 2023-08-14
Thanks for helping !                                                            

 2025-05-07

