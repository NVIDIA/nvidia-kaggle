╭─ Tip - Don't forget about the Cabin! 📈 ─────────────────────────────────────╮
│ Author: Alex Teboul | Votes: 38 | Comments: 0 | Created: 2022-02-24          │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/309693      │
╰──────────────────────────────────────────────────────────────────────────────╯

For a slight increase in accuracy, don't drop the Cabin column entirely from    
your modeling.                                                                  

For example:                                                                    

                                                                                
 # split column and add new columns to df                                       
 train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', expand=True)    
 # display the dataframe                                                        
 train.head()                                                                   
                                                                                

🌆 Screen-Shot-2022-02-24-at-4-07-00-PM.png                                     


                                 Notebook Link                                  

── 19 Comments ──

Robin S. 2022-02-25
I will work on this feature next. Nice code :D                                  

Raphael Filip 2022-02-26
I'm trying to split the Cabin using R, but when I tried this:                   

strsplit(dados$Cabin, split = "/")                                              

My kaggle notebook was completely freezing. Does someone has any tip?           

Bill Cruise ▲5 2022-02-26
Similarly, PassengerId can be split into Group and Number for another slight    
gain.                                                                           

 ▲4 2022-02-26

Raphael Filip 2022-02-27
Thanks!!!!                                                                      

Wongi Park 2022-02-28
Great !!!👍 It's so useful!                                                     

RandomRushGirl 2022-02-28
Yup, thinkin g of it logically (not from a code perspective) this makes a lot of
sense so should see some correlation here. Thanks :)                            

Shivam Negi 2022-03-15
Good find !  seems like there are some nice hidden features                     

DigitalJester 2022-04-03
Much more elegant than what I did!                                              

                                                                                
 data <- data %>%                                                               
   mutate(Cab_len = nchar(Cabin)) %>%                                           
   mutate(Deck.f = as.factor(str_sub(Cabin, 1, 1)),                             
          CabinNum = as.integer(str_sub(Cabin, 3, (Cab_len - 2))),              
          PorS.f = as.factor(str_sub(Cabin, Cab_len))) %>%                      
   select(-"Cab_len")                                                           
                                                                                

AlexTheDataScientist 2022-04-11
Is the increase in performance because this feature holds information or is it  
coincidental ? does this even matter if there is gain to be made ?              

DigitalJester ▲4 2022-04-12
The size of group, ie the second part of PassengerId, is definitely helpful.    
This information is a little like the sibsp and parch variables in the original 
Titanic challenger.                                                             

Zub3r 2022-07-16
Yes !!! , you can get size of group with the help of that  and Weather that     
passenger was travelling with Family or not                                     

Luke DUTHOIT ▲1 2022-07-20
I'm glad to discover that other people had the same idea, it means my intuition 
wasn't that bad ;) For the record, the deck position (after being label encoded 
: A-> 1, B-> 2, etc) and cabin number proved to be quite useful, but side (after
being one hot encoded, and split into two features 'S' and 'P') not that much.  
See below : after hyper-parameters optimization/cross-validation on train set   
with an AdaBoostClassifier, I displayed features importance. The horizontal line
is  equal to the inverse of the number of features, that is to say the          
theoretical importance if all features had the same importance for this         
algorithm. We see that features from Deck and Cabin are above, whereas features 
from Side are below...                                                          

Noman Bukhari 2022-09-13
I tried it with a SVM but my accuracy actually went down from about 78% to 76%, 
but I'm sure it's because I didn't do it very well.                             

Zub3r 2022-09-15
Can you please specify kernel trick used in training above SVM                  

Noman Bukhari 2022-09-18
It was rbf, default sklearn value.                                              

Peter Danilov 2023-05-27
Great idea! I had a feeling that with keeping it consolidated (not to mention   
dropping), we'd lose valuable information.                                      

Omikumar Bhavinkumar Makadia 2023-08-14
Thank you for the idea !!                                                       

Jacob Sultan 2024-07-08
I've found an algorithm to impute most of the cabins without guessing! Please   
let me know what you think :)                                                   

https://www.kaggle.com/code/jacobsultan/how-to-impute-nearly-every-cabin-correct
ly/notebook                                                                     

