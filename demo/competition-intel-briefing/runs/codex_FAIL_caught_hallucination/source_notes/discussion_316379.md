╭─ Cabin number chunking ──────────────────────────────────────────────────────╮
│ Author: Samuel Cortinhas | Votes: 13 | Comments: 0 | Created: 2022-04-01     │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/316379      │
╰──────────────────────────────────────────────────────────────────────────────╯

I've been analyzing the cabin feature today and I came across this insight. It  
seems that the cabin number is chunked into groups of 300 cabins, and within    
each group the target distribution is a little bit different.                   

🌆 Cabin-number-distribution.png                                                

This means we can compress this information by creating a new feature that      
tracks which chunk the passenger is in. I implemented this in my notebook, you  
can see here for details.                                                       
https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide  

── 5 Comments ──

DigitalJester ▲3 2022-04-02
Not all decks are equal! For example Deck T's highest cabin number is 4 while   
deck F has cabin numbers all the way upto 1894!                                 

samohao 2022-04-20
What a nice comment!                                                            

kikurage 2022-06-16
Good analysis!!                                                                 

Josh Miller 2022-07-11
Noticed the same thing today. I like your approach for turning this into a      
feature!                                                                        

Omikumar Bhavinkumar Makadia 2023-08-15
Nice Analysis !!                                                                

