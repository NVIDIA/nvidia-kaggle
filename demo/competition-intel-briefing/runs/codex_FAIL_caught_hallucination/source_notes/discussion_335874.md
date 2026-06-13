╭─ Interesting ways to fill NULL Values - Improving accuracy by ~2% ───────────╮
│ Author: Praveen Kumar | Votes: 18 | Comments: 0 | Created: 2022-07-08        │
│ https://www.kaggle.com/competitions/spaceship-titanic/discussion/335874      │
╰──────────────────────────────────────────────────────────────────────────────╯

During my data analysis phase, I discovered the following, more practical ways  
to fulfill Null values, at least to some specific features.                     

HomePlanet: Instead of directly it to be earth, you and feature engineer null   
values based on the following 2 scenarios. i) People who are travelling as a    
group, are more likely to come from the same home planet. Thus, if you find     
missing values within the same group, it's better to fill them based on group   
values. ii) People who are not traveling in a group, are not 100% to come from  
Earth. So, instead, you can look at the probability distribution of different   
home planets, and fill the remaining null values based on the normal            
distribution. This will additionally maintain consistency in data rather than   
creating bias. CryoSleep, Destination & VIP can be filled in a similar manner   

Additionally, people have combined 'RoomService, FoodCourt, shopping mall, Spa, 
VRDeck' into a single column by adding each value. However, 'RoomService &      
FoodCourt' are essentials for a living but not "shopping mall, Spa & VRDeck".   
So, these features can be split into 2 features. People who spend more on living
essentials seem to be richer, thus having higher status among all the people    
onboard.                                                                        

Bringing practical intuition into machine learning could give you better        
results. Hope this helps.                                                       

── 9 Comments ──

 2022-07-08

Viktor Taran ▲1 2022-07-12
Great, why not =)                                                               

Sarang Ali ▲1 2022-07-22
Can you show me how are you filling Homeplanet using groups, I m trying for many
hours but couldn't find any way.                                                

EfeGamii ▲1 2022-07-26
yeah this idea helped me a lot thanks man ^^                                    

Praveen Kumar 2022-07-26
I have written this function which works for all columns. It can be modified to 
make it more efficient, but it also serves the purpose and is easy to           
understand.                                                                     

🌆 Fill By Group                                                                

Omikumar Bhavinkumar Makadia 2023-08-14
Great Help thanks buddy !                                                       

AIDE-imec 2024-07-30
Clever strategy, I especially liked the solo-traveller filling method.          

Alexandre Cogordan 2024-10-26
I understand this is a very late reply, and I might be mistaken since I am going
through this very quickly, but are you sure that your second proposition holds  
true? The screenshot below seems to indicate otherwise:                         

🌆 
inbox%2F6734944%2F7c8be7feb1fde25b7bf60e4a5ad16dee%2FScreenshot%202024-10-26%20a
t%2016.50.45.png?generation=1729954255295616&alt=media                          

However, your other ideas are very constructive, thanks for the insight!        

PS: I am assuming you're referring to the fact that solo-travelling, hence      
having one unique group, would infer that the HomePlanet is not Earth. Is that  
right?                                                                          

Shivansh Chhawri 2024-10-27
Nice insight!                                                                   

