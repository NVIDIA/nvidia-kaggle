╭─ besides regression, also dwt (time warping)!  ──────────────────────────────╮
│ Author: hengck23 | Votes: 34 | Comments: 36 | Created: 2026-05-06            │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/697431                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

related to geosteering. Here, given a strip of MD-GR of the horizontal well,    
"stretch and fold" it so that it matches the TVT-GR reference of the typewell   

🌆 
inbox%2F113660%2Fb40517687f05f7ecea819763d3b150dc%2FSelection_3412.png?generatio
n=1778261192499811&alt=media                                                    

🌆 
inbox%2F113660%2F6f36c16fad39781039af137e753996dd%2FSelection_3304.png?generatio
n=1778038684983121&alt=media                                                    

────────────────────────────────────────────────────────────────────────────────
but, since the test and train locations are pretty close, pure regression might 
just also work                                                                  

── 36 Comments ──

hengck23 ▲2 2026-05-06
🌆 
inbox%2F113660%2F8653a6175531f307e6d3f22c4b95b560%2FSelection_3305.png?generatio
n=1778039471463012&alt=media https://github.com/luthfigeo/DTW-Stratigraphic-Corr

hengck23 ▲3 2026-05-06
🌆 
inbox%2F113660%2F7418e3be8c2b50fe18e87e7f8beb00e4%2FSelection_3321.png?generatio
n=1778064703636922&alt=media                                                    

hengck23 ▲3 2026-05-06
the trick to winning is to "somehow" reconstruct the "3d geological site" using 
the train AND test data, since the wells are in the same "site"                 

hengck23 ▲2 2026-05-06
forward model?                                                                  

                                                                                
 hfile = "0a57a29c__horizontal_well.csv"                                        
 tfile = "0a57a29c__typewell.csv"                                               
 h  = pd.read_csv(f"{KAGGLE_DIR}/train/{hfile}")                                
 tw = pd.read_csv(f"{KAGGLE_DIR}/train/{tfile}")                                
                                                                                
 tw_tvt = tw["TVT"].values                                                      
 tw_gr = tw["GR"].values                                                        
                                                                                
 h_tvt = h["TVT"].values                                                        
 h_gr = h["GR"].values                                                          
 query_gr = np.interp(                                                          
     h_tvt,                                                                     
     tw_tvt,                                                                    
     tw_gr,                                                                     
 )                                                                              
                                                                                

🌆 
inbox%2F113660%2F4690deb3957ae578318354dcd4385283%2FSelection_3332.png?generatio
n=1778085488837342&alt=media                                                    

hengck23 ▲1 2026-05-07
piece-wise fitting DTW.                                                         

model predict (start,end, dTVT/dMD slope) for each segment.                     

but i think the original DP in cost matrix is better                            

🌆 
inbox%2F113660%2F23521faa598aa443f46d761cc1f5f7c0%2FSelection_3349.png?generatio
n=1778129487699553&alt=media                                                    

🌆 
inbox%2F113660%2Fd0bdc08a78a3bb3360082ab1e9276493%2FSelection_3362.png?generatio
n=1778140477307972&alt=media                                                    

 • When the drill moves a long distance in MD / XYZ, it may still stay inside   
   almost the same geological position.                                         
 • That makes alignment hard because there is little “geological movement” to   
   match                                                                        

eugene 2026-05-07
Do I understand you method correct? You move a window along the hw_gr, after ps 
point. For each window, you find the closest window on tw_gr with DTW. Then you 
look at which TVT is closest to the window on tw_gr -> this TVT  is predict ? I 
did that way, but it totaly not worked for me 🤔                                

hengck23 ▲3 2026-05-07
It is not the normal dtw. The index can be reversed depending if the drillhead  
is travelling upwards or downward. The noise is quite large, maybe you need to  
restrict to local search.                                                       

The host post had a YouTube video link on rogii geosteer, that will clear things
up                                                                              

hengck23 2026-05-07
@evgeny000                                                                      

🌆 
inbox%2F113660%2F09eb08a581fad7ca45347e14199bab63%2FSelection_3366.png?generatio
n=1778157143610840&alt=media                                                    

eugene 2026-05-07
Thanks for the explanation! I still don't fully understand the data yet 😬, I   
hope it will be more clear after watching the video you mentioned. As far as I  
understand, you don't use tw data?                                              

The most confusing part to me  the tw data. The TVT in the tw data is a         
monotonous straight line, although as far as I understand, when different layers
intersect during drilling, the thickness of the different layers changes, and   
the TVT should be fluctuate 🤔                                                  

hengck23 ▲3 2026-05-07
You should think of it like that: reference vertical typewell has gr that       
encodes the geologic location called tvt. We are in horizontal well with unknown
location. We want to know what is our tvt. But we have signal gr. We need to    
move up or down to have generate enough gr signal signature for matching        
reference gr to guess the tvt.  Tvt is location relative to the geology layer we
want to be.                                                                     

hengck23 ▲2 2026-05-08
plot in 3d and it is a folding problem                                          

🌆 
inbox%2F113660%2Fa5cbb405fd3a89da3dd64ea731d85478%2FSelection_3390.png?generatio
n=1778215560856015&alt=media                                                    

hengck23 ▲1 2026-05-08
🌆 
inbox%2F113660%2Fc8c6eeef5151aab214e5273363907b35%2FSelection_3391.png?generatio
n=1778215665544705&alt=media                                                    

hengck23 ▲1 2026-05-09
plot of md vs dTVT                                                              

🌆 
inbox%2F113660%2F463aaa87facc21c28d172fec7a016510%2FSelection_3418.png?generatio
n=1778305370654718&alt=media                                                    

🌆 
inbox%2F113660%2Fd1dace62c505b33b21d01104722c3bf8%2FSelection_3419.png?generatio
n=1778305406732462&alt=media                                                    

this tells you how the ground truth is created ... reverse engineering?         

hengck23 ▲2 2026-05-09
why is dy constant? synthetic data????                                          

🌆 
inbox%2F113660%2F1bc50ab50551bbe3621be961745da6de%2FSelection_3427.png?generatio
n=1778312927936006&alt=media                                                    

🌆 
inbox%2F113660%2Ffc1b73c99aa3ca582d498619a119804f%2FSelection_3428.png?generatio
n=1778312940805533&alt=media                                                    

hengck23 ▲2 2026-05-09
deep net logit:  horizontal md length x location of reference (each location is 
a class)                                                                        

🌆 
inbox%2F113660%2F3df800b866b035d8ea3668f479cd241b%2FSelection_3456.png?generatio
n=1778350565993562&alt=media                                                    

training iterations of the transformer:                                         

 • it figures out the best way is to grow from PS?  I actually expect it to find
   anchor points first                                                          

Navneet 2026-05-10
Cool info on geosteering @hengck23                                              

hengck23 ▲2 2026-05-16
related:                                                                        

🌆 
inbox%2F113660%2F183302a3299a70acb01e682f43f45de6%2FSelection_3562.png?generatio
n=1778895793013734&alt=media related: https://github.com/hhschumann/LWD_inversio
"This project aims to use gamma ray loging while drilling (LWD) measurements to 
invert for the position of a geologic interval relative to the wellbore"        

🌆 
inbox%2F113660%2Fd9c9158c49c5a610b3bfce0a6def03c0%2FSelection_3563.png?generatio
n=1778895891144447&alt=media                                                    

hengck23 ▲3 2026-05-16
🌆 
inbox%2F113660%2F3357765646b064976681c3c8410ff588%2FSelection_3564.png?generatio
n=1778896502903825&alt=media                                                    

Real-time forward modeling and inversion of logging-while-drilling              
electromagnetic measurements in horizontal wells                                

hengck23 ▲1 2026-05-16
https://www.rogii.com/blog/starsteer-geoassist-enhanced-eagle-ford-reservoir    
ROGII implemented StarSteer's ML-based GeoAssist to automate geosteering.       

which parts are the most and least confident? what are ML looking at?           

🌆 
inbox%2F113660%2F0517074d09fcb4213950a5daaa3b75f6%2FSelection_3566.png?generatio
n=1778897341430331&alt=media                                                    

hengck23 ▲1 2026-05-16
where are the typewells? how is the global dip related to the xy slant          
horizontal drill path?                                                          

🌆 
inbox%2F113660%2Fba921864d6ffa56e24368627be5574d1%2FSelection_3568.png?generatio
n=1778898557640166&alt=media                                                    

hengck23 ▲1 2026-05-16
their heatmap looks very good (heatmap is some similarity between horizontal GR 
and reference geology?)                                                         

🌆 
inbox%2F113660%2F834455cdaf83bc4e0ba26e6670934287%2FSelection_3567.png?generatio
n=1778898507823700&alt=media                                                    

https://www.rogii.com/blog/the-hidden-cost-of-switching-between-geoscience-tools
how wells are planned                                                           

hengck23 ▲1 2026-05-16
🌆 
inbox%2F113660%2Fef1fce03482eca082114a0817f89a6fd%2FSelection_3613.png?generatio
n=1778956371898934&alt=media                                                    

neighbour can help! e,g, they tell you the range of horizontal tvt              

PatrickAIForFun ▲1 2026-05-16
You are showing/comparing the typewell logs, correct? If yes, then this is      
expected an has already been found (although not shifted matches, but exact     
matches):                                                                       
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion
/698449                                                                         

Tom ▲1 2026-05-17
Any Physics-informed ML approaches discovered? @hengck23                        

hengck23 ▲5 2026-05-17
issue is not physics modeling. you can verify your results with forward         
differentiable model:                                                           

                                                                                
 error = observed GR - generated GR = typwell GR (torch inteploated tvt as look 
 index)                                                                         
                                                                                

issue is inverse problem. many possible solutions  and we have to learn the data
"preferred solution" and not the best solution.                                 

it is like given 2 points A,B on map, predict how to get from A to B. there are 
some rules, but not strictly followed ...                                       

hengck23 ▲2 2026-05-17
Someone should probe the hidden typewell to see if they are offset copies of    
train. I think some are. If they are, you have free geology infotmation copied  
from train                                                                      

PatrickAIForFun ▲1 2026-05-17
I can neither confirm nor deny with full certainty, however based on my         
observations and testing it also seems very likely that the hidden test set     
shares some typewells with the public training set. However, apart from the     
geology labels of the typewell, this does not provide much more information.    
Furthermore ,in a real world application these topsets for the typewell would   
also most likely be available.  Thus, this does not compromise the general goal 
of the competition in my opinion.                                               

@igorakuvaev is this intended/known?                                            

hengck23 ▲1 2026-05-17
you can check the host competition PPT. he shows the hidden test well location  

hengck23 ▲1 2026-05-17
"apart from the geology labels of the typewell, this does not provide much more 
information" you are wrong. the model now become tvt = model(shared type well,  
known tvt, full tvt of neighbours (include site geology and gr) )               

Tom ▲2 2026-05-17
🌆 
inbox%2F4310004%2F53b4b42c9e20e0b927c4868cbc9ae1fd%2F532.png?generation=17790256
23576570&alt=media                                                              

With these information and referencing the typewells, you can even build a very 
strong transformer or GNN to encode them                                        

hengck23 ▲4 2026-05-17
suddenly, i think of folding protein solution: alphafold. here we are           
essentially folding the horizontal GR signal. each possible trajectory is a     
confomer. typewell and neigbhours GR are "binding sites". Let's called it       
alphaSteer                                                                      

hengck23 ▲6 2026-05-18
surprise surprise surprise there are only 69 unique typewells in train data     

hengck23 ▲1 2026-05-18
🌆 
inbox%2F113660%2F08d9018ad30f9230de44a25e038465e0%2FSelection_3670.png?generatio
n=1779134589932757&alt=media data augmentation                                  

hengck23 ▲1 2026-05-24
🌆 
inbox%2F113660%2F455e0d537cbbf0969d560dd5100e5a19%2FSelection_3752.png?generatio
n=1779630800094346&alt=media                                                    

PC Jimmmy 2026-05-26
hengck23                                                                        

When you checked the ppt did you end up with 45 test well paths?                

PC Jimmmy ▲1 2026-05-26
If you look for shifted matches I found only 57 type wells in the entire field. 

