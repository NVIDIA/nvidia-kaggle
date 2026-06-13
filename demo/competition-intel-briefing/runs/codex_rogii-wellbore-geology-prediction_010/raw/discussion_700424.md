╭─ Share an UI visualizer  ────────────────────────────────────────────────────╮
│ Author: Tom | Votes: 46 | Comments: 60 | Created: 2026-05-17                 │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/700424                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

🌆 
inbox%2F4310004%2F8fe8f43532b152aa5a7637b7d6f1a2fe%2FUI1.png?generation=17790220
20724846&alt=media                                                              

Feel free to use it: https://github.com/tom99763/rogii-viewer                   

And some EDA & directions in attachments. I suggest people to read glossary.html
first to clarify some confused definition.                                      

I really like this explaination about TVT:                                      

                                                                                
 Analogy: TVT is the “floor number” in a geological building                    
                                                                                
 Imagine the geology as a completed high-rise building, where each floor        
 represents a different rock layer (ANCC, Austin Chalk, Eagle Ford, Buda, etc.) 
                                                                                
 Typewell = the elevator shaft: it goes vertically from the top floor to the    
 basement, recording “the GR at this layer is X” as it passes through each floo 
 Horizontal well = a person walking inside the building:  moving along the      
 hallway of a certain floor, sometimes going up or down between floors.         
 TVT = which floor is this person currently on?                                 
                                                                                

                         Update GR Mismatch Visualizer                          

🌆 
inbox%2F4310004%2F5b91ffbfc5d669a76d10fe1cb409dcb5%2FGR.png?generation=177903061
1267623&alt=media                                                               

── 60 Comments ──

Tom ▲3 2026-05-17
This plot seems showing a hidden insight                                        

🌆 
inbox%2F4310004%2Fd0df1bc4db7dfa9a0ac8a8689037f967%2Finsights.png?generation=177
9025047785552&alt=media                                                         

hengck23 ▲3 2026-05-17
U can add following to cnn feature:                                             

 1 Self correlation, good for identifying moving reverse                        
 2 Neighbouring well correlation                                                

hengck23 ▲4 2026-05-17
Normal dtw assume monotonic seq and cannot match reverse index, so be careful if
you use it.                                                                     

 2026-05-17

Gaurav Rawat ▲2 2026-05-18
love the eda via claude here very nice to understand the comp ..                

Tom ▲3 2026-05-18
This comp is really complicated. There are many details haven't been coveraged  

hengck23 ▲3 2026-05-18
one of the few competitions left that humans must do the problem definition     
first before applying agent optimization                                        

Tom ▲1 2026-05-18
Turning "plan mode" in Claude and carefully define the problem by myself worked 
well for me.                                                                    

Tom ▲3 2026-05-18
Workflow (every new experiment):                                                

 1 Restate the geological problem (physical reality + constraints + signals)    
 2 Abstract it into a mathematical problem (alignment? inpainting? inverse      
   problem? assignment? consensus?)                                             
 3 Map each domain concept to a model component                                 
 4 Explore multiple reasonable formulations in parallel                         
 5 Only implement after the formulation is finalized                            

Navneet ▲-1 2026-05-18
Cool UI visualizer @tom99763                                                    

Tom ▲4 2026-05-18
I made a NN-based approach with some probabilistic modeling similar to          
@jeroencottaar Yale/UNC-CH solution. I haven't submitted yet but it seems a good
direction.                                                                      

🌆 
inbox%2F4310004%2F501a4be8bddc4492da4fd2bfac3fbd52%2Fensemble_vs_truth_5wells_v2
.png?generation=1779095472915796&alt=media                                      

🌆 
inbox%2F4310004%2F44863581fda421aee8b3c431f68898ae%2Fheatmap_boxes_5wells.png?ge
neration=1779096241756462&alt=media                                             

hengck23 ▲2 2026-05-18
maybe this is useful for you: https://www.youtube.com/watch?v=fEf6i2A0jdo       
https://www.youtube.com/watch?v=vQDbKR3NAlM                                     

check the lecture from 00 to 05 etc                                             

PatrickAIForFun ▲2 2026-05-18
When you say similar to Jeroen's solution, do you mean you actually modelled a  
prior and are optimizing the TVT math to minimize some measruement of mismatch  
between the TVTs? If yes, this is highly interesting, but I don't see how a CNN 
would fit into this. What am I missing or are you willing to share more?        

Tom ▲3 2026-05-18
Yes, I’ve actually built few priors and have been working on minimizing several 
measurements. I’ll release it once I have more completed experiments and the    
full map is built.                                                              

hengck23 ▲2 2026-05-18
", but I don't see how a CNN would fit into this" . put your prior in the loss: 

loss = regression loss + classification loss  + "too different from the prior   
loss"                                                                           

if you use probability:                                                         
too different from the prior loss = P( solution not drawn from prior            
disturibution)                                                                  

if you assume Gaussian, then it is related to "distance from prior solution"    

────────────────────────────────────────────────────────────────────────────────
similarly, if you use physics or geology equations (as prior), then it is loss =
regression loss + classification loss  + "much much the solution follows physics
equations"                                                                      

────────────────────────────────────────────────────────────────────────────────
if there is prior or physics, maybe predict residual is easier: solution =      
residual + prior                                                                

Tom 2026-05-18
I use this package: https://docs.pyro.ai/en/stable/                             

hengck23 ▲2 2026-05-18
i have a suggestion for you. use seq transformer to learn segments (span of md  
of similar dip) and output as:                                                  

                                                                                
 segmentation output:                                                           
 1111222222333333333333334444444444444444455666666666666666                     
                                                                                
 auxilarly ouput                                                                
 dip                                                                            
 xxxyyyzzzz                                                                     
                                                                                
 DTW                                                                            
 etc ....                                                                       
                                                                                
 tvt                                                                            
                                                                                

actually segmenting should be the first step                                    

you can measure the goodness of each segment by GR fitting or DTW, so it is a   
"physics model"                                                                 

ask chatgpt to make segmentation ground truth by considering gradients (-1,0,+1)
of tvt and link them up (or a very slow method called monotonic regression on   
gradient)                                                                       

────────────────────────────────────────────────────────────────────────────────
note, the segmntation results is not unqiue. so we can have top-k segmentation  
prediction                                                                      

Mohit ▲-1 2026-05-18
Great work out there also how is nn aproach used here?                          

Gaurav Rawat 2026-05-18
I try to do nowadays grill me for it to also grill before advising ..           

Durga Kumari ▲-1 2026-05-18
This is incredibly helpful, especially the TVT analogy.                         

hengck23 ▲3 2026-05-18
https://youtu.be/VgzFt7xknGo?si=rwz9Kv2oi3ZwniBE                                

time 27:50 shows (results? or the infrence  process?) ROGII automatic alignment 
and  segmentation                                                               

Tom ▲4 2026-05-19
This is what a casual segformer can achieve for me now                          

🌆 
inbox%2F4310004%2F8cb928831c0531e60d76135b542d654b%2Fheatmap_boxes_5wells.png?ge
neration=1779169616302808&alt=media                                             

hengck23 ▲2 2026-05-19
Azimuthal LWD Data Interpretation for UBCTDGeosteering Using a Physics-Informed 
Neural Network https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6000576      

🌆 
inbox%2F113660%2F0d83424de73277a9b0029db77e206b16%2FSelection_3682.png?generatio
n=1779172309683194&alt=media                                                    

"stay / steer_up / steer_down" this could be signed dip (-1,0,+1)               

Tom ▲1 2026-05-19
I think there's a opportunity to add some physical constraint in each segments  
for special behavior. Like controlling the curvature.                           

Tom ▲1 2026-05-19
🌆 
inbox%2F4310004%2Ff7f2ffedb845a1bda8d8796e78a188d4%2Fensemble_vs_truth_5wells_v2
.png?generation=1779175832490103&alt=media                                      

Tom ▲2 2026-05-19
Soft input segment worked. Just reach 0.97 overall CV with SegFormer.           

                                                 
  Fold      baseline   soft_seg input   Δ        
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  1         9.45       9.18             −0.27    
  2         10.73      10.43            −0.30    
  3         9.44       8.68             −0.76 ★  
  4         11.39      10.48            −0.91 ★  
  5         10.45      9.58             −0.87 ★  
  Overall   10.32      9.70 ⭐          −0.63    
  LB        10.576     ?                         
                                                 

Tom ▲2 2026-05-19
Update:                                                                         

Baysian Physical-informed SegFormer got very good result.  (0.94 cv score)      

                                                                                
                                                      bpinn                     
            baseline        soft_seg                  (077bpinn)                
  Fold      (077o)          input (077q)    Δ q-o     ⭐⭐           Δ bpinn-q  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  1         9.45            9.18            −0.27     9.18           −0.00      
  2         10.73           10.43           −0.30     9.79           −0.64 ★    
  3         9.44            8.68            −0.76 ★   8.45           −0.23 ★    
  4         11.39           10.48           −0.91 ★   10.13          −0.35 ★    
  5         10.45           9.58            −0.87 ★   9.35           −0.23 ★    
  Overall   10.32           9.70            −0.63     9.40 ⭐⭐      −0.30      
  LB        10.576 ✓        TBD             –         TBD            –          
                                                                                

It can formulate measurement equations to constrain the neural network through  
probabilistic modeling. I believe this could push the limits of GBDTs.          

🌆 
inbox%2F4310004%2F1dd9a77e93f676bdb46343b65d9d8a53%2F213.png?generation=17791888
00831447&alt=media                                                              

hengck23 ▲2 2026-05-19
If you consider just gr fitting aline, transformer is clearly better. Current   
notebook has better results because it is a fusion of multiple methods.         

You should analyse 1. Comparison with last value baseline( and public plane     
fitting baseline) 2. Normalised rmse ( ie divided by std if signal) 3. Error    
coming from well that is not sharing common vertical well? There are only about 
69 unique vertical wells.                                                       

If baseline is better than the worst cases, constraint to baseline should help  

hengck23 2026-05-19
you can try to add the following to the transformer as features, it should      
improve  results by 1~2                                                         

 • shared common typewell id                                                    
 • x,y,z, amz, inc                                                              
 • plane z  sampled from fitted geology plane  ancc, buda , etc ....            
 • different sommon filter                                                      
 • drop segment in training                                                     

in alphafold there is MSA  template matching to guide protein folding. in theory
all neighbour wells  can be used as template and input to transformer           

Mohit 2026-05-19
even that much is good                                                          

hengck23 ▲1 2026-05-20
if you can set some equations on                                                

                                                                                
                                                                                
 forward GR =   F.interpolate (predict_tvt, typewell_tvt, typewell_GR)          
                                                                                
 besides l2 loss, how to compare forward GR  and observation horizontal GR      
                                                                                
 or                                                                             
                                                                                
 forward geological formation = func(TVT, ...)                                  
                                                                                

Gaurav Rawat 2026-05-20
Thats great                                                                     

Tom ▲2 2026-05-20
On DTW inverse problem in the wavelet domain                                    

hengck23 2026-05-20
in the typewell GR domain would be better?                                      

Tom ▲2 2026-05-21
about fourier formation perspective on this problem                             

NobelK 2026-05-21
That's amazing!! How is this learning being done? I'm very curious about the    
details.                                                                        

Tom ▲2 2026-05-22
single bpinn reach 9.+ lb                                                       

🌆 
inbox%2F4310004%2F5365423f114f60c6b5d6ed711665b09e%2F123432.png?generation=17794
11315220089&alt=media                                                           

Gaurav Rawat 2026-05-22
Super what CV you getting ?                                                     

Tom 2026-05-22
9.62 cv score                                                                   

Gaurav Rawat 2026-05-22
ahh awesome cv GBDTs are bit worse ,,,                                          

Tom ▲2 2026-05-23
Sharing a diffeomorphic warping approach from my vesiuvius challenge solution.  
(Warp from a flat line instead                                                  

Tom 2026-05-23
So, building on Giba’s last-value baseline, a promising next step is to predict 
the direction at each MD position using sign(x), and then iteratively refine the
correction magnitude over N steps.                                              

Tom ▲1 2026-05-23
🌆 
inbox%2F4310004%2F166e648d61d3dfc05c97f85c5ff64b26%2F555.png?generation=17795020
89436723&alt=media                                                              

hengck23 ▲1 2026-05-23
i have a slightly different but similar idea:                                   

                                                                                
 treat it like a RL game:                                                       
                                                                                
 input solution                                                                 
 while some stop condition not meet:                                            
 - analyse and select a segment                                                 
 - push the segment up or down                                                  
 - accept if  action produce better results, else reject                        
 - repeat                                                                       
                                                                                
 such methods benefit from large data                                           
                                                                                

for me, since CNN+SDF can give end-to-end solution and can fit train data, my   
next plan is:                                                                   

 • massive train data by sythetic generator                                     
 • hierachy/iterative approcale to improve resoution                            

────────────────────────────────────────────────────────────────────────────────
i treat normal and wraped  as a kind of TTA for esembling (and also augmentation
for training) and different have different warping template                     

Tom ▲2 2026-05-23
Share another approach: curvature integration with teacher forcing warm start   

Tom ▲1 2026-05-23
🌆 
inbox%2F4310004%2Fbd55adc259858926aa5af71ed0c216b9%2F1.png?generation=1779536126
303326&alt=media                                                                

Now I develop a piecewise correction model by defining multiple pieces using two
split points, t1 and t2, along md.                                              

A piece is defined as:                                                          

$$ piece = sign(correction_{t1}^{oof} - model_{t1}^{oof}) \times                
sign(correction_{t2}^{oof} - model_{t2}^{oof})=-1 $$                            

The idea is to partition the space according to whether the correction          
directions are consistent between the two split points.                         

If the correction direction or magnitude inside a piece is incorrect, the model 
learns an adjustment term of the form:                                          

$$ sign(\alpha)\cdot |\alpha| $$                                                

Here, sign(alpha) controls the correction direction, while |alpha| controls the 
correction magnitude.                                                           

This allows the model to learn localized residual corrections in a piecewise    
manner.                                                                         

Tom ▲3 2026-05-23
Probabilistic modeling is shining (No GBDT, finish in 20min)                    

🌆 
inbox%2F4310004%2F60e41418f8624915f2bca7b830f6bc98%2F1.png?generation=1779544886
799314&alt=media                                                                

Gaurav Rawat 2026-05-23
Did dtw work for you                                                            

Tucker Arrants ▲7 2026-05-23
🌆 
inbox%2F4379159%2F22175fc1da830b548c670f8312200156%2FScreenshot%202026-05-23%201
45911.png?generation=1779562775107735&alt=media                                 

~1.2 ft behind you with simple UNet model. No physics constraints yet.          

Pre-training on synthetic wells gave a decent boost...                          

Tom ▲4 2026-05-27
🌆 
inbox%2F4310004%2Fd5b35c29393af30fe67920b379ef2185%2Ffield.png?generation=177988
8583205959&alt=media Working on Neural SDE now. I discover that forward-stepping
it start to learn. This is really badass.                                       

Tom ▲4 2026-06-03
Explaination about current trick and physical meaning in attachment             

And is it possible to do point sampling instead and making model predicting     
point connections. Just initial thoughts.                                       

🌆 
inbox%2F4310004%2F3dad47de922281e411d11a249991181c%2Fexample.png?generation=1780
491961954126&alt=media                                                          

hengck23 ▲4 2026-06-03
the public notebook (recent lb 8.63) uses meta-heuristics to decide k-beam or   
pf. this means some information is embedded at "whole well level", e.g. len of  
csv, min and max of well z, etc...                                              

problem of PF is that it is local and sequential (error accumulation):          
p(s_t|s_t-1).                                                                   

how about: p(s_t|distribution up to s_t-1).                                     

or even p(s_t|distribution exluding to s_t).                                    

🌆 
inbox%2F113660%2Feab42d972538dcdcf5912b11fc3086df%2FSelection_3869.png?generatio
n=1780530617337253&alt=media                                                    

🌆 
inbox%2F113660%2F438338a874344cbae87002c86d67d751%2FSelection_3870.png?generatio
n=1780530630436236&alt=media                                                    

🌆 
inbox%2F113660%2Fc849d82db43ad30674f359a52e1e4e56%2FSelection_3871.png?generatio
n=1780530643655204&alt=media                                                    

reference: Introducing DiffusionBlocks: Block-wise Neural Network Training via  
Diffusion Interpretation                                                        
http://pub.sakana.ai/diffusionblocks                                            

Sangram Patil ▲1 2026-06-06
How are you using the 2D U-Net? My input is [B, C, H, W] and the output is [B,  
H, W], but the model isn't performing well. I can't get the FT score below 14 no
matter what I try. Do you have any suggestions?                                 

Tucker Arrants 2026-06-08
Your output has the right shape. Ask yourself what each of the H rows along a   
single column is competing to be and whether you're scoring that competition, or
just regressing its shadow.                                                     

Look at what you're not feeding it that you already have. Review what Hengck is 
doing for a concrete example.                                                   

hengck23 ▲1 2026-06-09
"you're scoring that competition, or just regressing its shadow."               

within one column, SDF actually ranks all (typewell, horizontal) matches and    
gives results in "distance form". that is why it is so powerful.                

looks like regression, but we are actually doing ranking                        

Sangram Patil ▲1 2026-06-10
Thanks, guys, @tuckerarrants and @hengck23. I'm still confused about most parts,
so I've been using Claude and Gemini to help me understand them. At least I     
managed to build an SDF baseline that matches the CV-LB.                        

https://www.kaggle.com/code/sangrampatil5150/lb-15-41-cnn-sdf-train             

If you have any suggestions for improving the current pipeline, it would be a   
huge help. Thanks!                                                              

hengck23 ▲2 2026-06-10
Nice work getting an SDF baseline running.                                      

One caution: “matches CV–LB” is useful but not a litmus test, but it is not     
enough to prove the implementation is correct. e.g. A wrong local CV and a wrong
LB submission can still have a similar gap.                                     

I would suggest checking the pipeline in stages:                                

 1 start with baseline submission = last tvt value: local CV =12, LB=15 (we now 
   LB is about +2 worse than local)                                             
 2 any learning model must perform better than local cv 12 if it is correct     
   becuase if there is no better solution it sould predict null which leads to  
   12                                                                           
 3 to check correct modeling:                                                   
   a) overfit train (this is upper bound if there is no generalisation error).  
   SDF should easily gets to 8                                                  
   b) generalise validation to train. without augmentation, SDF should easily   
   get 11+. you probably can reduce it to 9/10 with augmentation.               
   c) submit to LB and try to keep gap. CV/LB consistent or less than 2         

Tucker Arrants ▲1 2026-06-10
Good start - keep trying and lean on the LLMs. It took me a little to get mine  
running, but it will give you a very good understanding of the problem / data   
once you do. I think there are a lot of approaches for this competition, so     
enjoy and be creative with your modeling.                                       

Tom ▲1 2026-06-12
Even though this was 5 years ago, I try seasonal trends again.                  

https://www.kaggle.com/code/tom99763/tensorflow-probability-inference?kernelSess
ionId=79007085                                                                  

Tom 2026-06-12
🌆 
inbox%2F4310004%2F2addfecdec246f6e27dc9016915f53a2%2F5255.png?generation=1781277
165146138&alt=media                                                             

