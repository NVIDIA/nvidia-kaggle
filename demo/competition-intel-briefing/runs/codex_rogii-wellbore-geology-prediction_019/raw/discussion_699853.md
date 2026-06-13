╭─ multi-trajectory prediction (MTP) with deep CNN for welllog inversion ──────╮
│ Author: hengck23 | Votes: 47 | Comments: 92 | Created: 2026-05-15            │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/699853                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

example notebook                                                                
https://www.kaggle.com/code/hengck23/cnn-mtp-example?scriptVersionId=320093395  

arvix paper: "Direct Multi-Modal Inversion of Geophysical Logs Using Deep       
Learning" - Sergey Alyaev                                                       
https://arxiv.org/pdf/2201.01871                                                
https://nfes.org/assets/workshop2022/ambrus_sequential_multi_mode_inversion_post
er.pdf                                                                          

                                                                                
 [2D Heatmap Input] ──> [Regression Head (CNN)] ──> [MDN Predictor (MLP)] ──>   
 [Multi-Trajectory Output]                                                      
                                                                                
 example of heatmap is shown below.                                             
 The Mixture Density Network (MDN) Predictor : for multiple paths hypothesis    
 (like k-beam).                                                                 
                                                                                
 if you can identify match keypoints in GR signals, then you decide how to      
 move/traverse between the matched keypoints.                                   
                                                                                

🌆 
inbox%2F113660%2F7482e84e8978ff732cdcc907d6f9d684%2FSelection_3550.png?generatio
n=1778852910817071&alt=media                                                    

🌆 
inbox%2F113660%2F6c2c44a7ecf5436b444bd418ea7b96bb%2FSelection_3551.png?generatio
n=1778852946405152&alt=media                                                    

🌆 
inbox%2F113660%2F2d908d758ec3e6c9b53ef3ea87db67f4%2FSelection_3552.png?generatio
n=1778852956595369&alt=media                                                    

── 92 Comments ──

 2026-05-15

hengck23 ▲5 2026-05-15
code and lesson (lecture notes)                                                 
https://github.com/geosteering-no/inversion_school_geosteering/tree/main        

🌆 
inbox%2F113660%2F65b2005e96e40dd4d414b3dfdd85433e%2FSelection_3558.png?generatio
n=1778855698254751&alt=media                                                    

🌆 
inbox%2F113660%2F0adbab27e641d3a2aac5b4068661f0ef%2FSelection_3559.png?generatio
n=1778855723413707&alt=media                                                    

hengck23 ▲4 2026-05-16
i make some plots. i think the formulation is not the issue. the issue is that  
the data is really noisy. It is difficult for human to match if we only see a   
window segment of vertical and horizontal GR signals.                           

🌆 
inbox%2F113660%2F367117bf4943a3dfbe13c65d2610ee58%2FSelection_3581.png?generatio
n=1778920361496093&alt=media                                                    

🌆 
inbox%2F113660%2F96e77a4ed1f167b5e6cd3db115f6b957%2FSelection_3580.png?generatio
n=1778920371424682&alt=media                                                    

🌆 
inbox%2F113660%2Fddc8ba9331ae0027d8867fb7ece5dc64%2FSelection_3579.png?generatio
n=1778920384125737&alt=media                                                    

hengck23 ▲2 2026-05-16
🌆 
inbox%2F113660%2F3286e32576b8ff4b0475bfc587e82d03%2FSelection_3583.png?generatio
n=1778925410203832&alt=media                                                    

hengck23 ▲1 2026-05-16
🌆 
inbox%2F113660%2F012dfe2b736f40de81808fb3b159e91f%2FSelection_3587.png?generatio
n=1778926195189495&alt=media  another example                                   

hengck23 ▲1 2026-05-16
🌆 
inbox%2F113660%2Fe45e674897cadfc1c911fe41a001025e%2FSelection_3590.png?generatio
n=1778927255860736&alt=media                                                    

hengck23 ▲5 2026-05-17
results is good at least for short-term forecast of 8 future interval steps.    
(each interval uses average of 32 GR values). here are validation resuits. black
is truth, orange is probability weighted average, red are top 8 paths (shade =  
probability)                                                                    

maybe top 6 is enough, becuase the last 2 never get activated                   

🌆 
inbox%2F113660%2F932e196fe4095e6f4f55da8c97a3c6ec%2FSelection_3639.png?generatio
n=1779007928439707&alt=media                                                    

🌆 
inbox%2F113660%2Ff8a5f84ff4a1d2c074772ae3a6c7f5ab%2FSelection_3636.png?generatio
n=1779008072963874&alt=media                                                    

hengck23 ▲5 2026-05-17
try a longer horizon of future steps=16, history=9. as expected, prediction     
starts to diverge. but good news is that the truth  path is still predicted as a
lower score candiate, eg top-6 solution .... maybe it can be saved.             

🌆 
inbox%2F113660%2F3d1618fab3daa67de8f8b21dffc2195d%2FSelection_3645.png?generatio
n=1779010588953513&alt=media                                                    

🌆 
inbox%2F113660%2F704e7019425783c99183719b432ab9b5%2FSelection_3646.png?generatio
n=1779010631239305&alt=media                                                    

🌆 
inbox%2F113660%2F4bdab51b7e1e45d5e54a6e3bf2c33d0b%2FSelection_3644.png?generatio
n=1779010684140700&alt=media                                                    

hengck23 ▲6 2026-05-17
example notebook                                                                
https://www.kaggle.com/code/hengck23/cnn-mtp-example?scriptVersionId=320093395  

hengck23 ▲1 2026-05-18
learning distance fields                                                        

🌆 
inbox%2F113660%2F8fd810720da6c9bb43a459af4c00d881%2FSelection_3663.png?generatio
n=1779087516149628&alt=media                                                    

🌆 
inbox%2F113660%2F7b979c9dbe6b996c6622453815b34dc6%2FSelection_3664.png?generatio
n=1779087502905288&alt=media                                                    

hengck23 ▲3 2026-05-18
Take-home message: mathematical correlation versus machine-learned correlation. 

So anything that is imperfect can be made perfect by learning. eg, we have our  
DTW needs to take care of reverse indexing. Although i found a paper on drop-DTW
(dropping invalid segments), it didn't work well because of noise. maybe i      
should learn the dropping and wraping i instead (of using DP)                   

🌆 
inbox%2F113660%2Ffc738ccfc225df547faa141545164e56%2FSelection_3665.png?generatio
n=1779091598108732&alt=media                                                    

🌆 
inbox%2F113660%2F6764f6c760325974da37aeef6fc5ccaf%2FSelection_3666.png?generatio
n=1779091874834092&alt=media                                                    

ROGII used another kind of feature image                                        

hengck23 ▲2 2026-05-18
🌆 
inbox%2F113660%2F6b3c30814618336c2f27b192da981e59%2FSelection_3667.png?generatio
n=1779093686204434&alt=media                                                    

cnn should be very good to capture these micro box patterns (pairs of 2d        
signal). these are just like 2d tokens. But i need to recreate ROGII segment    
endpoints annotations.                                                          

🌆 
inbox%2F113660%2Fe2b9ca5e21d0b0824f07e6bf5f7fb51e%2FSelection_3669.png?generatio
n=1779094329932256&alt=media                                                    

Tom ▲1 2026-05-18
Thanks, this is very useful info                                                

hengck23 ▲4 2026-05-20
what you get if you use unet and do "blood vessel" segmentation                 

🌆 
inbox%2F113660%2F1644b9d462a6dc335c4524519b9f0fc4%2FSelection_3697.png?generatio
n=1779280718291901&alt=media validtation                                        

🌆 
inbox%2F113660%2Fa1374e43674b9a496d943f24c69b25f6%2FSelection_3696.png?generatio
n=1779280730592253&alt=media 🌆 
inbox%2F113660%2Ff2f50ae71956df1388063a8347bf0ae8%2FSelection_3693.png?generatio
n=1779280617582300&alt=media  training                                          

                                                                                
                                                                                
     def forward(self, typewell, horizontal, hint):                             
                                                                                
         #todo raw signal channel                                               
         B,T = typewell.shape                                                   
         B,H = horizontal.shape                                                 
                                                                                
         image = torch.concat([                                                 
             typewell.reshape(B,1,T,1).expand(B,1,T,H),                         
             horizontal.reshape(B,1,1,H).expand(B,1,T,H),                       
             hint,  #input tvt                                                  
         ], dim=1)                                                              
                                                                                

hengck23 ▲1 2026-05-20
i am surprised that some results are perfect and it is bidirectional and needs  
not to be continuous (e.g. match can happen in the middle of image and propagate
out)                                                                            

🌆 
inbox%2F113660%2F60655737a6b91ca7dbf19120c99679af%2FSelection_3698.png?generatio
n=1779281129246499&alt=media                                                    

Tom ▲2 2026-05-20
might can consider problem as iterative image inpainting I think.               

hengck23 ▲2 2026-05-20
I am surprised there is no multiple paths. I only use bce loss. Some paths      
diverted from the truth with high confidence. It means that if we use gr        
information, we can very similar train labels that “diverge” from the validation
labels. I have no ideas how to correct these                                    

hengck23 ▲4 2026-05-22
one challenge of the competition is to find good representation. Here is using  
cnn + sdf (signed distance function)                                            

🌆 
inbox%2F113660%2F82addc9995928668d4d83a594a3aba6d%2FSelection_3707.png?generatio
n=1779416645785838&alt=media 🌆 
inbox%2F113660%2F021757f49164af8ba6d54ff2eda88ba2%2FSelection_3708.png?generatio
n=1779416678139291&alt=media                                                    

Tom ▲2 2026-05-22
SDF seems like a solid option. This also reminds me of the Vesuvius Challenge,  
might be able to transfer some tricks from there.                               

hengck23 ▲5 2026-05-22
@tom99763                                                                       

demo inference and training code are up:                                        
https://www.kaggle.com/code/hengck23/cnn-sdf-example                            
https://www.kaggle.com/datasets/hengck23/hengck23-rogii-cnn-mtp-demo  (training 
py file)                                                                        

hengck23 ▲1 2026-05-22
The fact that CNN can detect micro 2d pattern makes me think that the data are  
probably synthetic or the signal modelling in geology is really good?           

hengck23 ▲1 2026-05-22
i am thinking of predicting the geology plane, eg ANCC = tvt -z instead. such   
planes are more linear and benefit from sdf (natural smoothness and planar      
regularisation from ground truth!)                                              

Tom ▲1 2026-05-22
Tvt - z can work better than directly predicting tvt.                           

hengck23 ▲2 2026-05-22
instead of                                                                      

                                                                                
 mistfit_gr = t_gr-h_gr                                                         
                                                                                

use                                                                             

                                                                                
 mistfit_gr = t_gr- interpolate( h_tvt-well_z, h_tvt, h_gr)                     
                                                                                

maybe you can see a linear zero line (matched gr)                               

Tom ▲2 2026-05-22
🌆 
inbox%2F4310004%2F186f7cae20c4672192ec571a8ca4a0a6%2F555.png?generation=17794633
72319471&alt=media                                                              

hengck23 ▲3 2026-05-22
🌆 
inbox%2F113660%2Fb2173b1873b283eaa3fe6f29ff3b27ca%2FSelection_3719.png?generatio
n=1779464474310165&alt=media                                                    

i tried some toy data                                                           

hengck23 ▲2 2026-05-22
So the pf, k-beam, dp, viterbi etc searches are just detecting lines or multi   
ple lines hypothesis.                                                           

But there is an issue, ancc plane anchoring means the range of tvt is very small
if the geological plane is horizontal, ie no gr pattern to match. Need to modify
the anchoring                                                                   

hengck23 ▲6 2026-05-22
update on cnn+sdf:                                                              

 • some backbone and decoder architecture  are better                           
 • augmention using flip + different stretch improve results                    
 • time to spend on generator to generate more possible train data: create path 
   --> sample from typewell --> add oise (actually we can do it in test-time or 
   better still offline since, we have the hidden testwell location in host     
   slides)                                                                      

sleep3r ▲1 2026-05-23
Tried a sliding-window CNN that predicts TVD corrections over the base prior,   
using horizontal GR + typewell correlation. Added synthetic pretraining to teach
the correlation - worked great on synthetic data (93% accuracy), completely     
failed to transfer to real wells. Spent way too long on that                    

Net result: ~+0.03 ft over baseline. Looking at the leaderboard that's basically
nothing                                                                         

sleep3r ▲2 2026-05-24
I rechecked with real train-well panels. The issue seems not only CNN/MTP       
capacity: the true TVT path often is not a reliable high-score ridge in the     
GR/typewell heatmap. In our localized/stretch panel, normal GR top10 coverage is
25.6%, shuffled GR is 26.1%. The first thing that strongly changes the search is
stratigraphic/zone restriction, but direct Geology/formation labels are not     
available in test                                                               

🌆 
inbox%2F2776455%2Fb8c348c0d1b8f1e0b36cd00cec46f10b%2Freal_heatmap_failure_cases_
verified.png?generation=1779616479104845&alt=media                              

🌆 
inbox%2F2776455%2Fb025a627679711185317634b5568525d%2Freal_zone_restriction_case_
verified.png?generation=1779616493930764&alt=media                              

hengck23 ▲1 2026-05-24
how about let GR = concate (gr values, location values). then each GR value is  
diiferent. correlation is match of values and distance                          

sleep3r ▲1 2026-05-24
I tested this exact idea: combine GR matching with a test-safe location prior   

Concatenating / combining location with GR absolutely helps remove global false 
ridges But the effect seems to come from the location prior, not from GR itself 

When I keep the exact same location prior and replace lateral GR with shuffled  
GR, the heatmap still looks very similar and the top1 path behaves similarly. So
the problem is not just “GR needs location”; the remaining GR/typewell          
likelihood still does not pass shuffled-GR sanity                               

🌆 
inbox%2F2776455%2F3de4d88a923923400a084ecba04d9e50%2FConditional%20heatmap%20sto
ry.png?generation=1779638149048668&alt=media                                    

hengck23 ▲1 2026-05-24
The problem of geosteering is actually "move the wellbore between the target top
and bottom geology region." Hence, here the inversion is localised, where is the
wellbore within the layers? You can estimate the limits first                   

hengck23 ▲1 2026-05-25
🌆 
inbox%2F113660%2Fe010a894530d74bdcaf984b63dd75a87%2FSelection_3756.png?generatio
n=1779668181982385&alt=media check 10a1281a.png in the train dataset            

🌆 
inbox%2F113660%2F23efbd5535f7f5a5ada87119166bd94c%2FSelection_3757.png?generatio
n=1779668326819854&alt=media                                                    

the reference TW GR signal for matching is only "so short". many of the         
horziontal GR "windows" are not useful at all except for the peaks              

hengck23 ▲3 2026-05-25
🌆 
inbox%2F113660%2F432b163cebebbc135cc5dd172f30bd81%2FSelection_3768.png?generatio
n=1779686548582667&alt=media i can do some fast match from visual inspection if 
the well                                                                        

 • look for highest and lowest point                                            
 • check neighbourhood values from that point                                   
 • then you can find large segment and you can almost get find min/max of well  
   tvt                                                                          

────────────────────────────────────────────────────────────────────────────────
it seems to me the logic is:                                                    

 • if you are lost, continue to move in a direction when you find a prominent GR
   pattern (usually high or low values), so that you can reset to a known       
   position.                                                                    
 • then back track to where you are lost.                                       

Tom ▲2 2026-05-25
Developing a “Trace Back” mechanism could further improve the score. One        
possible approach is to build a dictionary (or bag-of-signals) that serves as a 
strong reference for matching                                                   

hengck23 ▲5 2026-05-25
i suddenly have a cheat method.                                                 

 1 you are at typewell location s at PS.                                        
 2 we are not interested in tracinig the well trajetory. rather we are          
   interested in detecting the max and min offset values, where TW_GR( a*tvt +  
   offset) can be matched in horizontal well.                                   
 3 so we can create many templates of TW_GR(a*tvt + offset) with different      
   values a and scale.                                                          
 4 once we have this, just predict trajectory = (max tvt + min tvt)/2. if you do
   this correctly, you get rmse about 8.5                                       

sleep3r ▲1 2026-05-25
i tried a similar direction: instead of trusting one global GR heatmap, i build 
local GR-event candidates and then use a chunk-level DP policy to stitch/select 
a smooth path                                                                   

early result: this does add useful candidate space. on an 80-well diagnostic,   
oracle use of these traceback bands improved baseline rmse from ~9.99 to ~8.76  
on covered rows. a small chunk-DP smoke test also improved ~9.47 → ~9.24        

but the caveat is important: shuffled/zero-GR sanity is still not clean, so the 
current signal is not pure GR matching yet. it seems useful as sparse reset     
anchors / candidate bands, but needs full OOF validation before trusting it     

hengck23 ▲5 2026-05-25
i discover a hack!                                                              

🌆 
inbox%2F113660%2Fcbec65ca59cef57ac8e87cda01a89638%2FSelection_3770.png?generatio
n=1779707786411769&alt=media                                                    

first fig: dz                                                                   
second fig: dtvt                                                                
why? annotation leak!  (that is how starsteer works)                            

hengck23 ▲1 2026-05-25
🌆 
inbox%2F113660%2F7a3e5b34d2d39a46578bbc0612432d7a%2FSelection_3772.png?generatio
n=1779708063963239&alt=media                                                    

Tom ▲2 2026-05-25
The red/blue = direction segments sign(dtvt). My test just confirmed the        
structure underneath it: ANCC (formation top) is ~piecewise-linear with ~15     
control points per well (~323 rows apart). That is the sparse StarSteer dip     
annotation. LOL                                                                 

hengck23 ▲1 2026-05-25
maybe just prediction dtvt = a(dz)*dz. i.e. your network predict dtvt and use   
both local dtvt loss and global cumsum tvt loss                                 

sleep3r ▲1 2026-05-25
yeah, this seems real. I tried using ANCC only as a train-time teacher:         

target: sign(dANCC) = down/flat/up features: test-safe MD/X/Y/Z/GR/TVT_input    
only                                                                            

5-fold OOF hidden direction accuracy is ~0.927. so formation-top annotation     
seems distillable into a test-safe state model. now checking if this state helps
chunk/DP path selection                                                         

🌆 
inbox%2F2776455%2F3b867a232864f3e27d0ec194472fdce5%2Ftop%20state%20confusion.png
?generation=1779711803375834&alt=media                                          

hengck23 ▲2 2026-05-25
🌆 
inbox%2F113660%2F2e33fe9a6792e3c2cf6a0bec380f83b1%2FSelection_3774.png?generatio
n=1779712938987344&alt=media                                                    

i plot dz and dtvt on the same plot. they are the same scale !!!!  maybe        
competition will reset                                                          

hengck23 ▲3 2026-05-25
                                                                                
     h_tvt = h["TVT"].values                                                    
     h_z = h["Z"].values                                                        
     h_md = h["MD"].values                                                      
     h_dtvt = np.gradient(h_tvt)                                                
     h_dz   = np.gradient(h_z)                                                  
                                                                                
     plt.plot(h_md, -h_dz)                                                      
     plt.plot(h_md,  h_dtvt)                                                    
     plt.axvline(x=h_md[h_ps], color='red', alpha=1)                            
     #plt.plot(dz_smooth)                                                       
     plt.show()                                                                 
                                                                                

🌆 
inbox%2F113660%2F9c7fa4aecce837ad1c9555fd58260894%2FSelection_3776.png?generatio
n=1779713436118745&alt=media                                                    

🌆 
inbox%2F113660%2F0f7c1cba4a69dfe468bdab179848576f%2FSelection_3775.png?generatio
n=1779713451250997&alt=media                                                    

Tom ▲2 2026-05-25
−dz and dtvt being the same scale and overlapping in long stretches means:      
wherever the formation is flat, dtvt = −dz exactly (dANCC=0 → TVT = −Z + C).    
They only diverge at dip events (your ~15 control points), and the              
parallel-offset stretches in your middle plot are exactly those flat segments   
where TVT = −Z + a constant                                                     

Tom ▲1 2026-05-25
time to reset now                                                               

hengck23 ▲1 2026-05-25
🌆 
inbox%2F113660%2F74c2ea8b52185f4e7271635bc13f4406%2FSelection_3777.png?generatio
n=1779714564807668&alt=media                                                    

                                                                                
    h_dtvt = np.gradient(h_tvt)                                                 
     h_dz   = np.gradient(h_z)                                                  
                                                                                
     H_unknown = len(h_tvt) - h_ps                                              
     truth_tvt = h_tvt[h_ps:]                                                   
     ##---                                                                      
     #find offset                                                               
     offset = h_dtvt[h_ps-500:]+h_dz[h_ps-500:]                                 
     offset = np.median(offset)  #use ML to learn offset                        
                                                                                
     predict_dtvt = -h_dz[h_ps:]+offset                                         
     predict_tvt = np.zeros((H_unknown,))                                       
     predict_tvt[0] = h_tvt[h_ps]                                               
     for i in range(1, H_unknown):                                              
         predict_tvt[i] = predict_tvt[i-1] + predict_dtvt[i]                    
     #                                                                          
     print(len(predict_tvt), len(truth_tvt)) #additional point at h_ps          
     rmse = np.sqrt(np.nanmean((predict_tvt - truth_tvt)**2))                   
                                                                                
     plt.plot(predict_tvt, label=f"predict_tvt {rmse:0.2f}")                    
     plt.plot(h_tvt[h_ps:], label="h_tvt")                                      
                                                                                

hengck23 ▲1 2026-05-25
i think the offset could be fixed values. my experiments seems to suggest they  
are limited to set of values                                                    

hengck23 ▲4 2026-05-26
🌆 
inbox%2F113660%2Fcc3444aeeff0e19ed1c4c96eca54ea6d%2FSelection_3781.png?generatio
n=1779758427853890&alt=media                                                    

🌆 
inbox%2F113660%2F38f6ef9c3f549066e8ccf0ffb6ea0559%2FSelection_3782.png?generatio
n=1779758442569126&alt=media                                                    

Tom ▲1 2026-05-26
cumsum(−dz − offset) with a discrete offset  => 7.7 rmse                        

hengck23 2026-05-26
Just need a classifier to choose global offset                                  

sleep3r ▲1 2026-05-26
a fine offset-grid oracle gives ~7.64 RMSE on train hidden rows for me          

but choosing the offset is the hard part: known-prefix offset gives ~37-39 RMSE,
and my fold-safe selector only gets ~14.8. So I think the next step is not      
direct TVT regression, but learning the offset/state with cumulative TVT loss   

I started a no-prior model around this idea: predict residual dC / offset-state 
from test-safe MD/X/Y/Z/GR/TVT_input, then reconstruct TVT by cumsum            

still early, but this formulation feels much closer to the leak than my previous
GR/MTP attempts                                                                 

Tom 2026-05-26
Fuzzy inference or mixture desnity network would help                           

hengck23 2026-05-26
The first try should be :                                                       

                                                                                
 1) given current location s                                                    
 2) given a list of offset = -0.1 to 1.0                                        
 3) given a list of  future location s1 = 25,50,75, 100, ... 300                
 4) compute tvt rmse for each candidate pair (offset,s1) above :   tvt rmse =   
 rmse (true tvt[s0:s1], tvt derived from dz and offset)                         
 5) train a regressor : score = model( h_gr_smooth[s0:s1], sampled gr using dz  
 and offset, aux input)                                                         
 6) score from (5) must correlate with  tvt rmse from (4). or at least the min  
 point should coincide                                                          
                                                                                

hengck23 2026-05-26
brute force search is  12.18 for one fold                                       

                                                                                
                                                                                
     t = pd.read_csv(f"{KAGGLE_DIR}/train/{sample_id}__typewell.csv")           
     h = pd.read_csv(f"{KAGGLE_DIR}/train/{sample_id}__horizontal_well.csv")    
     h_ps = int(np.flatnonzero(h["TVT_input"].notna().values)[-1])              
                                                                                
     h_gr_filled = h["GR"].interpolate().bfill().ffill().values                 
     h_gr_smooth = savgol_filter(h_gr_filled, 100, 3)                           
     h_tvt = h["TVT"].values                                                    
     h_z = h["Z"].values                                                        
     h_md = h["MD"].values                                                      
     h_ancc  = h["ANCC"].values                                                 
     h_dtvt  = np.gradient(h_tvt)                                               
     h_dz    = np.gradient(h_z)                                                 
     h_dancc = np.gradient(h_ancc)  #ground truth offset                        
                                                                                
                                                                                
     span = [100]  # let's try one                                              
     offset =  np.linspace(-0.8, 0.8, 201)  # covers 90% of cases               
                                                                                
                                                                                
     rmse_tvt = []                                                              
     rmse_gr  = []                                                              
     rmse_tvt_score = np.zeros((len(span), len(offset)))                        
     rmse_gr_score = np.zeros((len(span), len(offset)))                         
                                                                                
     predict = []                                                               
     s0=h_ps                                                                    
     while s0<len(h_tvt):                                                       
         best_tvt = None                                                        
         best_tvt_rmse = np.inf                                                 
         best_gr = None                                                         
         best_gr_rmse = np.inf                                                  
         for si, sp in enumerate(span):                                         
             s1 = s0+sp                                                         
             s1 = min(s1,len(h_tvt))                                            
             for j in range(len(offset)):                                       
                                                                                
                 sm_tvt = last + (h_dz[s0:s1]-offset[j]).cumsum()               
                 sm_gr  = np.interp(sm_tvt, t["TVT"].values, t["GR"].values)    
                 r_tvt =  do_rmse(sm_tvt,h_tvt[s0:s1])                          
                 r_gr  =  do_rmse(sm_gr,h_gr_smooth[s0:s1]) #- si*0.5           
                                                                                
                 rmse_gr_score[si,j] = r_gr                                     
                 rmse_tvt_score[si,j] = r_tvt                                   
                 if r_gr<best_gr_rmse:                                          
                     best_gr_rmse = r_gr                                        
                     best_gr = [s0,s1,sp,j,offset[j], r_tvt]                    
                                                                                
                 if r_tvt < best_tvt_rmse:                                      
                     best_tvt_rmse = r_tvt                                      
                     best_tvt = [s0, s1, sp, j, offset[j], r_gr]                
                                                                                
         rmse_tvt.append(best_tvt_rmse)                                         
         rmse_gr.append(best_gr_rmse)                                           
                                                                                
         #                                                                      
 plt.imshow(np.hstack([stats.zscore(rmse_tvt_score),stats.zscore(rmse_gr_score) 
 )                                                                              
         # plt.waitforbuttonpress()                                             
                                                                                
         s0, s1,_, j, _, r_tvt = best_gr                                        
         s1 = int(0.8*s0+0.2*s1)  #back track ... don't trust it                
         if s0==s1: s1=s0+1                                                     
                                                                                
         p_gr  = last + (h_dz[s0:s1]-offset[j]).cumsum()                        
         predict.append(p_gr)                                                   
         print(s0, s1-s0, offset[j], r_tvt, best_gr_rmse)                       
         s0=s1                                                                  
                                                                                
     predict = np.concatenate(predict)                                          
     r = do_rmse(predict,truth)                                                 
     print('***',r)  #rmse for one                                              
     all.append(r)                                                              
                                                                                
 print("-------------------------------------")                                 
 print(np.mean(all)) #12.18 (not the same as lb metric with mean over all rows  
 (not sample wise)                                                              
 exit(0)                                                                        
                                                                                

hengck23 ▲2 2026-05-26
there is a mxiture/DP transformer that chatgpt recommend:                       

Lattice Deduction Transformers https://arxiv.org/html/2605.08605v1              

                                                                                
 class RogiiLatticeTransformer(nn.Module):                                      
     """                                                                        
     Simple lattice transformer for trajectory prediction.                      
                                                                                
     Input:                                                                     
         h_seg:       (B, S, L) horizontal GR split into S segments, each lengt 
 L                                                                              
         t_gr_bins:   (B, N, L) typewell/reference GR windows for each TVT bin  
         alive:       (B, S, N) current lattice candidates, 1=alive, 0=removed  
                                                                                
     Output:                                                                    
         keep_logits: (B, S, N) logits saying whether candidate TVT bin should  
 remain alive                                                                   
         move_logits: (B, S-1, A) optional movement logits between segments     
     """                                                                        
                                                                                

 2026-05-26

hsiaosuan ▲1 2026-05-29
Reminds me of Vesuvius!!                                                        

Tucker Arrants ▲-1 2026-05-29
I think they need to reset. Surely providing the post-PS trajectory (X/Y/Z) is a
problem? It's causally downstream of the answer - the driller steered based on  
where the formation actually was, so the trajectory ahead of the bit already    
encodes what we're supposed to be predicting. Feels like it should be masked.   

hengck23 ▲1 2026-05-29
Not direct nor obvious answer. Still needs some clever hack to work. But does   
make getting answer easier.                                                     

PatrickAIForFun ▲2 2026-05-29
I don't think a reset is necessary. If you look at all training videos and      
resources by ROGII one can clearly see that there are two types of geosteering  
which are done in the real world:                                               

 • live geosteering: get the data from the current bore-head and give it        
   directions to stay in the oil. Here you are also given XYZ and GR up to the  
   current position and can't change previous decisions.                        
 • post-drilling steering: Here you are also given the full XYZ trajectory amd  
   the full GR log and now have to determine the rock structure you are drilling
   through. This is exactly what our task is and what is shown in most          
   StarSteer-Geosteering training videos. In the real world you are also given  
   the true XYZ and can assume thag during live-steering it followed the rock   
   formation. I guess the goal here is to get a post-hoc understanding of the   
   rock for future well planning.                                               

Either way, in the real world application we are also given this exact same     
data.                                                                           

hengck23 ▲2 2026-06-01
effects of hacks                                                                

no GR features are used.                                                        
input only use x,y,z,dz,dtvt history, tvt history                               

🌆 
inbox%2F113660%2F5cbe8ff0043173577783c4f8888a0072%2FSelection_3815.png?generatio
n=1780280656032761&alt=media                                                    

left: validation, right: train                                                  
red: predict, blackL ground truth                                               
(do note the scale of the y axis when interpreting results)                     

predict task: dtvt                                                              
history = 256, future horizon = 1024 (2048 shows smiliar results)               
model: just normal transformer                                                  

there are some dift, if i can solve those using GR, maybe good results.         
i am thinking of estimate dift = oberserved GR - interp(predict tvt,            
typewell_tvt, typewell_gr) at some fixed intervals/anchors (maybe CNN is useful 
here)                                                                           

hengck23 2026-06-01
                                                                                
         seq = torch.cat([                                                      
            h_dtvt_history.reshape(B,H,1),                                      
            h_tvt_mask.reshape(B,H,1),                                          
            h_dz.reshape(B,H,1),                                                
            h_x.reshape(B,H,1),                                                 
            h_y.reshape(B,H,1),                                                 
            h_cos.reshape(B,H,1),                                               
            h_sin.reshape(B,H,1),                                               
         ], dim=2)                                                              
         #print(seq.shape)                                                      
                                                                                
         seq = self.to_seq(seq) #project to dmodel                              
         h_idx = torch.arange(H, device=device)                                 
                                                                                
         seq = seq + self.h_idx_emb(h_idx).reshape(1,H,-1) #pos encode          
         seq = torch.concat([                                                   
             self.cls.reshape(1,1,-1).repeat(B,1,1),                            
             seq,                                                               
         ], dim=1)                                                              
                                                                                
         padding_mask = h_padding>0.5 #convert to bool (B,H)                    
         padding_mask = F.pad(padding_mask, (1,0), value=False) # add cls (B,1+ 
         hidden = self.tx_encoder(seq,                                          
 src_key_padding_mask=padding_mask)#src_key_padding_mask include cls            
         cls, hidden = hidden[:,0], hidden[:,1:]                                
                                                                                
         dtrajectory = self.dtrajectory(hidden)                                 
         dtrajectory = dtrajectory.permute(0,2,1) + h_dtvt_history.permute(0,2, 
 #B,K,H                                                                         
                                                                                

hengck23 ▲3 2026-06-01
🌆 
inbox%2F113660%2F273a4ff880f68393b18cd7616bff9799%2FSelection_3816.png?generatio
n=1780291755847831&alt=media                                                    

transformer MTP of the previous post. I just need a good verifier               

hengck23 2026-06-01
🌆 
inbox%2F113660%2Feabb38ebc179e42f8e422a636e5acbe0%2FSelection_3817.png?generatio
n=1780293803969071&alt=media                                                    

wqi876 ▲2 2026-06-01
Thank you very much for your discussion. It has been very helpful to me. And    
your profile picture is so cute!                                                

sleep3r 2026-06-01
gr matching is ill-conditioned: even at the true tvt horizontal <--> typewell gr
only corr ~0.7, and offset error compounds                                      

hengck23 2026-06-01
The best way is to make a model that can recover h tvt from h gr = interp( h    
tvt, tw gr, tw tvt). This is perfect correlation but has multiple fp matches. If
that works, you can introduce gausssian noise, offset noise, scale noise,       
simplifcation noise, etc as                                                     

hengck23 2026-06-01
My feeling is that we need to train a ranker or scorer rather than rely on      
generic correlation                                                             

sleep3r 2026-06-01
been down exactly this road                                                     

mtp heatmap net + a learned catboost ranker over the modes (pairwise yetirank) +
the h_tvt-from-h_gr recovery + gaussian/offset/scale/simplification aug         

two walls i couldn't pass: gr is barely discriminative for selection - no_gr ≈  
shuffled_gr ≈ real_gr on top1, the net basically ignores it. and the ranker     
looks amazing in-window (spearman score <-> error ~−0.92, top3 rate .92) but it 
does NOT convert row-level once you go strict well-grouped oof - my best honest 
gain over gbm was ~+0.03ft, the big in-sample numbers were pure ranker leakage  

on top of that ~23% of wells have no good candidate at all, so the scorer is    
capped no matter how good it is                                                 

hengck23 2026-06-01
we do not need to match all GR. we have good dtvt estimate. we just need a few  
anchor points to push the whole tvt curve to correct the pace.                  

sleep3r ▲1 2026-06-01
agreed a few anchors is all you need - with oracle anchors i get k≈10 down to   
~4ft, k≈20 to ~1.7ft, so your pace-correction framing is right                  

the catch is placing them: a local gr shift-search around an anchor just can't  
localize it. the typewell gr repeats, so sliding the curve +-tens of ft fits the
log about equally well                                                          

my gr-picked anchors gave basically zero gain over no correction, even after    
gating to only high-corr anchors. so imo the wall isn't "match all gr vs a few  
anchors", it's getting even one trustworthy anchor out of gr. how are you       
deciding which anchors to trust?                                                

hengck23 ▲3 2026-06-02
@sleep3r                                                                        

my suggestion is that you start with the native PF method from                  
https://www.kaggle.com/code/sunnywu27/rogii-wellbore-tvt-physical-model         

then replace the likelihood scorer with a learned local CNN one                 

                                                                                
         #initialisation                                                        
         # tvt = geo_z - z + bias                                               
         # geo_z+bias = tvt + z                                                 
         w = np.ones(num_particle) / num_particle                               
         pos = last_tvt + last_z + 2.0 * rng.standard_normal(num_particle)      
         vel = last_vel + 0.01 * rng.standard_normal(num_particle)              
                                                                                
         cum_log_likelihood = 0.0                                               
         output = h['TVT_input'].values                                         
         for i in range(h_ps+1, len(h)):                                        
             dm = h_md[i] - h_md[i-1]                                           
                                                                                
             #create particle                                                   
             vel = 0.998 * vel + 0.002 * rng.standard_normal(num_particle)      
 #chnage this to torch tensor                                                   
             pos = pos + vel*dm + 0.005 * rng.standard_normal(num_particle)     
                                                                                
             tvt = pos - h_z[i]                                                 
             tvt = np.clip(tvt, t_tvt[0] - 100, t_tvt[-1] + 100)                
             pos = tvt + h_z[i]                                                 
                                                                                
                                                                                
                                                                                
             #---                                                               
                                                                                
                                                                                
             ##--- change this to CNN learned likelhood score ---------------   
             #gr  = np.interp(tvt, t_tvt, t_gr)                                 
             #gr_error = gr - h_gr[i]                                           
             #gr_std = 30                                                       
             #gr_likelihood = np.exp(-0.5 * np.minimum((gr_error / gr_std) ** 2 
 600.))                                                                         
             ##--- change this to CNN learned likelhood score ---------------   
             #e.g.                                                              
             t_win_gr = extract a window from t_gr (refrence well)              
             h_win_gr = extract a window from h_gr (horizontal well)            
             gr_likelihood = net(t_win_gr, h_win_gr)                            
                                                                                
             image = torch.cat((t_win_gr.unsqueze(1), h_win_gr.unsqueze(2)),    
 dim=0)                                                                         
             feature = cnn(image)                                               
             --> learn to match sdf = t_win_tvt - h_win_tvt                     
                                                                                
                                                                                
             likelihood = gr_likelihood                                         
             likelihood = np.maximum(likelihood, 1e-300)                        
             avg = float((w * likelihood).sum())                                
             cum_log_likelihood += np.log(max(avg, 1e-300))                     
                                                                                
             #updte weight                                                      
             w = w * likelihood                                                 
             if w.sum() == 0:                                                   
                 w = np.ones(num_particle) / num_particle                       
             else:                                                              
                 w = w / w.sum()                                                
                                                                                
             #resample                                                          
             effective = 1.0 / np.sum(w ** 2)                                   
             if effective< 0.5 * num_particle:                                  
                 idx = rng.choice(num_particle, size=num_particle, replace=True 
 p=w)                                                                           
                 pos = pos[idx] + 0.1 * rng.standard_normal(num_particle)       
                 vel = vel[idx] + 0.001 * rng.standard_normal(num_particle)     
                 w = np.ones(num_particle) / num_particle                       
                                                                                
             #prediction                                                        
             predict = (w*pos).sum() - h_z[i]                                   
             output[i] = predict                                                
                                                                                

sleep3r 2026-06-02
tried your cnn-likelihood idea pretty hard                                      

window matcher (h_gr vs typewell, learn the sdf) + noise aug, fold-safe.        
couldn't get it to beat the plain point-gr likelihood                           

matcher's peak sits ~200ft off the true tvt and AUC caps ~0.7, the gr just      
repeats too much to localize a window                                           

────────────────────────────────────────────────────────────────────────────────
what I did confirm though:                                                      

pf framework itself is great - drop in an oracle likelihood and even at         
alpha=30ft it nails ~1ft, so the whole game is purely likelihood centering and  
gr alone can't give it                                                          

the thing that actually moved my score was blending the pf with a gbm - their   
error tails are decorrelated, big drop                                          

does your cnn ever beat point-sr on a strict well-grouped oof? that's where mine
died                                                                            

hengck23 ▲1 2026-06-02
Cnn does improve on specific cases and but general cases.                       

hengck23 2026-06-02
examples of different methods                                                   
CNN+sdf (using gr) : global gr waveform pattern                                 

🌆 
inbox%2F113660%2F032ab3969a45e90ee2a8550cf1b56060%2FSelection_3828.png?generatio
n=1780444354405051&alt=media                                                    

🌆 
inbox%2F113660%2Fdceca7fac2a4b00782ce8925f43aeecf%2FSelection_3824.png?generatio
n=1780444415007630&alt=media transformer on dz (not using gr) : dz prior        
                                                                                

🌆 
inbox%2F113660%2F528ac14031f416233593579c70efad9d%2FSelection_3829.png?generatio
n=1780444463192312&alt=media PF on single value GR :  local  gr match based on l

hengck23 ▲2 2026-06-03
The PF code uses lookup geo plane for training wells. what if we model the geo  
surface using grid interpolation? a validation rmse of 11.09 (non optimized)    

🌆 
inbox%2F113660%2F55ea900509c0b2fedd3cbbce2c6669a3%2FSelection_3846.png?generatio
n=1780470740954921&alt=media                                                    

PatrickAIForFun 2026-06-03
Yes, I can confirm - basic / non optimized kriging of geology layers gave a     
local RMSE ~11 and test rmse ~13.5 for me.                                      

hengck23 2026-06-03
Try better offset adjustment. Plot the graphs. Validation rmse should be near   
11. Use the tvt input, geo predict and given well z near ps to determine best   
offset                                                                          

hengck23 2026-06-03
🌆 
inbox%2F113660%2Fb98b9842eb587b380ecfe9bed6792695%2FSelection_3849.png?generatio
n=1780489051202374&alt=media maybe this is helpful                              

hengck23 2026-06-03
🌆 
inbox%2F113660%2Ff10355233099636bce19b6866d936faf%2FSelection_3867.png?generatio
n=1780493446006634&alt=media                                                    

i show the same solution in two different visualisations. z prior is much       
stronger than gr prior.                                                         

hengck23 2026-06-03
eg, it is "easy" to correct this error?                                         

the initial offset before PS tells a lot about the distance between well z and  
geo z                                                                           

🌆 
inbox%2F113660%2F8bfe84f62101ba0e71d74fd91cc5e7aa%2FSelection_3868.png?generatio
n=1780493955235969&alt=media                                                    

hengck23 ▲1 2026-06-04
🌆 
inbox%2F113660%2Fd32d36628bf6945c3f5f3142b2eedbc7%2FSelection_3878.png?generatio
n=1780571032978090&alt=media                                                    

🌆 
inbox%2F113660%2F22b04f6875317dee0ec9cb30ce29bbc8%2FSelection_3879.png?generatio
n=1780571048071952&alt=media plot of tvt max - tvt min verus tvt length         

hengck23 2026-06-04
🌆 
inbox%2F113660%2Fa6fba4a916b4948c06263aa01ff80031%2FSelection_3880.png?generatio
n=1780571670031828&alt=media                                                    

🌆 
inbox%2F113660%2Fb822aac1c542535cfa09be1caedb99b0%2FSelection_3881.png?generatio
n=1780571688085476&alt=media                                                    

hengck23 ▲3 2026-06-06
🌆 
inbox%2F113660%2F6c066e98586dd0273891bb4c0e4ba460%2FSelection_3906.png?generatio
n=1780777215271692&alt=media                                                    

hengck23 ▲1 2026-06-07
CNN+SDF +MTP:                                                                   

 • top3 prob path and last one is mean of top4+5                                

for the first time, we have correct prediction inside the top 3. input is 512   
window of h tvt (compression=2) and 64 window of t tvt                          

                                                                                
                                                                                
         #add features-------------------------------------                     
         history  = (                                                           
             t_tvt.reshape(B, 1, 1, T).expand(B, 1, H, T)                       
             - h_tvt_history.reshape(B, 1, H, 1).expand(B, 1, H, T)             
         )                                                                      
         mask = h_tvt_mask.reshape(B, 1, H, 1).expand(B, 1, H, T)               
         history = history*mask                                                 
                                                                                
         #-----------------------------------------                             
                                                                                
         image = torch.concat([                                                 
             t_gr.reshape(B,1,1,T).expand(B,1,H,T),                             
             h_gr.reshape(B,1,H,1).expand(B,1,H,T),                             
             t_gr.reshape(B,1,1,T).expand(B,1,H,T)-h_gr.reshape(B,1,H,1).expand 
 ,1,H,T),                                                                       
             history,                                                           
             mask,                                                              
         ], dim=1)                                                              
         image   = self.norm(image)                                             
         feature = self.backbone(image)                                         
                                                                                

no x,y,z  information. so i am sure the CNN is learning something useful from   
the gr data. my transformer cannot learn from gr which i don't know why (i      
suspect it is a normalisation issue?)                                           

🌆 
inbox%2F113660%2Faae7b5453d76333fa66317e0385a568a%2FSelection_3909.png?generatio
n=1780816807102864&alt=media                                                    

hengck23 2026-06-07
results of training mixture                                                     

🌆 
inbox%2F113660%2Fa5f61ac8fef7ac912d93448075e76aab%2FSelection_3910.png?generatio
n=1780817322600678&alt=media                                                    

i think it has to be a mixture model because I can see the prediction hopping   
around a few modes                                                              

Tom 2026-06-07
They look more like some basis                                                  

hengck23 ▲2 2026-06-07
validation results for full length  tvt (by probing, all hidden test well has   
kength <12_000). h tvt  window = 384, h tvt window 768 (at compression =16)     

you can see  sdf is bending correctly, i.e. it indeed learned the steering. the 
magic is adding well dip tangent feature (sin and cos of dmd/dz) and well       
direction tagent feature (sin and cos of dx/dy, i.e. geology dip) + gr heatmap  

🌆 
inbox%2F113660%2F116b6d71ae10aa112ffa2cac3b9512a5%2FSelection_3916.png?generatio
n=1780844328779967&alt=media                                                    

hengck23 2026-06-07
@tom99763                                                                       

they are less primitive after plotting at the recovered TVT                     
sdf = t_tvt - h_tvt, hence h_tvt = t_tvt - sdf  for sdf.abs()<2                 

instead of generating more K per model, it is better to save a few models at    
different iterations and run them at inference                                  

🌆 
inbox%2F113660%2Fa6884624231284d7819468c3e3528620%2FSelection_3920.png?generatio
n=1780874906661462&alt=media                                                    

different iteration                                                             

🌆 
inbox%2F113660%2F27ab0fbe2d10d46e9040fd9efca4069a%2FSelection_3919.png?generatio
n=1780874926438715&alt=media                                                    

hengck23 ▲1 2026-06-08
🌆 
inbox%2F113660%2Fd6968b6a7459c0658a59f9d74cbbe43f%2FSelection_3931.png?generatio
n=1780913667286478&alt=media                                                    

i think rmse error is some how biased (e.g. increases with length due to error  
accumulation. post processing your results may help)                            

