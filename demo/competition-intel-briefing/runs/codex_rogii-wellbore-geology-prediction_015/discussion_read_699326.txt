╭─ stage.1 : global search using linear prior tvt = linear(md,z) ──────────────╮
│ Author: hengck23 | Votes: 18 | Comments: 5 | Created: 2026-05-13             │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/699326                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

🌆 
inbox%2F113660%2F1aabc18ee91163e6ba228c5e1db9b147%2FSelection_3505.png?generatio
n=1778687376098048&alt=media                                                    

Next post: stage.2 iterative local search for refinement                        

── 5 Comments ──

hengck23 ▲1 2026-05-13
🌆 
inbox%2F113660%2Fc56ecec1fbf2e65293ed60b919a8d170%2FSelection_3507.png?generatio
n=1778691758365065&alt=media                                                    

initalise with fitted line of md,z after PS and also using tvt_input. need to   
think of  a way to make it "smooth"                                             

hengck23 ▲1 2026-05-13
🌆 
inbox%2F113660%2F56d68326c7bf5fb5347999c5889b0513%2FSelection_3508.png?generatio
n=1778692190154785&alt=media                                                    

🌆 
inbox%2F113660%2F65ed9281146bfcc077582dd4d4e54495%2FSelection_3509.png?generatio
n=1778692204746245&alt=media                                                    

🌆 
inbox%2F113660%2F528e2db812b1dec908fe6de6e42c9e74%2FSelection_3513.png?generatio
n=1778693089162977&alt=media                                                    

top to bottom: typewell TVT-GR after PS, typewell TVT-GS, horizontal            
MD-smoothedGS showing forward and reverse, horizontal MD-smoothedGS showing TVT 
as color, horizontal TVT-smoothedGS                                             

hengck23 ▲3 2026-05-13
🌆 
inbox%2F113660%2Fdd4117e1392d8e76e50fec03ec6111df%2FSelection_3512.png?generatio
n=1778692902303316&alt=media 🌆 
inbox%2F113660%2F3909dbc9cfac77444c5d3e6acfabc790%2FSelection_3511.png?generatio
n=1778692880943488&alt=media gemini suggested this but i haven't tried:         

hengck23 ▲4 2026-05-14
this is why prior (constraints) is important.                                   

lower GR fitting doesn't mean lower T ** it is an inverse problem ! **          

You should learn the 2-parameter prior space using TVT RMSE, not GR RMSE.       

🌆 
inbox%2F113660%2F862c8e7dcb7c2c0594d4c769da63c3ca%2FSelection_3518.png?generatio
n=1778728817525025&alt=media VT.                                                

Franklin Gois 2026-06-03
@hengck23 Thank you!                                                            

