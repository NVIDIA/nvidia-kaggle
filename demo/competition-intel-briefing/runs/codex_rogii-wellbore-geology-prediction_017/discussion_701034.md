╭─ Surface columns (ANCC, ASTNU, etc.) are in TVD (Z), NOT in TVT ─────────────╮
│ Author: Nicolas Bridelance | Votes: 9 | Comments: 2 | Created: 2026-05-18    │
│ https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discus │
│ sion/701034                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

Hi everyone,                                                                    

I spent some time debugging a visualization issue that might trip up someone    
somewhere, so sharing it here.                                                  

                                   The issue                                    

The six geological surface columns in the horizontal well files:                

ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA                                          

are stored as negative TVD values (same unit as the Z column, typically ranging 
from -9500 to -7500 ft), not in TVT.                                            

If you try to plot them directly against TVT (which ranges from ~11000 to 12000 
ft), they'll appear completely off-scale and useless.                           

                                   Quick fix                                    

Map each surface value from Z-space to TVT-space using the well's own Z→TVT     
relationship. Since Z and TVT are almost perfectly correlated (|r| ≈ 0.999), a  
simple linear interpolation works well:                                         

                                                                                
 from scipy.interpolate import interp1d                                         
                                                                                
 hw_clean = hw.dropna(subset=['Z', 'TVT']).sort_values('Z')                     
 z_to_tvt = interp1d(hw_clean['Z'].values, hw_clean['TVT'].values,              
                     kind='linear', bounds_error=False, fill_value='extrapolate 
                                                                                
 surface_tvt = float(z_to_tvt(hw['ANCC'].dropna().iloc[0]))                     
                                                                                

Why it matters for modeling Using raw Z values as if they were TVT will         
introduce an error of ~20,000 ft (the offset between negative TVD and positive  
TVT).                                                                           

Hope this saves someone a few hours! 🙂                                         

── 2 Comments ──

shanzhong8 2026-05-19
Does this preprocessing strategy result in information leakage?                 

Nicolas Bridelance 2026-05-21
Good question. The mapping doesn't introduce leakage because:                   

Z (TVD) is fully provided for all wells including test — it's the borehole      
trajectory, independent of the TVT target The interpolator uses Z ↔ TVT_input   
pairs from the known (non-masked) portion of the well The geological surface    
depths (ANCC etc.) are formation tops in the vertical section, well within the  
range covered by TVT_input The transformation is purely geometric (coordinate   
conversion), not predictive. The TVT values you're actually predicting are at   
intermediate MD positions along the lateral, not at the surface depths.         

