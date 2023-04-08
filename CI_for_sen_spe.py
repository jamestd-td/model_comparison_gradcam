from math import sqrt


# =============================================================================
#  Confidence Intervel for Sensitivity and Specificity
#  The Confidence Interval use Wilson Score interval formula to calculate 
#  the upper and lower bound
#  The arguments to pass are the following:
#
#  TP = True positive: Sick people correctly identified as sick
#  FP = False positive: Healthy people incorrectly identified as sick
#  TN = True negative: Healthy people correctly identified as healthy
#  FN = False negative: Sick people incorrectly identified as healthy
# =============================================================================

def confidence_interval_sen_spe(tp,tn,fp,fn):
      
    sen = tp/(tp+fn) # sensitivity
    spe = tn/(tn+fp) # specificiry
    n = tp+fn        # total number of sick individuals 
    n1 = tn+fp       # total number of healthy individuals 
    
    z=1.96           # Desired Confidence Interval 1.96 for 95%  
    
    adj_sen = (sen + (z*z)/(2*n))/(1 + ((z*z)/n)) # Adjusted sensitivity 
    adj_spe = (spe + (z*z)/(2*n1))/(1 + ((z*z)/n1)) # Adjusted specificity
      
    ul_sen=((sen + (z*z)/(2*n))+(z*(sqrt(((sen*(1-sen))/n) + ((z*z)/(4*(n*n)))))))/(1 + ((z*z)/n)) # Upper level sensitivity
    ll_sen=((sen + (z*z)/(2*n))-(z*(sqrt(((sen*(1-sen))/n) + ((z*z)/(4*(n*n)))))))/(1 + ((z*z)/n)) # Lower level sensitivity
    ul_spe=((spe + (z*z)/(2*n1))+(z*(sqrt(((spe*(1-spe))/n1) + ((z*z)/(4*(n1*n1)))))))/(1 + ((z*z)/n1)) # Upper level specificity
    ll_spe=((spe + (z*z)/(2*n1))-(z*(sqrt(((spe*(1-spe))/n1) + ((z*z)/(4*(n1*n1)))))))/(1 + ((z*z)/n1)) # Lower level specificity
    
    return (adj_sen,ll_sen,ul_sen,adj_spe,ll_spe,ul_spe)
