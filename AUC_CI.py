from math import sqrt

# N1= True Positive N2= True Neg in samples

def ci(auc, n1, n2):
    AUC = auc
    N1 = n1
    N2 = n2
    Q1 = AUC / (2 - AUC)
    Q2 = 2*(AUC*AUC) / (1 + AUC)
    SE_AUC = sqrt((((AUC*(1 - AUC)) + ((N1 - 1)*(Q1 - AUC*AUC)) + ((N2 - 1)*(Q2 - AUC*AUC)))) / (N1*N2))
    lower = AUC - 1.959*SE_AUC
    upper = AUC + 1.959*SE_AUC
    return (lower, upper)
