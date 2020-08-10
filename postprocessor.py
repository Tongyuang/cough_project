import numpy as np

def metrics(lbl,pred):
    '''
    calculate metrics
    acc = correct num / total num
    prec = TP/(TP+FP)
    TNR = TN/(TN+FP) 
    TPR = TP/(TP+FN)
    F1_Score = 2*TP/(2*TP+FP+FN)

    '''
    assert lbl.shape == pred.shape

    TP = FP = TN = FN = 0
    for i in range(lbl.shape[0]):
        y1 = lbl[i]
        y2 = pred[i]
        if y1==0 and y2==0:
            TN += 1
        elif y1==0 and y2==1:
            FP += 1
        elif y1==1 and y2==0:
            FN += 1
        elif y1==1 and y2==1:
            TP += 1
    acc = (TP+TN)/(TP+FP+TN+FN)
    prec = 0 if TP==0 else TP/(TP+FP) 
    TNR = 0 if TN==0 else TN/(TN+FP) 
    TPR = 0 if TP==0 else TP/(TP+FN)
    F1_Score = 2*TP/(2*TP+FP+FN)
    return (acc,prec,TPR,TNR,F1_Score)

def normalization(lbl,yrange=(0,1)):
    '''
    normalization into yrange
    para lbl: input lbl
    '''
    assert len(lbl.shape)<=2
    x_max = np.max(lbl)
    x_min = np.min(lbl)

    y_max = yrange[1]
    y_min = yrange[0]

    k = (y_max-y_min)/(x_max-x_min) if x_max>x_min else 0
    b = -k*x_min+y_min

    y = k*lbl+b

    return y

def binary(lbl,gate=0.5):
    y = normalization(lbl)
    y[y>=gate]=1
    y[y<gate]=0
    return y