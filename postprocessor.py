import numpy as np
import matplotlib.pyplot as plt
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

def visualize_duration_error(file_dir='./evaluations/duration_error.txt',num_columns=50,x_range=(-5000,5000),skip_zero_pred=True):
    '''
    visualize the duration error using histogram

    para file_dir: dir of duration error file, see calculate_error() in evaluate.py
    para num_columns: number of columns in the histogram
    para x_range: x range
    '''

    try:
        f = open(file_dir,'r')
    except:
        raise Exception('could not open file:{}'.format(file_dir))

    # by domain
    error_list_dict = {'audioset':[],'coughsense':[],'southafrica':[],'jotform':[],'whosecough':[]}
    error_list_dict_by_percent = {'audioset':[],'coughsense':[],'southafrica':[],'jotform':[],'whosecough':[]}
    #error_list = []
    # all the errors
    for line in f.readlines():
        line_list = line.split(',')
        error = int(line_list[len(line_list)-1])
        domain = line_list[0]
        # append
        if skip_zero_pred:
            pred = int(line_list[len(line_list)-2])
            if pred == 0:
                continue

        error_list_dict[domain].append(error)
        # by percent
        ground_truth = int(line_list[len(line_list)-3])*10
        if ground_truth > 0:
            error_percent = float(float(error) / float(ground_truth))
            error_list_dict_by_percent[domain].append(error_percent)
    
    interval = 2*int((x_range[1]-x_range[0])/num_columns)
    x_ticks = np.arange(x_range[0],x_range[1]+interval,interval)

    for domain in list(error_list_dict.keys()):

        error_list = error_list_dict[domain]
        mean_ = np.mean(np.abs(np.asarray(error_list)))
        median_ = np.median(np.asarray(error_list))
        plt.hist(x=error_list, bins=num_columns, range=x_range, color='steelblue',edgecolor='black',normed=True)

        plt.xlabel('time duration error (ms), truth-pred')
    
        plt.ylabel('frequency')
        plt.xticks(x_ticks)

        plt.title('domain:%s,mean_of_abs_error:%d,median_error:%d'%(domain,mean_,median_))

        plt.show()
    
        error_list = error_list_dict_by_percent[domain]
        mean_ = np.mean(np.abs(np.asarray(error_list)))
        median_ = np.median(np.asarray(error_list))

        range_ = (-1,1)
        num_columns_ = 20
        interval = (range_[1]-range_[0])/num_columns
        #ticks_ = np.arange(range_[0],range_[1]+interval,interval)
        plt.hist(x=error_list, bins=num_columns_, range=range_, color='steelblue',edgecolor='black',normed=True)

        plt.xlabel('time duration percentage error (%), (truth-pred)/truth')
    
        plt.ylabel('frequency')
        #plt.xticks(ticks_)

        plt.title('domain:%s,mean_of_abs_percent_error:%d,median_percent_error:%d'%(domain,mean_,median_))

        plt.show()




if __name__ == "__main__":
    visualize_duration_error()
    