import pandas as pd

def list_to_csv(input_list,output_dir='output.csv'):
    '''
    write the training results to a .csv file
    '''

    columns = ('epoch','loss','accuracy','precision','recall','F1')
    df = pd.DataFrame(columns=columns)
    for (i,sub_list) in enumerate(input_list):
        subsublist = sub_list[1]
        epoch = sub_list[0]
        s = {'epoch':epoch*10, 
        'loss':subsublist[0],
        'accuracy':subsublist[1],
        'precision':subsublist[2],
        'recall':subsublist[3],
        'F1':subsublist[4]}
        df.loc[i] = s

    # to csv
    df.to_csv(output_dir)
    return df
if __name__ == "__main__":
    pass