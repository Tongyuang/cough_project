import os
import pandas as pd
import pandas.errors
from collections import Counter

from config import config as c
from preprocess import get_min_samps

label_file_header = ['start', 'end', 'label']

MIN_SAMPS, HALF_MIN_SAMPS, MIN_TIME, HALF_MIN_TIME = get_min_samps()

def load_df_label_file(path):
    try:
        with open(path) as file:
            df = pd.read_csv(file, sep=r'\t', header=None, engine='python')
            df.columns = label_file_header
            return df
    except FileNotFoundError:
        print('ERROR - File:', path, 'could not be found')
        return False
    except pd.errors.EmptyDataError:
        print('ERROR - No labels found in label file:',path)
        return False

def space_to_tab():

    path = os.path.join(c['folder_raw'], '19-n_label.txt2')
    path_new = os.path.join(c['folder_raw'], '19-n.label.txt2')
    df = pd.read_csv(path, sep=r' ', header=None, engine='python')
    df.to_csv(path_new, sep='\t', index=False, header=False)

def fix_incorrect_labels():
    # files = [f for f in os.listdir(c['folder_raw']) if f.endswith('.label.txt2')]
    files = [f for f in os.listdir(c['folder_raw']) if "TDRS" in f and (f.endswith('.label.txt') and not 'gen' in f)]

    for f in files:
        # if not(f.startswith('19-n')):
        #     continue
        changed=True#False
        path = os.path.join(c['folder_raw'], f)
        print("starting", f)

        # load the label file
        df = load_df_label_file(path)
        if isinstance(df,bool):
            continue

        cnt_not_found=0
        for i in range(len(df.index)):
            row = df.iloc[i]
            label = row.label
            if label not in c['label_types']:
                cnt_not_found+=1
                if label in ["screaming", "crying", "yelling", "cry", "yell", "hiccup", "sigh", "laugh", "scream",
                             "laughter","sneeze","clearing throat","throat clearing","yawn","snort","whistle", "blowing nose"]:
                    df.at[i,'label'] = 'other'
                elif label in ["breathing"]:
                    df.at[i, 'label'] = 'breath'
                elif label in ["nosie","buzzing"]:
                    df.at[i, 'label'] = 'noise'
                elif label in ['tv',"TV",'Tv',None]:
                    df.at[i, 'label'] = 'silence'
                else:
                    print("label -",label,"- not found")
                changed=True
                # print("label changed from -",label, "- to -", df.at[i, 'label'])
                # print(row)
                # print(df.iloc[i])

        if changed:
            path_new = os.path.join(c['folder_raw'], f.split('.')[0] + '_new.label.txt2')
            df.to_csv(path_new, sep='\t', index=False,header=False)
            # print("saved",path_new)
            print("saved. cnt not found:",cnt_not_found)

def add_silence():

    files = [f for f in os.listdir(c['folder_raw']) if "TDRS" in f and f.endswith('_new.label.txt2')]

    for f in files:
        path = os.path.join(c['folder_raw'], f)
        # path_new = os.path.join(c['folder_raw'], f.split('.')[0]+'_sil.label.txt2')
        print("starting", f)

        df = load_df_label_file(path)
        if isinstance(df,bool):
            continue

        cntr = Counter()
        cntr_new = Counter()
        df_new = pd.DataFrame([], columns=label_file_header)
        end_last = 0
        for i in range(len(df.index)):
            row = df.iloc[i]
            start = row.start
            end = row.end
            label = row.label
            cntr[label] +=end-start

            # print(end-start)
            if (start - end_last) > .001:
                df_new = df_new.append(pd.DataFrame([[end_last, start,'silence']], columns=label_file_header), ignore_index=True)
            df_new = df_new.append(pd.DataFrame([[start,end,label]], columns=label_file_header), ignore_index=True)
            end_last = end

        #combine silences
        df_new_no_sil = pd.DataFrame([], columns=label_file_header)
        prev_silence = False
        for i in range(len(df_new.index)):
            row = df_new.iloc[i]
            start = row.start
            end = row.end
            label = row.label
            if label=='silence':
                if prev_silence == False:
                    prev_silence = [start, end]
                else:
                    prev_silence = [prev_silence[0],end]
            else:
                if prev_silence != False:
                    df_new_no_sil = df_new_no_sil.append(pd.DataFrame([[prev_silence[0], prev_silence[1], 'silence']], columns=label_file_header),ignore_index=True)
                    prev_silence = False
                df_new_no_sil = df_new_no_sil.append(pd.DataFrame([[start, end, label]], columns=label_file_header), ignore_index=True)

        for i in range(len(df_new.index)):
            row = df_new.iloc[i]
            start = row.start
            end = row.end
            label = row.label
            cntr_new[label] += end - start

        print(cntr)
        print(cntr_new)
        #save over old, can easily be regenerated
        df_new_no_sil.to_csv(path, sep='\t', index=False,header=False)


if __name__ == '__main__':
    # space_to_tab()
    fix_incorrect_labels()
    add_silence()