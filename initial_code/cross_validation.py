import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config
from config import config as c
from config import config_data as c_d

import progressbar

def get_typ_and_domain_probabilities_dicts():

    df_p = pd.read_excel(c['percentages_xlsx'], header=1, index_col=0)

    subtypes = df_p.subtype.unique()
    subtypes_dict = {}
    percentages_dict = {}

    dp_columns = [d for d in df_p.columns if d.endswith('_dp')]
    domains = np.array([d.split('_')[0] for d in dp_columns])
    for idx in df_p.index:
        percentages_dict[idx] = {}
        pcts = df_p.loc[idx, dp_columns].values
        idxs = np.argwhere(pcts).reshape(-1)
        percentages_dict[idx]['pcts'] = np.array(pcts[idxs], dtype=np.float64)
        percentages_dict[idx]['doms'] = domains[idxs]

    for s in subtypes:

        subtypes_dict[s] = {}
        df_s = df_p[df_p.subtype == s]
        subtypes_dict[s]['types'] = list(df_s.index.values)
        subtypes_dict[s]['pcts'] = list(df_s.Probability_mod.values)

    return(domains, subtypes_dict, percentages_dict)


def CV_for_audioset_coughs_and_flusense():

    #Give CV labels to audioset and flusense (assume coughsense, pediatric, etc already have CV labels)

    # This spreadsheet has coughsense, pediatric, etc already manually labeled for CV but still need audioset coughs
    # and flusense labeled
    df = pd.read_excel(c['files_cv_split'],index_col=2,converters={'name':str})

    # just get flusense and audioset samples
    df_f_a = df[df['domain'].isin(['flusense','audioset'])]

    # 9 or so files don't have name (weird conversion thing when went from csv to excel. we'll just give these all CV 0
    df_f_a = df_f_a[df_f_a.index.notna()]

    fnames = np.array(df_f_a.index.unique())
    kf = KFold(n_splits=4,shuffle=True)

    # Label with the cross_validation
    for i,(_, te_idx) in enumerate(kf.split(fnames)):
        fnames_cv = fnames[te_idx]
        for fname in fnames_cv:
            df.at[fname,'CV'] = i

    # we removed rows that don't have a valid index earlier, now just mark those as 0
    row_mask = df.index.isna()
    df.at[row_mask,'CV'] = 0

    #save to excel
    df.to_excel(c['files_cv_split'].replace('.xlsx','_new.xlsx'))

def get_CV_for_FSDKaggle_and_audioset_noncough():

    # For FSDKaggle and noncough audioset, the preprocessing is different so we need to do the cross validation differently
    columns = ['type','domain','name','cv']
    kf = KFold(n_splits=4, shuffle=True)

    if not os.path.exists(c['files_cv_split_fsd_audioset']):

        #create new one if none exists
        df = pd.DataFrame([],columns=columns)
    else:

        #otherwise load
        df = pd.read_csv(c['files_cv_split_fsd_audioset'])
        df.to_csv(c['files_cv_split_fsd_audioset'].replace('.csv','_old.csv'))

    for i, (root, dirs, files) in enumerate(os.walk(c['folder_wav'])):

        typ = os.path.basename(root)
        if 'FSDKaggle' in dirs or 'audioset' in dirs:

            # if typ == 'throat-clearing':
            #     print("Throat clear")

            for dom in ['FSDKaggle','audioset']:

                folder = os.path.join(root, dom)
                if not os.path.exists(folder) or (dom == 'audioset' and typ == 'cough'):
                    continue
                fnames = list(set([f.split('_')[1] for f in os.listdir(folder) if f.endswith('.wav')]))

                # Checks if there are any samples that already have a CV from another "typ" - this is fine, there aren't many
                # repeat_fnames = [f for f in fnames if f in df.name.values]
                # if repeat_fnames:
                #     print(typ,dom, "repeat fnames:", len(repeat_fnames), repeat_fnames)

                # Remove ones we've labeled previously
                fnames = [f for f in fnames if f not in df.name.values]

                if len(fnames)<1:
                    print("All files already have CV split for:",dom,typ)
                    continue

                df_new = np.empty((len(fnames),4),dtype=object)
                df_new[:,0] = typ
                df_new[:,1] = dom
                df_new[:,2] = fnames
                if len(fnames)<4:
                    df_new[:, 3] = 0
                else:
                    for i,(_, te_idx) in enumerate(kf.split(fnames)):
                        df_new[te_idx,3] = i


                df = df.append(pd.DataFrame(df_new,columns=columns),ignore_index=True)
                df.to_csv(c['files_cv_split_fsd_audioset'],index=False)
                print(len(fnames), "new files labeled for CV", typ, dom)

def get_file_list(CV, TEST=False):

    file_name = 'files_cv{}_{}.pkl'.format(CV,c['sr_str'])

    if TEST:
        with open(os.path.join(c_d['folder_data'], 'meta', file_name), 'rb') as handle:
            return(pickle.load(handle))

    files_tr = {}
    files_te = {}
    df_1 = pd.read_excel(c['files_cv_split'],index_col=0)
    df_2 = pd.read_csv(c['files_cv_split_fsd_audioset'],index_col=2)
    df_wav_folders = pd.read_csv(c['percentages_xlsx'].replace('.xlsx', '.csv'),index_col=0)

    df_1 = df_1['CV']
    df_2 = df_2['cv'].rename('CV')
    df = pd.concat((df_1, df_2))

    # df.index = df.index.replace('_','-')
    # print(df.index.values[:10])
    df = df.loc[df.index.dropna()]
    df.index = [f.replace('_','-') for f in df.index.values]

    not_found = []
    print("Getting tr/te file list")

    n = len(df_wav_folders.index.values)
    for i,typ in enumerate(df_wav_folders.index.values):

        typ_path = os.path.join(c['folder_wav'], typ)
        files_tr[typ] = {}
        files_te[typ] = {}

        for dom in os.listdir(typ_path):

            dom_path = os.path.join(typ_path, dom)

            files_tr[typ][dom] = []
            files_te[typ][dom] = []

            for f in os.listdir(dom_path):
                name = f.split('_')[1]

                # if we can't find it, just throw it in training
                if name not in df.index:
                    if name not in not_found:
                        not_found.append(name)
                        print("new not found:", f)
                    files_tr[typ][dom] += [f]
                    continue
                # if dom =='flusense':
                #     print("Found flusense")

                cv = df.loc[name]
                # if we accidentally have the file in there twice, just use the first one
                if isinstance(cv, pd.Series):
                    cv = cv.iloc[0]

                if cv == CV:
                    files_te[typ][dom] += [f]
                else:
                    files_tr[typ][dom] += [f]

        if i%10==0:
            print("Completed tr/te", i, "of", n)

        with open(os.path.join(c_d['folder_data'], 'meta', file_name), 'wb') as handle:
            pickle.dump([files_tr,files_te],handle)

    return(files_tr, files_te)

if __name__ == '__main__':

    CV=0
    # get_CV_for_FSDKaggle_and_audioset_noncough()
    files_tr, files_te = get_file_list(CV=CV)
