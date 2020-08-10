import os
import numpy as np
import pandas as pd
import progressbar

from config import config as c

folder = c['folder_wav']

df_all_files = pd.read_csv(c['files_csv'])
domains = list(df_all_files.domain.unique())

for root, typs, files in os.walk(folder):
    if os.path.basename(root) == '{}khz'.format(int(c['sr']/1000)):
        break
df_new = pd.DataFrame([], columns=domains)
df_new.index.name = 'fname'
cnts = np.zeros((len(typs), len(domains)))

bar = progressbar.ProgressBar(maxval=len(typs)).start()
for i, typ in enumerate(typs):
    path_typ = os.path.join(folder, typ)
    for root, dirs, files in os.walk(path_typ):
        bname = os.path.basename(root)
        if bname == typ:
            continue
        cnts[i, domains.index(bname)] = len(files)
    bar.update(i)

pd.DataFrame(cnts, columns=domains, index=typs).to_csv(r'Z:\research\cough_count\data\meta\wav_folders_new.csv')
