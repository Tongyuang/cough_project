import pandas as pd
import json
import config
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import os
# create and a dictionary based on config file

SOUND_DICT = {
'cough':['cough'],
'easy':['bark','bus','chime','computer-keyboard','gong','keys-jangling','meow','scissors','writing'],
'hard':['bass-drum','cowbell','double-bass','drawer-open-or-close','gunshot-or-gunfire','hi-hat','shatter','snare-drum','tearing'],
'human-misc':['baby-cry,-infant-cry','burping-or-eructation','fart','snore','squeak','vomit'],
'instrument':['instrument'],
'kitchen':['cutlery,-silverware','dishes,-pots,-and-pans'],
'laugh':['baby-laughter','belly-laugh','chuckle,-chortle','giggle','laughter','snicker'],
'medium':['applause','finger-snapping','fireworks','knock','microwave-oven','telephone'],
'music':['music'],
'noise':['noise'],
'other':['other'],
'respiratory':['breath','gasp','hiccup','sneeze','sniff','throat-clearing','wheeze'],
'silence':['etc','silence'],
'speech':['speech'],
'unknown':['unknown']
}

output_dict0 = {(subtype,0):[] for subtype in SOUND_DICT.keys()}
output_dict1 = {(subtype,1):[] for subtype in SOUND_DICT.keys()}
output_dict2 = {(subtype,2):[] for subtype in SOUND_DICT.keys()}
output_dict3 = {(subtype,3):[] for subtype in SOUND_DICT.keys()}

subtype_cv_sample_dict = {}
subtype_cv_sample_dict.update(output_dict0)
subtype_cv_sample_dict.update(output_dict1)
subtype_cv_sample_dict.update(output_dict2)
subtype_cv_sample_dict.update(output_dict3)

silence_dict = {}
subtypes_list = list(SOUND_DICT.keys())

df = pd.read_csv('../data/all_files.csv')
silence_ampli=0.08
silence_range=8000

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

for i in tqdm(range(df.shape[0])):
    
    # build_subtype_sample_dict
    current_file = df.iloc[i]

    if current_file['domain'] in ['flusense','pediatric']:
        continue
    find_flag = False # used to jump out of loop

    for subtype in subtypes_list:
        for sub_subtype in SOUND_DICT[subtype]:
            if current_file[sub_subtype]>0:
                CV = current_file['CV']
                subtype_cv_sample_dict[(subtype,CV)].append((current_file['domain'],current_file['name-long']))
                find_flag = True
                break
        if find_flag:
            break
    
    # build silence dict
    domain_path = 'wav_dur_{}_{}'.format(config.sr_str,config.domain_name_dict[current_file['domain']])
    wav_path = config.config_data['folder_data']+'/'+domain_path+'/'+current_file['name-long'] +'.wav'

    try:
        _,wav =  wavfile.read(wav_path,'w')
        wav = np.array(wav)
    except:
        print('could not get file:',wav_path)
        continue

    silence_list = []
    start_flag = 0
    end_flag = 0

    for i in range(wav.shape[0]):
        if np.abs(wav[i]) < silence_ampli:
            if start_flag<=end_flag:
                end_flag = i
            else:
                start_flag = i
                end_flag = start_flag
        else:
            if start_flag >= end_flag:
                continue
            else:
                    # if silence range
                if end_flag-start_flag>silence_range:
                    silence_list.append((start_flag,end_flag))
                start_flag = i
                end_flag=start_flag
    
    silence_dict[current_file['name-long']] = silence_list
    
print('dict built')

str_dict = {str(k):subtype_cv_sample_dict[k] for k in subtype_cv_sample_dict}
subtype_cv_sample_dict_json = json.dumps(str_dict)

silence_dict_json = json.dumps(silence_dict)

output_name = '../data/subtype_cv_sample_dict.json'

with open(output_name,'w') as f:
    f.write(subtype_cv_sample_dict_json)
    f.close()

print('saved dict as .json file in {}'.format(output_name))

output_name = '../data/silence_dict_json.json'

with open(output_name,'w') as f:
    f.write(silence_dict_json)
    f.close()

print('saved dict as .json file in {}'.format(output_name))
