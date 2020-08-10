import pandas as pd
import json

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


subtypes_list = list(SOUND_DICT.keys())

df = pd.read_csv('../data/all_files.csv')

for i in range(df.shape[0]):
    
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
            
print('dict built')

str_dict = {str(k):subtype_cv_sample_dict[k] for k in subtype_cv_sample_dict}
subtype_cv_sample_dict_json = json.dumps(str_dict)

output_name = '../data/subtype_cv_sample_dict.json'

with open(output_name,'w') as f:
    f.write(subtype_cv_sample_dict_json)
    f.close()

print('saved dict as .json file in {}'.format(output_name))
