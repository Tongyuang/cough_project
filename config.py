import os


config_data = {'folder_data': r'../data'}


all_files_path = '../data/all_files.csv'
sr = 16000
sr_str = '{}khz'.format(int(sr/1000))
duration_t = .01 #time in bewteen labels for duration prediction task, in seconds
batch_num_per_epoch=100

llf_max_pool = 10
llf_stride_1 = 16
llf_stride_2 = 2
llf_output_frames = 48
# max samps for 96 x 64 output of llf #int(2*3**8 * sr/16000)
duration = 5
MAX_SAMPS = int(sr*duration) #5 seconds
OUTPUT_LENGTH = 5 # output size after preprocessor is 4.86 secs

#print("MAX_SAMPS", MAX_SAMPS)

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


config = {
'folder_raw':os.path.join(config_data['folder_data'], 'raw'),
'folder_wav':os.path.join(config_data['folder_data'], 'wav_duration', sr_str),
'folder_wavfile':os.path.join(config_data['folder_data'], 'wavfile'),
'folder_aug':os.path.join(config_data['folder_data'], 'aug_duration', sr_str),
'folder_meta':os.path.join(config_data['folder_data'], 'meta_duration', sr_str),
'llf_pretrain_dict': os.path.join(config_data['folder_data'], 'llf_pretrain_dict.pkl'),
'sr':sr,
'sr_str':sr_str,
'label_types':['cough', 'silence', 'speech', 'noise', 'other', 'breath', 'sniff', 'sound'], #'rejected',
'label_convert':{'breathe':'breath', 'burp':'burping-or-eructation', 'sniffle': 'sniff'},
'domains':['coughsense','pediatric','south_africa','whosecough','respeaker4mic','FSDKaggle', 'audioset','madagascar','real-world'],
'domains_devices':['coughsense','pediatric','south_africa','whosecough_sp','whosecough_rp','whosecough_bm','whosecough_rs1',
                   'whosecough_rs2','respeaker4mic','FSDKaggle','audioset','madagascar','real-world'],
'batch_size':32,
'duration_t':duration_t,
'duration_samps':int(duration_t*sr),
'duration':duration,
'max_samps':MAX_SAMPS,
'output_length':OUTPUT_LENGTH,
'all_files_path':all_files_path,
'batch_num_per_epoch':batch_num_per_epoch
}

samps_per_subtype_32 = {
'cough':20,
#'easy':1,
'hard':4,
'human-misc':1,
'instrument':1,
'kitchen':1,
'laugh':1,
#'medium':1,
'music':1,
'noise':1,
#'other':2,
'respiratory':1,
#'silence':1,
'speech':1,
} # 32

samps_per_subtype_16 = {
'cough':7,
'easy':1,
'hard':1,
'human-misc':1,
'kitchen':1,
'medium':1,
'noise':1,
'respiratory':1,
#'silence':1,
'speech':2,
} # 16

samps_per_subtype_coughsense = {
'cough':10,
'easy':1,
'hard':4,
'human-misc':2,
'instrument':1,
'kitchen':2,
'laugh':1,
'medium':1,
'music':2,
'noise':1,
'other':2,
'respiratory':2,
#'silence':1,
'speech':3,
}

domain_name_dict = {
'audioset':'audioset',
'coughsense':'cs',
'flusense':'flusense',
'FSDKaggle':'fsd',
'jotform':'jotform',
'pediatric':'ped',
'whosecough':'whosecough',
'southafrica':'southafrica'
}

#if sum([v for k, v in samps_per_subtype.items()]) != config['batch_size']:
 #   raise ("Incorrect samps per subtype")

#GETTING TENSORBOARD TO WORK ON AREA 51
#'ssh -N -L 16006:127.0.0.1:6006 remote_sever -i path_to_priv_key'
#For example:
#'ssh -N -L 16006:127.0.0.1:6006 mattw@area51.cs.washington.edu -i C:\Users\mattw12\.ssh\mattw12.ppk'

###Area 51 Gpu #s

# #os.evision(0) -> GPU1
# #os.evision(1) -> GPU2
# #os.evision(2) -> GPU0
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"