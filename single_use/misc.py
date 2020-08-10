import os

def convert_whosecough_speech_names():
    folder = r'Z:\research\cough_count\data\wav\8khz\speech\whosecough'

    files = [f for f in os.listdir(folder)]

    for f in files:

        f_new = f.split('_')
        if 'speech' in f_new[1]:
            continue
        f_new[1] = f_new[1].replace('speec', 'speech')
        f_new = '_'.join(f_new)
        os.rename(os.path.join(folder, f), os.path.join(folder, f_new))

if __name__ == '__main__':
    pass