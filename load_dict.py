import json

subtype_cv_sample_dict_path='../data/subtype_cv_sample_dict.json'

silence_dict_path='../data/silence_dict_json.json'

def str2tuple(str_in):
    '''
    when saving dict as .json file, tuple-like keys have been 
    switched into string-like keys, so this function transforms
    string back to tuple

    input:"('xxxxxx',0)"
    '''
    CV = int(str_in[-2:-1])
    name = str_in[2:-5]
    return (name,CV)

def listlist2tuplelist(list_in):
    '''
    still, when saving dict as .json file, tuple-like values have been 
    switched into list-like values, so this function transforms
    string back to tuple-list

    input: [[a,b],[c,d],[e,f]]
    is the list of list
    output: [(a,b),(c,d),(e,f)]
    is the list of tuple
    that's why this function's name is listlist2tuplelist
    '''
    output_list = [] 
    for sub_list in list_in:
        output_list.append((sub_list[0],sub_list[1]))
    return output_list

def load_dict(json_dict_path=subtype_cv_sample_dict_path):
    with open(json_dict_path,'r') as json_dict:
        subtype_cv_sample_dict = json.load(json_dict)

    key_list = list(subtype_cv_sample_dict.keys())
    # rebuild the original dict
    origin_dict = {}
    for key in key_list:
        origin_dict[str2tuple(key)] = listlist2tuplelist(subtype_cv_sample_dict[key])

    with open(silence_dict_path,'r') as silence_dict_json:
        silence_dict = json.load(silence_dict_json)

    return (origin_dict,silence_dict)


if __name__ == "__main__":
    with open(silence_dict_path,'r') as silence_dict_json:
        silence_dict = json.load(silence_dict_json)
    
    #for key in list(silence_dict.keys()):
    #   if len(silence_dict[key]) > 1:
    #       print(key)
    #       break
    print(silence_dict['wav_audioset_--gqmSD5paQ_0'])