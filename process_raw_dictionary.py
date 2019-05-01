"""
Function process_data() takes a dictionary of source data and either output six processed lists (src_train,tgt_train,src_val,tgt_val,src_test,tgt_test) or write those lists to six files.


Nedd to change DATA_PATH before calling the function.
"""




DATA_PATH = '/scratch/ml5885/nlu/transformer/data/example/raw/'


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from itertools import combinations


def src_tgt_combination(df):
    perm_src = []
    perm_tgt = []
    perm = []
    c = 0
    for i in range(df.shape[0]):
        c+=1

        temp_list = [df.iloc[i]['s1']] + df.iloc[i]['s2']


        perm = list(combinations(temp_list,2))

        for ele in perm:
            perm_src.append(ele[0])
            perm_tgt.append(ele[1])
    return perm_src,perm_tgt


def remove_twitter_punc(sent_list):
    for i in range(len(sent_list)):
        if sent_list[i].find('\n') != -1:
            sent_list[i] = sent_list[i].replace("\n","")
        sent_list[i] = re.sub(r'(\s)@\w+', r'',sent_list[i])
        sent_list[i] = re.sub(r'(\s)#\w+', r'',sent_list[i])
    
    return sent_list


def write_file(file_to_write,path):
    with open(path,"w") as f:
        for i in range(len(file_to_write)):
            f.writelines(str(file_to_write[i])+'\n')
        

def process_data(raw,to_lower = False, write_files = False,output_path = DATA_PATH):
    
    '''
    Convert to lower case
    '''
    raw_lc = {}
    if to_lower == True:
        for k,v in raw.items():
            k = " ".join([t.lower() for t in k.split()])
            for i in range(len(v)):
                v[i] = " ".join([t.lower() for t in v[i].split()])
            raw_lc[k] = v
        print ("Done converting to lower cases.")
    else:
        raw_lc = raw
    
    '''
    Train/test split on key
    '''
    df = pd.Series(raw_lc,name='s2')
    df.index.name = 's1'
    df = df.reset_index()
    df = pd.DataFrame(df)
    train, tv = train_test_split(df, test_size=0.2, random_state=42)
    val,test = train_test_split(tv, test_size=0.5, random_state=42)
    
    print ("Done train/test split")
    
    '''
    Data combination
    '''
    src_train,tgt_train = src_tgt_combination(train)
    src_val,tgt_val = src_tgt_combination(val)
    src_test,tgt_test = src_tgt_combination(test)
    
    assert len(src_train) == len(tgt_train)
    print ("Done creating combinations")
    
    '''
    Remove \n and special puncs
    '''
    src_train = remove_twitter_punc(src_train)
    tgt_train = remove_twitter_punc(tgt_train)
    src_val = remove_twitter_punc(src_val)
    tgt_val = remove_twitter_punc(tgt_val)
    src_test = remove_twitter_punc(src_test)
    tgt_test = remove_twitter_punc(tgt_test)
    
    print ("Done removing unwanted puncs")
    
    '''
    Output Files
    If write_files == True, write. Else, return six lists.
    '''
    if write_files == True:
    
        write_file(src_train,output_path+'src-train.txt')
        write_file(tgt_train,output_path+'tgt-train.txt')
        write_file(src_val,output_path+'src-val.txt')
        write_file(tgt_val,output_path+'tgt-val.txt')
        write_file(src_test,output_path+'src-test.txt')
        write_file(tgt_test,output_path+'tgt-test.txt')
    
    else:

        return src_train,tgt_train,src_val,tgt_val,src_test,tgt_test