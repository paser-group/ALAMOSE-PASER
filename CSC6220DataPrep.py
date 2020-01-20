'''
Datasets for CSC 6220 
Akond Rahman 
Jan 20, 2020
'''
import pandas as pd 
import csv 
import os 
import numpy as np 
import  subprocess
import time 
import  datetime 
from collections import Counter
import _pickle as pickle

def getDiff(repo_, hash_):
    mod_files_list = []
   
    cdCommand   = "cd " + repo_ + " ; "
   
    diffCommand = "git diff " + hash_ + "~" + " " + hash_
    command2Run = cdCommand + diffCommand
    try:
      diff_output = subprocess.check_output(["bash", "-c", command2Run])
      mod_files_list =  getModFilesInDiff(diff_output)
    except subprocess.CalledProcessError as e_:
      diff_output = "NOT_FOUND" 
    return diff_output, mod_files_list

def getModFilesInDiff(diff_str):
    mod_files  = []
    splitted_lines = diff_str.split('\n')
    for x_ in splitted_lines:
      if '/' in x_  and '.' in x_: 
        if '---' in x_:
          del_ =  x_.split(' ')[-1].replace('a', '')
          mod_files.append(del_) 
        elif '+++' in x_:
          add_ =  x_.split(' ')[-1].replace('b', '')
          mod_files.append(add_)
    mod_files = np.unique(mod_files) 
    mod_files = [x_ for x_ in mod_files if '.jl' in x_ ]
    return mod_files

def getModFiles(repo_, hash_):
    mod_files_list = []
   
    cdCommand   = "cd " + repo_ + " ; "
   
    diffCommand = "git diff " + hash_ + "~" + " " + hash_
    command2Run = cdCommand + diffCommand
    try:
      diff_output = subprocess.check_output(["bash", "-c", command2Run])
      diff_output =  diff_output.decode('latin-1') ### exception for utf-8 
      # print(diff_output) 
      mod_files_list =  getModFilesInDiff(diff_output)
    except subprocess.CalledProcessError as e_:
      diff_output = "NOT_FOUND" 
    return  mod_files_list

def getFileLabels(df_param, root_path, out_file):
    file_dict = {}
    repo_dict = {}
    hash_ls   = np.unique( df_param['HASH'].tolist()  )
    for hash_ in hash_ls:
        hash_df   = df_param[df_param['HASH']==hash_]
        repo_name = hash_df['REPO'].tolist()[0] 
        hash_flag = hash_df['SECU_FLAG'].tolist()[0] 
        repo_path = root_path + repo_name 
        mod_files = getModFiles(repo_path, hash_) 
        for mod_file in mod_files: 
            mod_file_path = repo_path + mod_file 
            if mod_file_path not in repo_dict:
                repo_dict[mod_file_path] = repo_path        
            if mod_file_path not in file_dict:
                file_dict[mod_file_path] = [hash_flag] 
            else: 
                file_dict[mod_file_path] = file_dict[mod_file_path] + [hash_flag]
    needed_file_content = []
    for k_, v_ in file_dict.items(): 
        try:
            file_content = '' 
            if(os.path.exists(k_)) and (os.path.isdir(k_) == False):  
                num_lines     = sum(1 for line in open( k_ , 'r',  encoding='latin-1' ))
                repo_path     = repo_dict[k_] 
                with open(k_, 'r') as file_:
                    file_content  = file_.read() 
                file_flag     = '0'
                if (len (np.unique(v_)) == 1 ) and (np.unique(v_)[0]=='NEUTRAL'): 
                    file_flag = '0'
                else: 
                    file_flag = '1'
                needed_file_content.append( (k_, repo_path, file_content, num_lines , file_flag) ) 
        except ValueError as e_:
              print(e_) 
              print(k_)      
    print(needed_file_content[10:20])
    print(len(needed_file_content)) 
    with open(out_file, 'wb') as f_:
        pickle.dump(needed_file_content , f_ )
    
    
    
if __name__=='__main__':
    root_path = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Research/Insure/project_repos/' 
    labeled_dataset = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Research/Insure/Datasets/UNIQUE_BUG_COMM.csv'
    labeled_df  = pd.read_csv(labeled_dataset) 
    out_file = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Teaching/Spring2020/CSC6220-TNTECH/JULIA_SECURITY_FILE_LABELS.PKL'
    getFileLabels(labeled_df, root_path, out_file)  