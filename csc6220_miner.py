'''
Data for CSC6220 
Jan 05, 2019 
Akond Rahman 
'''
import pandas as pd 
import numpy as np 

import _pickle as pickle
import time
import datetime
import os 
import csv 
import sys
from git import Repo
import  subprocess
import time 
import  datetime 
from collections import Counter

def giveTimeStamp():
  tsObj = time.time()
  strToret = datetime.datetime.fromtimestamp(tsObj).strftime('%Y-%m-%d %H:%M:%S')
  return strToret

def getEligibleProjects(fileNameParam):
  repo_list = []
  with open(fileNameParam, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:
      repo_list.append(row[0])
  return repo_list

def getModFilesInDiff(diff_str):
    splitted_lines = diff_str.split('\n')
    mod_files = [x_.split('a')[-1] for x_ in splitted_lines if '---' in x_ ]
    return mod_files

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


def buildContent(df_, HOST_DIR):
    all_hash = np.unique( df_['HASH'].tolist() ) 
    for hash_ in all_hash: 
        hash_df = df_[df_['HASH']==hash_]
        repo_name = hash_df['REPO'].tolist()[0]
        secu_flag = hash_df['SECU_FLAG'].tolist()[0] 
        repo_path = HOST_DIR + repo_name + '/'
        diff_txt, file_list  = getDiff(repo_path, hash_) 
        print(file_list) 

if __name__=='__main__': 

    t1 = time.time()
    print('Started at:', giveTimeStamp() )
    print('*'*100 )

    CURATED_FILE = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Teaching/Spring2020/CSC6220-TNTECH/CURATED_SECURITY_FULL.csv' 
    HOST_DIR='/Users/arahman/TEACHING_REPOS/NON_JULIA_SCIENTIFIC_SOFTWARE/'

    CURATED_DF = pd.read_csv(CURATED_FILE) 
    buildContent(CURATED_DF)

    print('*'*100 )
    print('Ended at:', giveTimeStamp() )
    print('*'*100 )
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print('Duration: {} minutes'.format(time_diff) )
    print( '*'*100  )

