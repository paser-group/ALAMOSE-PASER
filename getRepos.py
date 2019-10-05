'''
Akond Rahman 
Aug 28, 2018 
Download scientific repos from Github 
'''
from itertools import izip_longest
import os 
import csv 
import subprocess
import numpy as np
import shutil
from collections import Counter 
import pandas as pd 

def getRepos(file_name):
    list_ = []
    with open(file_name, 'rU') as file_:
      reader_ = csv.reader(file_)
      next(reader_, None)
      for row_ in reader_:
          repo_dload_url = row_[0]
          list_.append(repo_dload_url)
    return list_

def cloneRepo(repo_name):
    cmd_ = "git clone " + repo_name
    try:
       subprocess.check_output(['bash','-c', cmd_])    
    except subprocess.CalledProcessError:
       print 'Skipping this repo ... trouble cloning repo', repo_name 

def printFileDist(directory):
    file_list = []
    for root_, dirnames, filenames in os.walk(directory):
        for file_ in filenames:
               file_list.append(os.path.join(root_, file_))
    ext_list = [ os.path.splitext(x_)[-1] for x_ in file_list ] # os.path.splitext('/path/to/somefile.ext') returns a tuple 
    ext_dict = dict(Counter(ext_list)) 
    print ext_dict 
    

def cloneRepos(repo_list):
    counter = 0     
    for repo_ in repo_list:
            counter += 1 
            print 'Cloning ', repo_
            cloneRepo(repo_)
            dirName = repo_.split('/')[-1].split('.')[0]
            print dirName
            printFileDist(dirName)
            ### get file count 
            all_fil_cnt = sum([len(files) for r_, d_, files in os.walk(dirName)])
            print '*'*50
            print "So far we have processed {} repos".format(counter)
            print '*'*50
               
def getRepoTypeDict(df_):
    dic = {'cyvcf2':'ComputationalBiology', 'deep-review':'ComputationalBiology', 'juicer':'ComputationalBiology',
           'photutils':'Astronomy', 'Roary':'ComputationalBiology', 'hap.py':'ComputationalBiology'
          }
    repos = np.unique(df_['REPO'].tolist())
    for repo_ in repos:
        repo_df = df_[df_['REPO']==repo_]
        type_ = np.unique(repo_df['REPO_TYPE'].tolist())[0]
        if repo_ not in dic: 
            dic[repo_] = type_ 
    return dic 

def assignType(single_val, repo_dict):
    if single_val in repo_dict:
        return repo_dict[single_val] 
    else:
        return 'ComputationalBiology'
 


if __name__=='__main__':
#    srcFile='/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/DownloadedRepos/repo_list.csv'
#    theList=getRepos(srcFile)      
#    list_ = np.unique(theList)
#    print 'Repos to download:', len(list_)
#    #cloneRepos(list_)

    full_dataset = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/Old-ProjectUpdate2-Dataset.csv'
    type_dataset = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/LOCKED_FINAL_SEC_BUG_TYPE.csv'
    csv_out_file = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/ProjectUpdate2-Dataset.csv'
    without_type_dataset = pd.read_csv(full_dataset) 
    type_dataset = pd.read_csv(type_dataset) 
    repo_type_dict = getRepoTypeDict(type_dataset)
    without_type_dataset['REPO_TYPE'] = without_type_dataset['REPO'].apply(assignType, args=(repo_type_dict,) ) 

    without_type_dataset.to_csv(csv_out_file, index=False, encoding='utf-8')

