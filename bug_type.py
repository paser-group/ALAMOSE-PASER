'''
Get security bug types 
Akond Rahman 
Sep 20, 2019 
'''
import pandas as pd 
import os 
import numpy as np 
import time
import datetime
from git import Repo
import  subprocess

def getBranchName(proj_):
    branch_name = ''
    proj_branch = {'biemond@biemond-oradb':'puppet4_3_data', 'derekmolloy@exploringBB':'version2', 'exploringBB':'version2', 
                   'jippi@puppet-php':'php7.0', 'maxchk@puppet-varnish':'develop', 'threetreeslight@my-boxen':'mine', 
                   'puppet':'production', 'deepvariant':'r0.8', 'galaxy':'dev', 'mdanalysis':'develop', 'pyGeno':'bloody',
                   'miso-lims':'develop' 
                  } 
    if proj_ in proj_branch:
        branch_name = proj_branch[proj_]
    else: 
        branch_name = 'master'
    return branch_name

def getSecMsg(single_commit, repo_name): 
    str_txt = ''
    repo_dir_absolute_path = '/Users/akond/TeachingRepos/' + repo_name + '/'
    print 'Started>' + repo_dir_absolute_path + '<' 
    repo_  = Repo(repo_dir_absolute_path)
    branchName = getBranchName(repo_name)
    all_commits = list(repo_.iter_commits(branchName))
    for commit_ in all_commits: 
        if commit_.hexsha==single_commit:
            msg_commit =  commit_.message 
            msg_commit = msg_commit.replace('\r', ' ')
            msg_commit = msg_commit.replace('\n', ' ')
            msg_commit = msg_commit.replace(',',  ';')    
            msg_commit = msg_commit.replace('\t', ' ')
            msg_commit = msg_commit.replace('&',  ';')  
            msg_commit = msg_commit.replace('#',  ' ')
            msg_commit = msg_commit.replace('=',  ' ')      
            msg_commit = msg_commit.lower()
            # msg_commit = msg_commit.encode('utf-8').strip() 
            msg_commit = msg_commit.encode('ascii', 'ignore').decode('ascii')
            str_txt    =  repo_name + ',' + single_commit + ',' + msg_commit + '\n'
            # print str_txt 
            # print commit_
    return str_txt
    
def dumpContentIntoFile(strP, fileP):
    fileToWrite = open( fileP, 'w')
    fileToWrite.write(strP )
    fileToWrite.close()
    return str(os.stat(fileP).st_size)

if __name__=='__main__':
    print '*'*100

    csc4220data  = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/TO_DETECT_SECURITY_BUGTYPE.csv'
    output_file  = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/CSC4220_SECURITY_BUGTYPE.csv'
    full_dataset = pd.read_csv(csc4220data)  

    all_commits =  np.unique( full_dataset['HASH'].tolist() )
    str_ = ''
    for commit_ in all_commits:
        repo = full_dataset[full_dataset['HASH']==commit_]['REPO'].tolist()[0]
        sec_msg = getSecMsg(commit_, repo)
        # print sec_msg 
        str_ = str_ + sec_msg 
    dumpContentIntoFile(str_, output_file ) 
    print '*'*100    
    