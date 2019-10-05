'''
Akond Rahman 
Aug 28, 2019 : Wednesday  
Script to mine commits from scientific software repos 
'''


import pandas as pd 
import cPickle as pickle
import time
import datetime
import os 
import csv 
import numpy as np
import sys
from git import Repo
import  subprocess
import time 
import  datetime 
from collections import Counter


secu_kws = [ 'race', 'racy', 'buffer', 'overflow', 'stack', 'integer', 'signedness', 'widthness', 'underflow',
             'improper', 'unauthenticated', 'gain access', 'permission', 'cross site', 'CSS', 'XSS', 'denial service', 
             'DOS', 'crash', 'deadlock', 'SQL', 'SQLI', 'injection', 'format', 'string', 'printf', 'scanf', 
             'cross site', 'request forgery', 'CSRF', 'XSRF', 'forged', 
             'security', 'vulnerability', 'vulnerable', 'hole', 'exploit', 'attack', 'bypass', 'backdoor', 
             'threat', 'expose', 'breach', 'violate', 'fatal', 'blacklist', 'overrun', 'insecure'
           ]

prem_bug_kw_list      = ['error', 'bug', 'fix', 'issue', 'mistake', 'incorrect', 'fault', 'defect', 'flaw', 'solve' ]
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

def getDiffLOC(diff_text):
    add_cnt, del_cnt = 0, 0 
    diff_text_list = diff_text.split('\n') 
    diff_text_list = [x_ for x_ in diff_text_list if (('---' not in x_) and ('+++' not in x_)) ]
    add_text_list  = [x_ for x_ in diff_text_list if x_.startswith('+')]
    del_text_list  = [x_ for x_ in diff_text_list if x_.startswith('-')]

    # print add_text_list, del_text_list 
    add_cnt, del_cnt = len(add_text_list), len(del_text_list)
    return add_cnt, del_cnt 

def getDiffMetrics(diff_param):
    loc_add, loc_del = getDiffLOC(diff_param) 
    loc_tot          = loc_add + loc_del 
    return loc_add, loc_del, loc_tot 

def getDevsOfRepo(repo_path_param):
    commit_dict       = {}
    author_dict       = {}

    cdCommand         = "cd " + repo_path_param + " ; "
    commitCountCmd    = " git log --pretty=format:'%H_%an' "
    command2Run = cdCommand + commitCountCmd

    commit_count_output = subprocess.check_output(['bash','-c', command2Run])
    author_count_output = commit_count_output.split('\n')
    for commit_auth in author_count_output:
       commit_ = commit_auth.split('_')[0]
       
       author_ = commit_auth.split('_')[1]
       author_ = author_.replace(' ', '')
       # only one author for one commit 
       if commit_ not in commit_dict:
           commit_dict[commit_] = author_ 
       # one author can be involved with multiple commits 
       if author_ not in author_dict:
           author_dict[author_] = [commit_] 
       else:            
           author_dict[author_] = author_dict[author_] + [commit_] 
    return commit_dict, author_dict   

def getDevsExp(auth_name, auth_dict):
  exp_ = float(0)
  # print auth_name 
  # # print auth_dict 
  # print '='*10
  if auth_name in auth_dict:
    auth_commits = auth_dict[auth_name] 
    exp_ = len(auth_commits) 
  return exp_
    
def calcRecentExp(commit_year_list):
    recent_exp_final = 0 

    year_list = [int(x_) for x_ in commit_year_list ]
    dict_ = dict(Counter(year_list)) 
    unique_years = list(np.unique(year_list)) 
    unique_years.sort(reverse = True) 
    recent_exp_list = []
    for year_index in xrange(len(unique_years)):
        year_ = unique_years[year_index] 
        contribs = dict_[year_] 
        recent_exp = float(contribs) / float(year_index + 1)
        recent_exp_list.append(recent_exp) 
    recent_exp_final = round(sum(recent_exp_list) , 5) 

    return recent_exp_final 


def getDevsRecentExp(auth_name, auth_dict, time_dict):
  recent_exp_ = float(0)
  if auth_name in auth_dict:
    auth_commits     = auth_dict[auth_name] # get all commits for the author 
    commit_time_list = [time_dict[x_] for x_ in auth_commits if x_ in time_dict] # get the timestamp for all commits for author
    commit_year_list = [x_.split('T')[0].split('-')[0] for x_ in commit_time_list]  # get the year for all commits for author
    recent_exp_      = calcRecentExp(commit_year_list) # pass the year list to func 
  return recent_exp_

def getTimeCommits(repo_dir_absolute_path, bra_):
    time_commit_dict = {}
    repo_  = Repo(repo_dir_absolute_path)
    all_commits = list(repo_.iter_commits(bra_))  
    for commit_ in all_commits:
        commit_hash = commit_.hexsha
        timestamp_commit = commit_.committed_datetime
        str_time_commit  = timestamp_commit.strftime('%Y-%m-%dT%H-%M-%S')
        time_commit_dict[commit_hash] = str_time_commit
    return time_commit_dict 

def extractCommits(repo, branchName):
  str_dump = ''
  full_list, diff_list  = [], []
  repo_dir_absolute_path = '/Users/akond/TeachingRepos/' + repo + '/'
  print 'Started>' + repo_dir_absolute_path + '<' 
  repo_  = Repo(repo_dir_absolute_path)
  all_commits = list(repo_.iter_commits(branchName))
  commit_dict, author_dict = getDevsOfRepo(repo_dir_absolute_path) 
  commit_time_dict = getTimeCommits(repo_dir_absolute_path, branchName)

  sec_cnt = 0 
  for commit_ in all_commits: 
    secu_flag  = 'NEUTRAL'
    msg_commit =  commit_.message 

    msg_commit = msg_commit.replace('\n', ' ')
    msg_commit = msg_commit.replace(',',  ';')    
    msg_commit = msg_commit.replace('\t', ' ')
    msg_commit = msg_commit.replace('&',  ';')  
    msg_commit = msg_commit.replace('#',  ' ')
    msg_commit = msg_commit.replace('=',  ' ')      
    msg_commit = msg_commit.lower()

    commit_hash = commit_.hexsha

    timestamp_commit = commit_.committed_datetime
    str_time_commit  = timestamp_commit.strftime('%Y-%m-%dT%H-%M-%S') ## date with time 

    sec_kws_lower = [x_.lower() for x_ in secu_kws]
    commit_dff, mod_files_list   = getDiff(repo_dir_absolute_path, commit_hash) 
    add_loc, del_loc, tot_loc = getDiffMetrics(commit_dff)
    author = commit_dict[commit_hash] 
    author_exp = getDevsExp(author, author_dict) 
    author_recent_exp = getDevsRecentExp(author, author_dict, commit_time_dict) 

    for sec_kw in sec_kws_lower:
      if sec_kw in msg_commit:
          secu_flag  = 'INSECURE'
          sec_cnt += 1 
          str_dump = str_dump + sec_kw + '\n' + '*'*25  + msg_commit + '\n' + '*'*25 + '\n' + repo_dir_absolute_path + '\n' + '*'*25 + '\n' + commit_hash + '\n' + '*'*25 + '\n' 
    if secu_flag  != 'INSECURE': 
          commit_dff = 'INSECURE_DIFF' 
    
    for mod_file in mod_files_list:
        mod_file = unicode(mod_file, errors='ignore')
        tup_ = (repo, commit_hash, str_time_commit, add_loc, del_loc, tot_loc, author_exp, author_recent_exp, mod_file, secu_flag) 
        full_list.append(tup_) 
    diff_list.append( (commit_hash, commit_dff) )
  print 'Finished>' + repo_dir_absolute_path + '<insecure commit count------>' + str(sec_cnt)  
  print str_dump 
  return full_list , diff_list



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

def dumpBugStatus(projects, csv_file):
    str_dump = ''
    sec_kws_lower = [x_.lower() for x_ in secu_kws]
    bug_kws_lower = [x_.lower() for x_ in prem_bug_kw_list ]

    full_list = []
    for proj_ in projects:
        sec_cnt, bug_cnt = 0, 0 
        secu_flag = 'NEUTRAL'
        branchName = getBranchName(proj_)     
        repo_dir_absolute_path = '/Users/akond/TeachingRepos/' + proj_ + '/'
        print 'Started>' + repo_dir_absolute_path + '<' 
        repo_  = Repo(repo_dir_absolute_path)
        all_commits = list(repo_.iter_commits(branchName))   
        for commit_ in all_commits: 
            commit_hash = commit_.hexsha
            
            msg_commit =  commit_.message         
            msg_commit = msg_commit.replace('\n', ' ')
            msg_commit = msg_commit.replace(',',  ';')    
            msg_commit = msg_commit.replace('\t', ' ')
            msg_commit = msg_commit.replace('&',  ';')  
            msg_commit = msg_commit.replace('#',  ' ')
            msg_commit = msg_commit.replace('=',  ' ')      
            msg_commit = msg_commit.lower()    
            
            commit_dff, mod_files_list   = getDiff(repo_dir_absolute_path, commit_hash) 
            for sec_kw in sec_kws_lower:
              if sec_kw in msg_commit:
                  secu_flag  = 'INSECURE'
                  sec_cnt += 1  
            for bug_kw in bug_kws_lower:
              if bug_kw in msg_commit:
                  bug_flag  = 'BUGGY'
                  bug_cnt += 1         
                  str_dump = str_dump + bug_kw + '\n' + '*'*25  + msg_commit + '\n' + '*'*25 + '\n' + repo_dir_absolute_path + '\n' + '*'*25 + '\n' + commit_hash + '\n' + '*'*25 + '\n' 
            for mod_file in mod_files_list:
                mod_file = unicode(mod_file, errors='ignore')
                tup_ = (proj_, commit_hash, mod_file, secu_flag, bug_flag)     
                full_list.append(tup_)
        print 'Bugs:{}, Security bugs:{}, All:{}'.format( bug_cnt, sec_cnt, len(all_commits) ) 
        print '='*50                            
        print str_dump
        print '='*50   

    bug_status_df = pd.DataFrame(full_list) 
    bug_status_df.to_csv(csv_file, header=['REPO','HASH', 'MOD_FIL', 'SECU_FLAG', 'BUG_FLAG'], index=False, encoding='utf-8')

def checkIfPrior(curr, other):

  d_curr  = datetime.datetime.strptime(curr,  '%Y-%m-%dT%H-%M-%S').date()
  d_other = datetime.datetime.strptime(other, '%Y-%m-%dT%H-%M-%S').date()
  # print d_curr, d_other
  return d_curr > d_other

def getSecBugTypeDetails(csv_fil):
    p_mul = 50
    df_  = pd.read_csv(csv_fil)
    type_list = df_['SEC_BUG_TYPE'].tolist()
    type_freq_dict = dict(Counter(type_list)) 
    # print type_freq_dict
    hashes = np.unique(df_['HASH'].tolist())
    out_str = '' 
    proj_types = np.unique(df_['REPO_TYPE'].tolist()) 
    print dict(Counter(proj_types))
    for hash_ in hashes:
      hash_df = df_[df_['HASH']==hash_]
      sec_type  = hash_df['SEC_BUG_TYPE'].tolist()[0]
      if sec_type!= 'NONE':
        repo_name = hash_df['REPO'].tolist()[0] 
        proj_type = hash_df['REPO_TYPE'].tolist()[0] 
        # print proj_type
        # print hash_
        repo_dir  = '/Users/akond/TeachingRepos/' + repo_name
        diff_text, mod_files = getDiff(repo_dir, hash_)
        hash_msg  = hash_df['MSG'].tolist()[0]
        out_str = out_str + '*'*p_mul + '\n' + hash_ + '\n' + '*'*p_mul + '\n' + repo_dir + '\n' + '*'*p_mul + '\n' + sec_type + '\n' + '*'*p_mul + '\n' + proj_type + '\n' + '*'*p_mul + '\n'  + diff_text + '\n' + '*'*p_mul + '\n' + hash_msg + '\n' + '*'*p_mul + '\n'

    return out_str 

if __name__=='__main__':

    t1 = time.time()
    print 'Started at:', giveTimeStamp()
    print '*'*100

    diff_out_pkl_fil  = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/ALL_DIFF_COMM.PKL'
    secu_out_pkl_fil  = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/ALL_SECU_COMM.PKL'
    secu_out_csv_fil  = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/ALL_SECU_COMM.csv'
    bug_status_csv    = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/ALL_BUG_STATUS.csv'
    full_bug_exp_csv  = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/FULL_WITH_BUG_EXP.csv'

    fileName     = '/Users/akond/TeachingRepos/eligible_repos.csv'
    elgibleRepos = getEligibleProjects(fileName)

    secu_dict, diff_dict = {}, {}    
    full_list = []

    # for proj_ in elgibleRepos:
    #     branchName = getBranchName(proj_) 
    #     commit_secu_list, commit_diff_list = extractCommits(proj_, branchName)
    #     secu_dict[proj_] = commit_secu_list
    #     diff_dict[proj_] = commit_diff_list 
    #     full_list = full_list + commit_secu_list

    # with open(diff_out_pkl_fil, 'wb') as fp_:
    #     pickle.dump(diff_dict, fp_)  
    # with open(secu_out_pkl_fil, 'wb') as fp_:
    #     pickle.dump(secu_dict, fp_)  
    
    # all_proj_df = pd.DataFrame(full_list) 
    # all_proj_df.to_csv(secu_out_csv_fil, header=['REPO','HASH','TIME', 'ADD_LOC', 'DEL_LOC', 'TOT_LOC', 'DEV_EXP', 'DEV_RECENT', 'MODIFIED_FILE', 'SECU_FLAG'], index=False, encoding='utf-8') 

    # dumpBugStatus(elgibleRepos, bug_status_csv)

    sec_bug_type_file = '/Users/akond/Documents/AkondOneDrive/OneDrive/JobPrep-TNTU2019/teaching/ProjectMaterials/LOCKED_FINAL_SEC_BUG_TYPE.csv'
    full_diff_text = getSecBugTypeDetails(sec_bug_type_file) 

    print full_diff_text 

    print '*'*100
    print 'Ended at:', giveTimeStamp()
    print '*'*100
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print "Duration: {} minutes".format(time_diff)
    print '*'*100  

        
