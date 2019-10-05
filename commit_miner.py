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
from datetime import datetime
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
  strToret = datetime.fromtimestamp(tsObj).strftime('%Y-%m-%d %H:%M:%S')
  return strToret

def getEligibleProjects(fileNameParam):
  repo_list = []
  with open(fileNameParam, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:
      repo_list.append(row[0])
  return repo_list

def getDiff(repo_, hash_):
   
    cdCommand   = "cd " + repo_ + " ; "
   
    diffCommand = "git diff " + hash_ + "~" + " " + hash_
    command2Run = cdCommand + diffCommand
    try:
      diff_output = subprocess.check_output(["bash", "-c", command2Run])
    except subprocess.CalledProcessError as e_:
      diff_output = "NOT_FOUND" 
    return diff_output

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

def getTimeCommits(repo_dir_absolute_path, branch_):
    time_commit_dict = {}
    time_list = []
    repo_  = Repo(repo_dir_absolute_path)
    all_commits = list(repo_.iter_commits(branch_))  
    for commit_ in all_commits:
        commit_hash = commit_.hexsha
        timestamp_commit = commit_.committed_datetime
        str_time_commit  = timestamp_commit.strftime('%Y-%m-%dT%H-%M-%S')
        time_commit_dict[commit_hash] = str_time_commit 
        time_list.append(str_time_commit) 
    return time_commit_dict , time_list  

def getAgeBefore(curr_val, all_list):
    curr_date    = datetime.strptime(curr_val, '%Y-%m-%dT%H-%M-%S')    
    all_dates    = [datetime.strptime(x_, '%Y-%m-%dT%H-%M-%S') for x_ in all_list]
    prior_dates  = [z_ for z_ in all_dates if z_ < curr_date ] 
    if (len(prior_dates) <= 0):
      prior_dates.append(curr_date) 
    start_date   = min(prior_dates) 
    age_diff     = curr_date - start_date 
    priorDays    = age_diff.days 
    return priorDays

def getFilesInDiff(diff_str):
  splitted_str = diff_str.split('\n') 
  valid_strs   = [x_ for x_ in splitted_str if '+++' in x_]  
  valid_file_names = [x_.split(' ')[-1] for x_ in valid_strs]
  return valid_file_names 

def getBranchName(proj_):
    branch_name = ''
    proj_branch = {'biemond@biemond-oradb':'puppet4_3_data', 'derekmolloy@exploringBB':'version2', 'exploringBB':'version2', 
                   'jippi@puppet-php':'php7.0', 'maxchk@puppet-varnish':'develop', 'threetreeslight@my-boxen':'mine', 
                   'puppet':'production', 'deepvariant':'r0.8', 'galaxy':'dev', 'mdanalysis':'develop', 'pyGeno':'bloody',
                   'miso-lims':'develop' , 'jevo':'Float64-optimization'
                  } 
    if proj_ in proj_branch:
        branch_name = proj_branch[proj_]
    else: 
        branch_name = 'master'
    return branch_name

def extractCommits(repo, branchName):
  str_dump = ''
  full_list, diff_list  = [], []
  repo_dir_absolute_path = '/Users/akond/TeachingRepos/' + repo + '/'
  print 'Started>' + repo_dir_absolute_path + '<' 
  repo_  = Repo(repo_dir_absolute_path)
  all_commits = list(repo_.iter_commits(branchName))
  commit_dict, author_dict = getDevsOfRepo(repo_dir_absolute_path) 
  commit_time_dict, time_ls  = getTimeCommits(repo_dir_absolute_path, branchName) 

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

    time_before_commit = getAgeBefore(str_time_commit, time_ls) ## days before this commit 

    sec_kws_lower = [x_.lower() for x_ in secu_kws]
    commit_dff  = getDiff(repo_dir_absolute_path, commit_hash) 
    changed_files = getFilesInDiff(commit_dff) 
    changed_file_cnt = len(changed_files) 
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
    tup_ = (repo, commit_hash, str_time_commit, add_loc, del_loc, tot_loc, author_exp, author_recent_exp, time_before_commit , changed_file_cnt , secu_flag) 
    # print tup_
    full_list.append(tup_) 
    diff_list.append( (commit_hash, commit_dff) )
  print 'Finished>' + repo_dir_absolute_path + '<insecure commit count------>' + str(sec_cnt)  
  return full_list , diff_list

def dumpBugStatus(projects, csv_file):
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
            for sec_kw in sec_kws_lower:
              if sec_kw in msg_commit:
                  secu_flag  = 'INSECURE'
                  sec_cnt += 1  
            for bug_kw in bug_kws_lower:
              if bug_kw in msg_commit:
                  bug_flag  = 'BUGGY'
                  bug_cnt += 1         
            tup_ = (proj_, commit_hash, secu_flag, bug_flag)     
            full_list.append(tup_)
        print 'Bugs:{}, Security bugs:{}, All:{}'.format( bug_cnt, sec_cnt, len(all_commits) ) 
        print '='*50                            

    bug_status_df = pd.DataFrame(full_list) 
    bug_status_df.to_csv(csv_file, header=['REPO','HASH','SECU_FLAG', 'BUG_FLAG'], index=False, encoding='utf-8')

def checkIfPrior(curr, other):

  d_curr  = datetime.datetime.strptime(curr,  '%Y-%m-%dT%H-%M-%S').date()
  d_other = datetime.datetime.strptime(other, '%Y-%m-%dT%H-%M-%S').date()
  # print d_curr, d_other
  return d_curr > d_other

def getBugExp(secu_out_fil, bug_status_fil, full_bug_out_fil):
    secu_out_df   = pd.read_csv(secu_out_fil) 
    bug_status_df = pd.read_csv(bug_status_fil) 
    all_commits   = np.unique(secu_out_df['HASH'].tolist() )
    all_repos     = np.unique( secu_out_df['REPO'].tolist() )
    lis           = []
    for repo_ in all_repos:
      print '='*50
      print repo_
      print 'Started at:', giveTimeStamp()
      print '='*50      
      repo_dir_absolute_path   = '/Users/akond/TeachingRepos/' + repo_ + '/'
      branchName = getBranchName(repo_)     
      commit_dict, author_dict = getDevsOfRepo(repo_dir_absolute_path) 
      time_commit_dict         = getTimeCommits(repo_dir_absolute_path)
      repo_df     = secu_out_df[secu_out_df['REPO']==repo_]
      repo_comm   = np.unique( repo_df['HASH'].tolist()  )
      for comm_ in repo_comm:
          temp_secu_list, temp_bug_list = [], []
          curr_time      = repo_df[repo_df['HASH']==comm_]['TIME'].tolist()[0]
          curr_secu_flag = bug_status_df[bug_status_df['HASH']==comm_]['SECU_FLAG'].tolist()[0]
          curr_bug_flag  = bug_status_df[bug_status_df['HASH']==comm_]['BUG_FLAG'].tolist()[0]
          if comm_ in commit_dict:
            author = commit_dict[comm_] 
            if author in author_dict:
               commits = author_dict[author]
               for auth_comm in commits:
                  time_comm = time_commit_dict[auth_comm]
                  if (checkIfPrior(curr_time, time_comm)):
                     secu_flag = bug_status_df[bug_status_df['HASH']==auth_comm]['SECU_FLAG'].tolist()[0]
                     bug_flag  = bug_status_df[bug_status_df['HASH']==auth_comm]['BUG_FLAG'].tolist()[0]
                     if secu_flag=='INSECURE':
                        temp_secu_list.append(auth_comm)
                     if bug_flag=='BUGGY':
                        temp_bug_list.append(auth_comm) 
          dev_bug_exp  = len(temp_bug_list)
          dev_secu_exp = len(temp_secu_list)
          tup = (repo_, comm_, curr_time, curr_secu_flag, curr_bug_flag, dev_bug_exp, dev_secu_exp)
          lis.append( tup )

      print 'Data collected so far .... {} entries'.format(len(lis)) 
      print '='*50
      print 'Ended at:', giveTimeStamp()
      print '='*50            
    bugExpDF = pd.DataFrame(lis)
    bugExpDF.to_csv(full_bug_out_fil, header=['REPO','HASH', 'TIME', 'SECU_FLAG', 'BUG_FLAG', 'PRIOR_BUG_EXP', 'PRIOR_SECU_EXP'], index=False, encoding='utf-8')     

          
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

    for proj_ in elgibleRepos:
        branchName = getBranchName(proj_) 
        commit_secu_list, commit_diff_list = extractCommits(proj_, branchName)
        secu_dict[proj_] = commit_secu_list
        diff_dict[proj_] = commit_diff_list 
        full_list = full_list + commit_secu_list

    # with open(diff_out_pkl_fil, 'wb') as fp_:
    #     pickle.dump(diff_dict, fp_)  
    # with open(secu_out_pkl_fil, 'wb') as fp_:
    #     pickle.dump(secu_dict, fp_)  
    
    all_proj_df = pd.DataFrame(full_list) 
    all_proj_df.to_csv(secu_out_csv_fil, header=['REPO','HASH','TIME', 'ADD_LOC', 'DEL_LOC', 'TOT_LOC', 'DEV_EXP', 'DEV_RECENT', 'PRIOR_AGE', 'CHANGE_FILE_CNT' , 'SECU_FLAG'], index=False, encoding='utf-8') 

    # dumpBugStatus(elgibleRepos, bug_status_csv)

    # getBugExp(secu_out_csv_fil, bug_status_csv, full_bug_exp_csv)

    print '*'*100
    print 'Ended at:', giveTimeStamp()
    print '*'*100
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print "Duration: {} minutes".format(time_diff)
    print '*'*100  

        
