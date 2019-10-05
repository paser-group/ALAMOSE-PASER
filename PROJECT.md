# Charectizing Security Bugs in Scientific Software Projects 

## Project Update#1 

> For what types of files do we see security bugs? 

-  Filter out noisy file name: look for text which doesn’t have extensions 
-  Separate out files that are security-related (‘INSECURE’) and that are not (‘NEUTRAL’)
-  Apply text mining (TF-IDF) on the two groups 
-  Get the text mining matrix, and sort it by TF-IDF scores for both groups  
-  Take top 1000 TF-IDF scores for both groups 
-  Look at the obtained features manually and see what features appear: each member must do it individually then discuss agreements and disagreements  


## Project Update#2

> Construct prediction models to predict security bugs in scientific software? 

- Type-based model: REPO_TYPE
- Size-based model: ADD_LOC, DEL_LOC, TOT_LOC
- Time-based model: PRIOR_AGE
- Full model: ADD_LOC, DEL_LOC, TOT_LOC, PRIOR_AGE, REPO_TYPE
- Repeat the following three steps for type-based model, size-based model, time-based model, and full model:
- Take CSV as input, separate out independent variable(s), and the dependent variable is SECU_FLAG
  - Apply Naïve Bayes, kNN, Decision Tree, ANN, and Random Forest
- Apply 10 by 10 fold cross validation , and then report prediction accuracy using precision, recall, and F-measure. 

