# Charectizing Security Bugs in Scientific Software Projects 

## Project Update#1 

> For what types of files do we see security bugs? 

[] Filter out noisy file name: look for text which doesn’t have extensions 
[] Separate out files that are security-related (‘INSECURE’) and that are not (‘NEUTRAL’)
[] Apply text mining (TF-IDF) on the two groups 
[] Get the text mining matrix, and sort it by TF-IDF scores for both groups  
[] Take top 1000 TF-IDF scores for both groups 
[] Look at the obtained features manually and see what features appear: each member must do it individually then discuss agreements and disagreements  
