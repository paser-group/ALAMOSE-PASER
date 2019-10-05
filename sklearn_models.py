'''
Akond Rahman
Oct 25 2018 
sklearn prediction
'''

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, recall_score
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation, svm
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import utility
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import math 

iterDumpDir       = '/Users/akond/Documents/AkondOneDrive/OneDrive/stvr/output/'

def dumpPredPerfValuesToFile(iterations, predPerfVector, fileName):
   str2write=''
   headerStr='AUC,PRECISION,RECALL,F1,ACC,GMEAN,'
   for cnt in xrange(iterations):
       auc_   = predPerfVector[0][cnt]
       prec_  = predPerfVector[1][cnt]
       recal  = predPerfVector[2][cnt]
       f1     = predPerfVector[3][cnt]
       acc    = predPerfVector[4][cnt]
       gmean  = predPerfVector[5][cnt]
       
       str2write = str2write + str(auc_) + ',' + str(prec_) + ',' + str(recal) + ',' + str(f1) + ',' + str(acc) + ',' + str(gmean) + ',' + '\n'
   str2write = headerStr + '\n' + str2write
   bytes_ = utility.dumpContentIntoFile(str2write, fileName)
   print "Created {} of {} bytes".format(fileName, bytes_)

def evalClassifier(actualLabels, predictedLabels):
  '''
    the way skelarn treats is the following: first index -> lower index -> 0 -> 'Low'
                                             next index after first  -> next lower index -> 1 -> 'high'
  '''
  target_labels =  ['N', 'Y']
  class_report = classification_report(actualLabels, predictedLabels, target_names=target_labels)

  conf_matr_output = confusion_matrix(actualLabels, predictedLabels)
  # print "Confusion matrix start"
  # print conf_matr_output
  # print "Confusion matrix end"
  # preserve the order first test(real values from dataset), then predcited (from the classifier )
  prec_ = precision_score(actualLabels, predictedLabels, average='binary')
  recall_ = recall_score(actualLabels, predictedLabels, average='binary')
  area_roc_output = roc_auc_score(actualLabels, predictedLabels)
  fscore_output = f1_score(actualLabels, predictedLabels, average='binary')  
  accuracy_score_output = accuracy_score(actualLabels, predictedLabels)
  gmean_out = math.sqrt ( prec_ * recall_ ) ##reff: https://stats.stackexchange.com/questions/174011/can-g-mean-be-larger-than-accuracy

  return area_roc_output, prec_, recall_, fscore_output, accuracy_score_output, gmean_out

def perform_cross_validation(classiferP, featuresP, labelsP, cross_vali_param, infoP):
  predicted_labels = cross_validation.cross_val_predict(classiferP, featuresP , labelsP, cv=cross_vali_param)
  area_roc_to_ret = evalClassifier(labelsP, predicted_labels)
  return area_roc_to_ret

def performCART(featureParam, labelParam, foldParam, infoP):
  theCARTModel = DecisionTreeClassifier()
  cart_area_under_roc = perform_cross_validation(theCARTModel, featureParam, labelParam, foldParam, infoP)
  return cart_area_under_roc

def performKNN(featureParam, labelParam, foldParam, infoP):
  theKNNModel = KNeighborsClassifier()
  knn_area_under_roc = perform_cross_validation(theKNNModel, featureParam, labelParam, foldParam, infoP)
  return knn_area_under_roc


def performRF(featureParam, labelParam, foldParam, infoP):
  theRndForestModel = RandomForestClassifier()
  rf_area_under_roc = perform_cross_validation(theRndForestModel, featureParam, labelParam, foldParam, infoP)
  return rf_area_under_roc

def performSVC(featureParam, labelParam, foldParam, infoP):
  theSVMModel = svm.SVC(kernel='rbf').fit(featureParam, labelParam)
  svc_area_under_roc = perform_cross_validation(theSVMModel, featureParam, labelParam, foldParam, infoP)
  return svc_area_under_roc

def performLogiReg(featureParam, labelParam, foldParam, infoP):
  theLogisticModel = LogisticRegression()
  theLogisticModel.fit(featureParam, labelParam)
  logireg_area_under_roc = perform_cross_validation(theLogisticModel, featureParam, labelParam, foldParam, infoP)
  return logireg_area_under_roc

def performNaiveBayes(featureParam, labelParam, foldParam, infoP):
  theNBModel = GaussianNB() ### DEFAULT
  # theNBModel = BernoulliNB() ### TUNED

  theNBModel.fit(featureParam, labelParam)
  gnb_area_under_roc = perform_cross_validation(theNBModel, featureParam, labelParam, foldParam, infoP)
  return gnb_area_under_roc


def performModeling(features, labels, foldsParam):
  #r_, c_ = np.shape(features)
  ### lets do CART (decision tree)
  performCART(features, labels, foldsParam, "CART")
  print "="*100
  ### lets do knn (nearest neighbor)
  performKNN(features, labels, foldsParam, "KNN")
  print "="*100
  ### lets do RF (ensemble method: random forest)
  performRF(features, labels, foldsParam, "RF")
  print "="*100
  ### lets do SVC (support vector: support-vector classification)
  performSVC(features, labels, foldsParam, "SVC")
  print "="*100
  ### lets do Logistic regession
  performLogiReg(features, labels, foldsParam, "LogiRegr")
  print "="*100



def performIterativeModeling(iterDumpDir, featureParam, labelParam, foldParam, iterationP=10):
  cart_prec_holder, cart_recall_holder, holder_cart, f1_holder_cart, acc_holder_cart, gmean_holder_cart = [], [], [], [], [], []
  knn_prec_holder,  knn_recall_holder,  holder_knn, f1_holder_knn, acc_holder_knn, gmean_holder_knn  = [], [], [], [], [], []
  rf_prec_holder,   rf_recall_holder,   holder_rf, f1_holder_rf, acc_holder_rf, gmean_holder_rf   = [], [], [], [], [], []
  svc_prec_holder,  svc_recall_holder,  holder_svc, f1_holder_svc, acc_holder_svc, gmean_holder_svc  = [], [], [], [], [], []
  logi_prec_holder, logi_recall_holder, holder_logi, f1_holder_lr, acc_holder_lr, gmean_holder_lr = [], [], [], [], [], []
  nb_prec_holder,   nb_recall_holder,   holder_nb, f1_holder_nb, acc_holder_nb, gmean_holder_nb   = [], [], [], [], [], []

  for ind_ in xrange(iterationP):
    ## iterative modeling for CART
    cart_area_roc, cart_prec_, cart_recall_, cart_f1, cart_accu, cart_gmean = performCART(featureParam, labelParam, foldParam, "CART")

    holder_cart.append(cart_area_roc)
    cart_prec_holder.append(cart_prec_)
    cart_recall_holder.append(cart_recall_)

    f1_holder_cart.append(cart_f1)
    acc_holder_cart.append(cart_accu)
    gmean_holder_cart.append(cart_gmean)

    cart_f1 = 0 
    cart_accu = 0 

    cart_area_roc = float(0)
    cart_prec_    = float(0)
    cart_recall_  = float(0)
    cart_gmean = 0 


    ## iterative modeling for KNN
    knn_area_roc, knn_prec_, knn_recall_, knn_f1, knn_acc, knn_gmean = performKNN(featureParam, labelParam, foldParam, "K-NN")

    holder_knn.append(knn_area_roc)
    knn_prec_holder.append(knn_prec_)
    knn_recall_holder.append(knn_recall_)

    f1_holder_knn.append(knn_f1)
    acc_holder_knn.append(knn_acc)
    gmean_holder_knn.append(knn_gmean)

    knn_area_roc, knn_prec_, knn_recall_, knn_f1, knn_acc, knn_gmean = 0, 0, 0, 0, 0, 0


    ## iterative modeling for RF
    rf_area_roc, rf_prec_, rf_recall_, rf_f1, rf_accu, rf_gm  = performRF(featureParam, labelParam, foldParam, "Rand. Forest")

    holder_rf.append(rf_area_roc)
    rf_prec_holder.append(rf_prec_)
    rf_recall_holder.append(rf_recall_)

    f1_holder_rf.append(rf_f1)
    acc_holder_rf.append(rf_accu)
    gmean_holder_rf.append(rf_gm)

    rf_area_roc, rf_prec_, rf_recall_, rf_f1, rf_accu, rf_gm = 0, 0, 0, 0, 0, 0

    ## iterative modeling for SVC
    svc_area_roc, svc_prec_, svc_recall_, svc_f1, svc_accu,svc_gm  = performSVC(featureParam, labelParam, foldParam, "Supp. Vector Classi.")

    holder_svc.append(svc_area_roc)
    svc_prec_holder.append(svc_prec_)
    svc_recall_holder.append(svc_recall_)

    f1_holder_svc.append(svc_f1)
    acc_holder_svc.append(svc_accu)
    gmean_holder_svc.append(svc_gm)

    svc_area_roc, svc_prec_, svc_recall_, svc_f1, svc_accu, svc_gm  = 0, 0, 0, 0, 0, 0

    ## iterative modeling for logistic regression
    logi_reg_area_roc, logi_reg_preci_, logi_reg_recall, lr_f1, lr_ac, lr_gm = performLogiReg(featureParam, labelParam, foldParam, "Logi. Regression Classi.")

    holder_logi.append(logi_reg_area_roc)
    logi_prec_holder.append(logi_reg_preci_)
    logi_recall_holder.append(logi_reg_recall)

    f1_holder_lr.append(lr_f1)
    acc_holder_lr.append(lr_ac)
    gmean_holder_lr.append(lr_gm)

    logi_reg_area_roc, logi_reg_preci_, logi_reg_recall, lr_f1, lr_ac, lr_gm = 0, 0, 0, 0, 0, 0

    ## iterative modeling for naiev bayes
    nb_area_roc, nb_preci_, nb_recall, f1_nb, acc_nb, nb_gm  = performNaiveBayes(featureParam, labelParam, foldParam, "Naive Bayes")

    holder_nb.append(nb_area_roc)
    nb_prec_holder.append(nb_preci_)
    nb_recall_holder.append(nb_recall)

    f1_holder_nb.append(f1_nb)
    acc_holder_nb.append(acc_nb)
    gmean_holder_nb.append(nb_gm)

    nb_area_roc, nb_preci_, nb_recall, f1_nb, acc_nb, nb_gm = 0, 0, 0, 0, 0, 0


  print "-"*50
  print "Summary: AUC, for:{}, mean:{}, median:{}, max:{}, min:{}".format("CART", np.mean(holder_cart),
                                                                          np.median(holder_cart), max(holder_cart),
                                                                          min(holder_cart))
  print "*"*25
  print "Summary: Precision, for:{}, mean:{}, median:{}, max:{}, min:{}".format("CART", np.mean(cart_prec_holder),
                                                                          np.median(cart_prec_holder), max(cart_prec_holder),
                                                                          min(cart_prec_holder))
  print "*"*25
  print "Summary: Recall, for:{}, mean:{}, median:{}, max:{}, min:{}".format("CART", np.mean(cart_recall_holder),
                                                                          np.median(cart_recall_holder), max(cart_recall_holder),
                                                                          min(cart_recall_holder))
  print "*"*25
  print "Summary: F1, for:{}, mean:{}, median:{}, max:{}, min:{}".format("CART", np.mean(f1_holder_cart),
                                                                            np.median(f1_holder_cart), max(f1_holder_cart),
                                                                            min(f1_holder_cart))
  print "*"*25
  print "Summary: Accuracy, for:{}, mean:{}, median:{}, max:{}, min:{}".format("CART", np.mean(acc_holder_cart),
                                                                            np.median(acc_holder_cart), max(acc_holder_cart),
                                                                            min(acc_holder_cart))
  print "*"*25
  print "Summary: G-Mean, for:{}, mean:{}, median:{}, max:{}, min:{}".format("CART", np.mean(gmean_holder_cart),
                                                                            np.median(gmean_holder_cart), max(gmean_holder_cart),
                                                                            min(gmean_holder_cart))
  print "*"*25
  cart_all_pred_perf_values = (holder_cart, cart_prec_holder, cart_recall_holder, f1_holder_cart, acc_holder_cart, gmean_holder_cart)
  dumpPredPerfValuesToFile(iterationP, cart_all_pred_perf_values, iterDumpDir+'PRED_PERF_CART.csv')
  print "-"*50


  print "Summary: AUC, for:{}, mean:{}, median:{}, max:{}, min:{}".format("KNN", np.mean(holder_knn),
                                                                          np.median(holder_knn), max(holder_knn),
                                                                          min(holder_knn))
  print "*"*25
  print "Summary: Precision, for:{}, mean:{}, median:{}, max:{}, min:{}".format("KNN", np.mean(knn_prec_holder),
                                                                          np.median(knn_prec_holder), max(knn_prec_holder),
                                                                          min(knn_prec_holder))
  print "*"*25
  print "Summary: Recall, for:{}, mean:{}, median:{}, max:{}, min:{}".format("KNN", np.mean(knn_recall_holder),
                                                                          np.median(knn_recall_holder), max(knn_recall_holder),
                                                                          min(knn_recall_holder))
  print "*"*25
  print "Summary: F1, for:{}, mean:{}, median:{}, max:{}, min:{}".format("KNN", np.mean(f1_holder_knn),
                                                                            np.median(f1_holder_knn), max(f1_holder_knn),
                                                                            min(f1_holder_knn))
  print "*"*25
  print "Summary: Accuracy, for:{}, mean:{}, median:{}, max:{}, min:{}".format("KNN", np.mean(acc_holder_knn),
                                                                            np.median(acc_holder_knn), max(acc_holder_knn),
                                                                            min(acc_holder_knn))
  print "*"*25
  print "Summary: G-Mean, for:{}, mean:{}, median:{}, max:{}, min:{}".format("KNN", np.mean(gmean_holder_knn),
                                                                            np.median(gmean_holder_knn), max(gmean_holder_knn),
                                                                            min(gmean_holder_knn))
  print "*"*25  
  knn_all_pred_perf_values = (holder_knn, knn_prec_holder, knn_recall_holder, f1_holder_knn, acc_holder_knn, gmean_holder_knn)
  dumpPredPerfValuesToFile(iterationP, knn_all_pred_perf_values, iterDumpDir+'PRED_PERF_KNN.csv')
  print "-"*50


  print "Summary: AUC, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Rand. Forest", np.mean(holder_rf),
                                                                          np.median(holder_rf), max(holder_rf),
                                                                          min(holder_rf))
  print "*"*25
  print "Summary: Precision, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Rand. Forest", np.mean(rf_prec_holder),
                                                                          np.median(rf_prec_holder), max(rf_prec_holder),
                                                                          min(rf_prec_holder))
  print "*"*25
  print "Summary: Recall, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Rand. Forest", np.mean(rf_recall_holder),
                                                                          np.median(rf_recall_holder), max(rf_recall_holder),
                                                                          min(rf_recall_holder))
  print "*"*25
  print "Summary: F1, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Rand. Forest", np.mean(f1_holder_rf),
                                                                            np.median(f1_holder_rf), max(f1_holder_rf),
                                                                            min(f1_holder_rf))
  print "*"*25
  print "Summary: Accuracy, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Rand. Forest", np.mean(acc_holder_rf),
                                                                            np.median(acc_holder_rf), max(acc_holder_rf),
                                                                            min(acc_holder_rf))
  print "*"*25
  print "Summary: G-Mean, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Rand. Forest", np.mean(gmean_holder_rf),
                                                                            np.median(gmean_holder_rf), max(gmean_holder_rf),
                                                                            min(gmean_holder_rf))
  print "*"*25  
  rf_all_pred_perf_values = (holder_rf, rf_prec_holder, rf_recall_holder, f1_holder_rf, acc_holder_rf, gmean_holder_rf)
  dumpPredPerfValuesToFile(iterationP, rf_all_pred_perf_values, iterDumpDir+'PRED_PERF_RF.csv')
  print "-"*50

  print "Summary: AUC, for:{}, mean:{}, median:{}, max:{}, min:{}".format("S. Vec. Class.", np.mean(holder_svc),
                                                                          np.median(holder_svc), max(holder_svc),
                                                                          min(holder_svc))
  print "*"*25
  print "Summary: Precision, for:{}, mean:{}, median:{}, max:{}, min:{}".format("S. Vec. Class.", np.mean(svc_prec_holder),
                                                                            np.median(svc_prec_holder), max(svc_prec_holder),
                                                                            min(svc_prec_holder))
  print "*"*25
  print "Summary: Recall, for:{}, mean:{}, median:{}, max:{}, min:{}".format("S. Vec. Class.", np.mean(svc_recall_holder),
                                                                            np.median(svc_recall_holder), max(svc_recall_holder),
                                                                            min(svc_recall_holder))
  print "*"*25
  print "Summary: F1, for:{}, mean:{}, median:{}, max:{}, min:{}".format("S. Vec. Class.", np.mean(f1_holder_svc),
                                                                            np.median(f1_holder_svc), max(f1_holder_svc),
                                                                            min(f1_holder_svc))
  print "*"*25
  print "Summary: Accuracy, for:{}, mean:{}, median:{}, max:{}, min:{}".format("S. Vec. Class.", np.mean(acc_holder_svc),
                                                                            np.median(acc_holder_svc), max(acc_holder_svc),
                                                                            min(acc_holder_svc))
  print "*"*25
  print "Summary: G-Mean, for:{}, mean:{}, median:{}, max:{}, min:{}".format("S. Vec. Class.", np.mean(gmean_holder_svc),
                                                                            np.median(gmean_holder_svc), max(gmean_holder_svc),
                                                                            min(gmean_holder_svc))
  print "*"*25  
  svc_all_pred_perf_values = (holder_svc, svc_prec_holder, svc_recall_holder, f1_holder_svc, acc_holder_svc, gmean_holder_svc)
  dumpPredPerfValuesToFile(iterationP, svc_all_pred_perf_values, iterDumpDir+'PRED_PERF_SVC.csv')
  print "-"*50

  print "Summary: AUC, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Logi. Regression", np.mean(holder_logi),
                                                                          np.median(holder_logi), max(holder_logi),
                                                                          min(holder_logi))
  print "*"*25
  print "Summary: Precision, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Logi. Regression", np.mean(logi_prec_holder),
                                                                            np.median(logi_prec_holder), max(logi_prec_holder),
                                                                            min(logi_prec_holder))
  print "*"*25
  print "Summary: Recall, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Logi. Regression", np.mean(logi_recall_holder),
                                                                            np.median(logi_recall_holder), max(logi_recall_holder),
                                                                            min(logi_recall_holder))
  print "*"*25
  print "Summary: F1, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Logi. Regression", np.mean(f1_holder_lr),
                                                                            np.median(f1_holder_lr), max(f1_holder_lr),
                                                                            min(f1_holder_lr))
  print "*"*25
  print "Summary: Accuracy, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Logi. Regression", np.mean(acc_holder_lr),
                                                                            np.median(acc_holder_lr), max(acc_holder_lr),
                                                                            min(acc_holder_lr))
  print "*"*25
  print "Summary: G-Mean, for:{}, mean:{}, median:{}, max:{}, min:{}".format("S. Vec. Class.", np.mean(gmean_holder_lr),
                                                                            np.median(gmean_holder_lr), max(gmean_holder_lr),
                                                                            min(gmean_holder_lr))
  print "*"*25  
  logireg_all_pred_perf_values = (holder_logi, logi_prec_holder, logi_recall_holder, f1_holder_lr, acc_holder_lr, gmean_holder_lr)
  dumpPredPerfValuesToFile(iterationP, logireg_all_pred_perf_values, iterDumpDir+'PRED_PERF_LOGIREG.csv')
  print "-"*50


  print "Summary: AUC, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Naive Bayes", np.mean(holder_nb),
                                                                          np.median(holder_nb), max(holder_nb),
                                                                          min(holder_nb))
  print "*"*25
  print "Summary: Precision, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Naive Bayes", np.mean(nb_prec_holder),
                                                                            np.median(nb_prec_holder), max(nb_prec_holder),
                                                                            min(nb_prec_holder))
  print "*"*25
  print "Summary: Recall, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Naive Bayes", np.mean(nb_recall_holder),
                                                                            np.median(nb_recall_holder), max(nb_recall_holder),
                                                                            min(nb_recall_holder))
  print "*"*25
  print "Summary: F1, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Naive Bayes", np.mean(f1_holder_nb),
                                                                            np.median(f1_holder_nb), max(f1_holder_nb),
                                                                            min(f1_holder_nb))
  print "*"*25
  print "Summary: Accuracy, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Naive Bayes", np.mean(acc_holder_nb),
                                                                            np.median(acc_holder_nb), max(acc_holder_nb),
                                                                            min(acc_holder_nb))
  print "*"*25
  print "Summary: G-Mean, for:{}, mean:{}, median:{}, max:{}, min:{}".format("Naive Bayes", np.mean(gmean_holder_nb),
                                                                            np.median(gmean_holder_nb), max(gmean_holder_nb),
                                                                            min(gmean_holder_nb))
  print "*"*25  
  nb_all_pred_perf_values = (holder_nb, nb_prec_holder, nb_recall_holder, f1_holder_nb, acc_holder_nb, gmean_holder_nb)
  dumpPredPerfValuesToFile(iterationP, nb_all_pred_perf_values, iterDumpDir+'PRED_PERF_NB.csv')
  print "-"*50
  return cart_all_pred_perf_values, knn_all_pred_perf_values, rf_all_pred_perf_values, svc_all_pred_perf_values, logireg_all_pred_perf_values, nb_all_pred_perf_values
