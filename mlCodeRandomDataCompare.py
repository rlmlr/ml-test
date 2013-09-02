from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, mean_squared_error
from sklearn.metrics import log_loss, average_precision_score, f1_score, confusion_matrix
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import randomData as rd
from itertools import cycle

# -----------------------------------------------------------------------------
# Load randomly generated data: Training/Testing -----------------------------------
# -----------------------------------------------------------------------------

# Create 1x2 array of target names
target_names = np.array(['Negatives','Positives'])

# Call data frame with Nb (negatives) and Ns (positives)
dfr = rd.rData(target_names,282679734,5000,1000)

# Print first 25 rows of data frame
print dfr.head(25)

# Create training and testing data frame based on value of 'isTrain' column
train, test = dfr[dfr['isTrain']==True], dfr[dfr['isTrain']==False]

# Choose columns containing features from six distributions (columns 0,1,2,3,4,5)
features = dfr.columns[0:6]

# Target values converted to integers for training and testing
trainTargets = np.array(train['Targets']).astype(int)
testTargets = np.array(test['Targets']).astype(int)

# -----------------------------------------------------------------------------
# Models --------------------------------------------
# -----------------------------------------------------------------------------

# List of model names
names = [
    'Naive Bayes', # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
    'Random Forest', # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    'Logistic Regression', # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    'SVM', # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    'Gradient Boosting', # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
    'Extra Trees', # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    'Decision Tree', # http://scikit-learn.org/stable/modules/tree.html#classification
    'Nearest Neighbors', # http://scikit-learn.org/stable/modules/neighbors.html
    'LDA', # http://scikit-learn.org/stable/modules/generated/sklearn.lda.LDA.html
    'QDA', # http://scikit-learn.org/stable/modules/generated/sklearn.qda.QDA.html
    'Stochastic Gradient', # http://scikit-learn.org/stable/modules/sgd.html#classification
    'AdaBoost' # http://scikit-learn.org/stable/modules/ensemble.html#adaboost
    ]

# List of classifiers coresponding to the model names
classifiers = [
    GaussianNB(),
    RandomForestClassifier(n_estimators=300,max_depth=None,n_jobs=-1,random_state=2746298),
    LogisticRegression(random_state=2746298),
    svm.SVC(C=1.0, kernel='linear', probability=True),
    GradientBoostingClassifier(n_estimators=300,max_depth=1,learning_rate=1.0,random_state=2746298),
    ExtraTreesClassifier(n_estimators=300, max_depth=None,min_samples_split=1,random_state=2746298),
    DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=2746298),
    KNeighborsClassifier(3),
    LDA(),
    QDA(),
    SGDClassifier(loss="log",n_jobs=-1,random_state=2746298,shuffle=True),
    AdaBoostClassifier(n_estimators=100)
    ]

# Initialize list for ploting
FP, TP, PR, RC, AVG, FI = [], [], [], [], [], []

# loop over classifiers and calculate metrics
for name, clf in zip(names, classifiers):
    y = clf.fit(train[features], trainTargets).predict(train[features])
    y_t = clf.predict(test[features])
    # Probabilities
    prob = clf.predict_proba(test[features])
    # Predictions matched to target names
    preds = target_names[clf.predict(test[features])]
    # ROC and Precision Recall Curve
    fpos, tpos, thr = roc_curve(testTargets, prob[:,1])
    pre, rc, thp = precision_recall_curve(testTargets, prob[:,1])

    # Confusion Matrix
    #cm = confusion_matrix(testTargets, y_t)
    cm = pd.crosstab(test['Type'], preds, rownames=['preds'], colnames=['actual'])
    
    tp = float(cm['Positives']['Positives'])
    tn = float(cm['Negatives']['Negatives'])
    fn = float(cm['Negatives']['Positives'])
    fp = float(cm['Positives']['Negatives'])   
    
    # Metrics 
    roc_auc = auc(fpos, tpos) # Area under ROC curve
    avp = average_precision_score(testTargets, y_t) # Area under Precision Recall curve
    acc = accuracy_score(testTargets, y_t) # (tp+tn)/(tp+tn+fp+fn)
    lift = (tp+fp)/(tp+tn+fp+fn)
    recall = recall_score(testTargets, y_t) # tp/(tp+fn)
    prec = precision_score(testTargets, y_t) # tp/(tp+fp)
    mcorr = matthews_corrcoef(testTargets, y_t) # (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))  
    f1 = f1_score(testTargets, y_t) # 2*tp/(2*tp+fn+fp)
    rms = mean_squared_error(testTargets, y_t) # np.sqrt(sum([(float(i)-float(j))**2 for i,j in zip(testTargets, y_t)])/len(y_t))
    ll = log_loss(testTargets, prob) # sum([(tT=1)log(P1)+(1-tT=0)log(1-P0)] for i,j,k,l in zip(testTargets,prob[:,0],prob[:,1]))
    average = (roc_auc+acc+recall+prec+f1+avp)/6
       
    # Compute fearture importance when available
    if hasattr(clf, "feature_importances_"):  
        FI.append((name, clf.feature_importances_))
      
# Print out metrics  
    print name, '---------------------------------------------' 
    print 'Area under ROC Curve = ', roc_auc
    print 'Area under Precision Recall curve = ', avp
    print 'Accuracy Score = ', acc
    print 'Recall Score = ', recall
    print 'Precision Score = ', prec
    print 'Matthews Corrcoef. = ', mcorr
    print 'f1 Score = ', f1
    print 'Lift = ', lift
    print 'R.M.S. = ', rms
    print 'Cross-Entropy = ', ll
    print 'Average over metrics in [0,1] = ', average
    print 'Confusion Matrix'
    print cm

    # Create list of values to plot
    FP.append(fpos)  
    TP.append(tpos)
    PR.append(pre)
    RC.append(rc)
    AVG.append(average)

# Print list of model and corresponding 6 metric average
print '--------------------------------------------------------'
print 'Model','         ', 'Score'
for k in zip(names,AVG):
    print k

# -----------------------------------------------------------------------------
# Plots -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

#colormap = plt.cm.gist_ncar
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(names))])

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

# Plot roc curves for all models in list
plt.figure(1, figsize=(12,12)).patch.set_facecolor('white')
for i in range(len(names)):
    plt.plot(FP[i], TP[i],next(linecycler),label=names[i])
plt.plot([0,1], [0, 1], 'k-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc_compare.png', bbox_inches=0)

# Plot precisionRecall curves for all models in list
plt.figure(2, figsize=(12,12)).patch.set_facecolor('white')
for i in range(len(names)):
    plt.plot(PR[i], RC[i],next(linecycler),label=names[i])
plt.plot(0.5*np.ones(len(PR[0])), 'k-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc="lower left")
plt.savefig('rpcurve.png', bbox_inches=0)

# Feature importance plots
plt.figure(3, figsize=(12,12)).patch.set_facecolor('white')
n = 0
for i,j in FI:
    plt.subplot(len(FI),1,n+1)
    sorted_idx = np.argsort(j)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.barh(pos, j[sorted_idx], align='center')
    plt.yticks(pos, dfr.columns[sorted_idx])
    plt.xlabel('% Relative Importance')
    plt.title('Variable Importance')
    plt.title(i)
    n += 1
    #plt.savefig('/tour/path/rfVariableImportance.eps', bbox_inches=0)

plt.show()
