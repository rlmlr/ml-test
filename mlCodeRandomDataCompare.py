"""
Created on Tues Sep  3 21:03:57 2013

@author: rlmlr
email: rralich@gmail.com
"""

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
import matplotlib.pyplot as plt
import randomData as rd
from itertools import cycle

# -----------------------------------------------------------------------------
# Load randomly generated data: Training/Testing ------------------------------
# -----------------------------------------------------------------------------

# Create 1x2 array of target names
target_names = np.array(['Positives','Negatives'])

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
# Models ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

classifiers = [('Naive Bayes', GaussianNB()), 
               ('Decision Tree', DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=2746298)),
               ('Random Forest', RandomForestClassifier(n_estimators=300,max_depth=None,n_jobs=-1,random_state=2746298)),
               ('Extra Trees', ExtraTreesClassifier(n_estimators=300, max_depth=None,min_samples_split=1,random_state=2746298)),
               ('Gradient Boosting', GradientBoostingClassifier(n_estimators=300,max_depth=1,learning_rate=1.0,random_state=2746298)),
               ('Logistic Regression', LogisticRegression(random_state=2746298)),
               ('SVM', svm.SVC(C=1.0, kernel='linear', probability=True)),
               ('Nearest Neighbors', KNeighborsClassifier(3)),
               ('LDA', LDA()),
               ('QDA', QDA()),
               ('Stochastic Gradient', SGDClassifier(loss="log",n_jobs=-1,random_state=2746298,shuffle=True)),
               ('AdaBoost', AdaBoostClassifier(n_estimators=100))
               ]

# Initialize list for saving values for plotting
FP, TP, PR, RC, AVG, SAR, FI, Names = [], [], [], [], [], [], [], []

# Loop over classifiers and calculate metrics
for name, clf in classifiers:
    y = clf.fit(train[features], trainTargets).predict(train[features])
    y_t = clf.predict(test[features])
    # Probabilities
    prob = clf.predict_proba(test[features])
    # Predictions matched to target names
    preds = target_names[clf.predict(test[features])]
    # ROC and PR Curve
    fpos, tpos, thr = roc_curve(testTargets, prob[:,1])
    pre, rc, thp = precision_recall_curve(testTargets, prob[:,1])

    # Confusion Matrix
    cm = confusion_matrix(testTargets, y_t)
    
    tp = float(cm[0,0])
    tn = float(cm[1,1])
    fn = float(cm[0,1])
    fp = float(cm[1,0])   
    
    # Metrics 
    roc_auc = auc(fpos, tpos) # Area under ROC curve
    avp = average_precision_score(testTargets, y_t) # Area under Precision Recall curve
    acc = accuracy_score(testTargets, y_t) # (tp+tn)/(tp+tn+fp+fn)
    lift = (tp+fp)/(tp+tn+fp+fn)
    recall = recall_score(testTargets, y_t) # tp/(tp+fn)
    prec = precision_score(testTargets, y_t) # tp/(tp+fp)
    #mcorr = matthews_corrcoef(testTargets, y_t) # BROKEN!!!
    mcorr = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))  
    f1 = f1_score(testTargets, y_t) # 2*tp/(tp+fn+fp)
    rms = mean_squared_error(testTargets, y_t) # np.sqrt(sum([(float(i)-float(j))**2 for i,j in zip(testTargets, y_t)])/len(y_t))
    ll = log_loss(testTargets, prob) # (y_T=1)log(P(y_T=1)+(1-y_T=0)log(1-P(y_T=0)
    sar = (acc + roc_auc + (1-rms))/3
    average = (roc_auc+acc+recall+prec+f1+avp+sar)/7
       
    # Compute fearture importance when available
    if hasattr(clf, "feature_importances_"):  
        FI.append((name, clf.feature_importances_))
      
# Print out metrics  
    print name, '---------------------------------------------' 
    print 'Area under ROC Curve = ', roc_auc
    print 'Area under PR curve = ', avp
    print 'Accuracy Score = ', acc
    print 'Recall Score = ', recall
    print 'Precision Score = ', prec
    print 'Matthews Corrcoef. = ', mcorr
    print 'f1 Score = ', f1
    print 'Lift = ', lift
    print 'R.M.S. = ', rms
    print 'Cross-Entropy = ', ll
    print 'SAR = ', sar
    print 'Average over metrics in [0,1] = ', average
    print 'Confusion Matrix'
    print cm

    # Create list of values to plot
    FP.append(fpos)  
    TP.append(tpos)
    PR.append(pre)
    RC.append(rc)
    AVG.append(average)
    SAR.append(sar)
    Names.append(name)

# Print list of model and corresponding Average and SAR
print '--------------------------------------------------------'
print 'Model','         ', 'Average','         ', 'SAR'
for k in zip(Names,AVG,SAR):
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
for i in range(len(classifiers)):
    plt.plot(FP[i], TP[i],next(linecycler),label=Names[i])
plt.plot([0,1], [0, 1], 'k-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('/home/ROC_All.png', bbox_inches=0)

# Plot precisionRecall curves for all models in list
plt.figure(2, figsize=(12,12)).patch.set_facecolor('white')
for i in range(len(classifiers)):
    plt.plot(PR[i], RC[i],next(linecycler),label=Names[i])
plt.plot(0.5*np.ones(len(PR[0])), 'k-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc="lower left")
plt.savefig('/home/PR_ALL.png', bbox_inches=0)


# Feature importance plots
plt.figure(3, figsize=(12,12)).patch.set_facecolor('white')
n = 0
for i,j in FI:
    plt.subplot(len(FI),1,n+1)
    sorted_idx = np.argsort(j)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.barh(pos, j[sorted_idx], align='center')
    plt.yticks(pos, dfr.columns[sorted_idx])
    if n == len(FI)-1:
        plt.xlabel('% Importance')
    plt.title('Variable Importance')
    plt.title(i)
    n += 1
plt.savefig('/home/FI_All.png', bbox_inches=0)

plt.show()
