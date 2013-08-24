from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, mean_squared_error, log_loss
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import randomData as rd
from itertools import cycle

# -----------------------------------------------------------------------------
# Randomly generated data: Training/Testing -----------------------------------
# -----------------------------------------------------------------------------

target_names = np.array(['Negatives','Positives'])

# call data frame with Nb = 50000 (negatives) and Ns - 10000 (positives)
dfr = rd.rData(target_names,282679734,50000,10000,0.0)

# Print first 25 rows of data frame
print dfr.head(25)

# Create training and testing data frame based on value of 'isTrain' column
train, test = dfr[dfr['isTrain']==True], dfr[dfr['isTrain']==False]

# Choose columns containing observables from four distributions (columns 0,1,2,3,4,5)
features = dfr.columns[0:6]

# Target values converted to integers for training and testing
trainTargets = np.array(train['Targets']).astype(int)
testTargets = np.array(test['Targets']).astype(int)

# -----------------------------------------------------------------------------
# Models --------------------------------------------
# -----------------------------------------------------------------------------

# list of model names
names = [
    'Naive Bayes',
    'Random Forest', 
    'Logistic Regression', 
    'SVM', 
    'Gradient Boosting',
    'Extra Trees',
    'Decision Tree',
    'Nearest Neighbors',
    'LDA', 
    'QDA',
    'Stochastic Gradient',
    'AdaBoost'
    ]

Ntrees = 400


# list of classifiers coresponding to the model names
classifiers = [
    GaussianNB(),
    RandomForestClassifier(n_estimators=Ntrees,max_depth=None,n_jobs=-1,random_state=2746298),
    LogisticRegression(random_state=2746298),
    svm.SVC(C=1.0, kernel='linear', probability=True),
    GradientBoostingClassifier(n_estimators=Ntrees,max_depth=1,learning_rate=1.0,random_state=2746298),
    ExtraTreesClassifier(n_estimators=Ntrees, max_depth=None,min_samples_split=1,random_state=2746298),
    DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=2746298),
    KNeighborsClassifier(3),
    LDA(),
    QDA(),
    SGDClassifier(loss="log",n_jobs=-1,random_state=2746298,shuffle=True),
    AdaBoostClassifier(n_estimators=100)
    ]

# initialize list for ploting
FP, TP, PR, RC, AVG = [], [], [], [], []

# loop over classifiers and calculate metrics
for name, clf in zip(names, classifiers):
    y = clf.fit(train[features], trainTargets).predict(train[features])
    y_t =  clf.predict(test[features])
    match = [i for i, j in zip(testTargets, y_t) if i == j]
    prob = clf.predict_proba(test[features])
    preds = target_names[clf.predict(test[features])]
    fpos, tpos, thr = roc_curve(testTargets, prob[:,1])
    pre, rc, thp = precision_recall_curve(testTargets, prob[:,1])

    cm = pd.crosstab(test['Type'], preds, rownames=['preds'], colnames=['actual'])
    
    tp = float(cm['Positives']['Positives'])
    tn = float(cm['Negatives']['Negatives'])
    fn = float(cm['Negatives']['Positives'])
    fp = float(cm['Positives']['Negatives'])   
    
    roc_auc = auc(fpos, tpos)
    acc = (tp+tn)/(tp+tn+fp+fn)
    lift = (tp+fp)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    prec = tp/(tp+fp)
    mcorr = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))  
    f1 = 2*tp/(2*tp+fn+fp)
    rms = np.sqrt(sum([(float(i) - float(j))**2 for i,j in zip(testTargets, y_t)])/len(y_t))
    #ll = -sum([i*np.log(float(j))+(1-i)*np.log(1-float(j)) for i,j in zip(testTargets,prob[:,1])])/len(y_t)
    ll = log_loss(testTargets, prob)
    average = (roc_auc+acc+recall+prec+mcorr+f1)/6
    
    if hasattr(clf, "feature_importances_"):  
        if name == 'Random Forest':
            feature_importance1 = clf.feature_importances_
            sorted_idx1 = np.argsort(feature_importance1)
            pos1 = np.arange(sorted_idx1.shape[0]) + 0.5
        elif name == 'Gradient Boosting':
            feature_importance2 = clf.feature_importances_
            sorted_idx2 = np.argsort(feature_importance2)
            pos2 = np.arange(sorted_idx2.shape[0]) + 0.5
        
# print out metrics  
    print name, '---------------------------------------------' 
    print '%Matches = ', float(len(match))/len(trainTargets)
    print '%Mis-Matches = ', 1 - (float(len(match))/len(trainTargets))
    print 'Area under ROC Curve = ', roc_auc
    print 'Accuracy Score = ', acc
    print 'Recall Score = ', recall
    print 'Precision Score', prec
    print 'Matthews Corrcoef. = ', mcorr
    print 'f1 Score = ', f1
    print 'Lift = ', lift
    print 'R.M.S. = ', rms
    print 'Cross-Entropy = ', ll
    #print 'Log Loss = ', ll2
    print 'Average = ', average
    print 'Confusion Matrix'
    print cm

# create list of values to plot
    FP.append(fpos)  
    TP.append(tpos)
    PR.append(pre)
    RC.append(rc)
    AVG.append(average)

# print list of model and corresponding 6 metric average
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

# plot roc curves for all models in list
plt.figure(1, figsize=(12,12)).patch.set_facecolor('white')
for i in range(len(names)):
    plt.plot(FP[i], TP[i],next(linecycler),label=names[i])
plt.plot([0, 1], [0, 1], 'k-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
#plt.savefig('/your/path/oc_compare.eps', bbox_inches=0)

# plot precisionRecall curves for all models in list
plt.figure(2, figsize=(12,12)).patch.set_facecolor('white')
for i in range(len(names)):
    plt.plot(PR[i], RC[i],next(linecycler),label=names[i])
plt.plot(0.5*np.ones(len(PR[0])), 'k-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc="lower left")
#plt.savefig('/your/path/rpcurve.png', bbox_inches=0)


# Observable importance of the Random Forest model 
plt.figure(3, figsize=(12,12)).patch.set_facecolor('white')
plt.barh(pos1, feature_importance1[sorted_idx1], align='center')
plt.yticks(pos1, dfr.columns[sorted_idx1])
plt.xlabel('% Relative Importance')
plt.title('Variable Importance')
plt.title('Random Forest')
#plt.savefig('/tour/path/rfVariableImportance.eps', bbox_inches=0)

# Observable importance of the Boosted Gradiend model 
plt.figure(4, figsize=(12,12)).patch.set_facecolor('white')
plt.barh(pos2, feature_importance2[sorted_idx2], align='center')
plt.yticks(pos2, dfr.columns[sorted_idx2])
plt.xlabel('% Relative Importance')
plt.title('Variable Importance')
plt.title('Boosted Gradient')

plt.show()