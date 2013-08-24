from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import roc_curve, precision_recall_curve, mean_squared_error, log_loss
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import randomData as rd
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
# -----------------------------------------------------------------------------
# Randomly generated data: Training/Testing -----------------------------------
# -----------------------------------------------------------------------------

target_names = np.array(['Negatives','Positives'])

dfr = rd.rData(target_names,282679734,50000,10000)

print dfr.head(25)

train, test = dfr[dfr['isTrain']==True], dfr[dfr['isTrain']==False]

features = dfr.columns[0:6]

trainTargets = np.array(train['Targets']).astype(int)
testTargets = np.array(test['Targets']).astype(int)

# -----------------------------------------------------------------------------
# Model  ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

names = dfr.columns[0:6].tolist()

print names

name = 'Gradient Boosting',


Ntrees = 100

clf = GradientBoostingClassifier(n_estimators=Ntrees,max_depth=1,learning_rate=1.0,random_state=2746298)#,
    

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
ll = log_loss(testTargets, prob)
average = (roc_auc+acc+recall+prec+mcorr+f1)/6
    
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
        
    
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
print 'Average = ', average
print 'Confusion Matrix'
print cm

# -----------------------------------------------------------------------------
# Plots -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

fig1, axs = plot_partial_dependence(clf, train[features], features[0:6], feature_names=names, n_jobs=-1, grid_resolution=50,facecolor='white')
fig1.suptitle('None')
pl.subplots_adjust(top=0.9) 


fig2 = pl.figure()
target_feature = (3, 4)
pdp, (x_axis, y_axis) = partial_dependence(clf, target_feature,X=train[features], grid_resolution=50)
XX, YY = np.meshgrid(x_axis, y_axis)
Z = pdp.T.reshape(XX.shape).T
ax = Axes3D(fig2)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=pl.cm.BuPu)
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')

plt.show()
