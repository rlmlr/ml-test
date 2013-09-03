from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm, grid_search
import numpy as np
import randomData as rd

# -----------------------------------------------------------------------------
# Import randomly generated data: Training/Testing -----------------------------------
# -----------------------------------------------------------------------------

target_names = np.array(['Negatives','Positives'])

# call data frame with Nb = 50000 (negatives) and Ns - 10000 (positives)
dfr = rd.rData(target_names,282679734,5000,1000)

# Print first 25 rows of data frame
print dfr.head(25)

dataTargets = np.array(dfr['Targets']).astype(int)
data = np.transpose(np.array((dfr['Beta'],dfr['Binary'],dfr['Exp'],dfr['Gamma'],dfr['Norm'],dfr['Quad'])))

print data.shape, dataTargets.shape

# Models ----------------------------------------------------------------------
# Logistic Regression
param_grid_lrc = [
  {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1']},
  {'C': [0.01, 0.1, 1, 10, 100], 'penalty':['l2']},
  ]

lrc = LogisticRegression(random_state=2746298)

clf = grid_search.GridSearchCV(lrc, param_grid_lrc, n_jobs=-1, cv=4)
clf.fit(data, dataTargets)

# Print best scores and bese parameters
print 'Logistic Regression ----------------------------------------------------'
print 'Best Score = ', clf.best_score_
print 'Best Estimator = ', clf.best_estimator_

#print 'All Scores = ', clf.grid_scores_
#print 'Best Parameters = ', clf.best_params_

# -----------------------------------------------------------------------------

param_grid_rfc = [
  {'n_estimators': [10, 100, 200, 300, 400, 500], 'criterion': ['gini']},
  {'n_estimators': [10, 100, 200, 300, 400, 500], 'criterion':['entropy']},
  ]

rfc = RandomForestClassifier(random_state=2746298)

clf = grid_search.GridSearchCV(rfc, param_grid_rfc, n_jobs=-1, cv=4)
clf.fit(data, dataTargets)

# Print best scores and bese parameters
print 'RandomForest ----------------------------------------------------------'
print 'Best Score = ', clf.best_score_   
print 'Best Estimator = ', clf.best_estimator_

#print 'All Scores = ', clf.grid_scores_
#print 'Best Parameters = ', clf.best_params_

# -----------------------------------------------------------------------------

param_grid_gbc = [
  {'n_estimators': [10, 100, 200, 300, 400, 500], 'learning_rate': [0.001,0.01,0.01,0.1,1,10,100]},
  {'n_estimators': [10, 100, 200, 300, 400, 500], 'learning_rate': [0.001,0.01,0.01,0.1,1,10,100]},
  ]

gbc = GradientBoostingClassifier(random_state=2746298)

clf = grid_search.GridSearchCV(gbc, param_grid_gbc, n_jobs=-1, cv=4)
clf.fit(data, dataTargets)

# Print best scores and bese parameters
print 'GradientBoosting -------------------------------------------------------'
print 'Best Score = ', clf.best_score_   
print 'Best Estimator = ', clf.best_estimator_

#print 'All Scores = ', clf.grid_scores_
#print 'Best Parameters = ', clf.best_params_

# -----------------------------------------------------------------------------

param_grid_svc = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2, 3]},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 0.0001], 'kernel': ['rbf']},
 ]

svc = svm.SVC(random_state=2746298)

clf = grid_search.GridSearchCV(svc, param_grid_svc, n_jobs=-1, cv=4)
clf.fit(data, dataTargets)

# Print best scores and bese parameters
print 'SVC -------------------------------------------------------------------'
print 'Best Score = ', clf.best_score_   
print 'Best Estimator = ', clf.best_estimator_

#print 'All Scores = ', clf.grid_scores_
#print 'Best Parameters = ', clf.best_params_

