from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import numpy as np

from RobustRF import *


X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)



regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
print(regr.feature_importances_)
print(regr.predict([[0, 0, 0, 0]]))
test=regr.estimators_[0]
X=X.astype(np.float32)
#test.tree_.apply(X[0])
RF=RobustRandomForest()

RF.fit(X[5:],y[5:])
val=RF.predict(X[0:5])
leaves=RF.apply(X)
predictions,weights=RF.robustPredict(X[0:5],X,y)
newPred=RF.robustPredictUsingHuber(X[0:5],X,y)
newPredTukey=RF.robustPredictUsingTukey(X[0:5],X,y)
RF.KneighborstPredict(5,X[0:5],X,y)

