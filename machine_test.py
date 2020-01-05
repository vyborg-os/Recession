import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB #importing GuassianNB from naive bayes in scikit-learn
from sklearn.model_selection import cross_val_score
mydataset = pd.read_excel("C:\\Users\\Vyborg\\Desktop\\MyProject\\corrected\\cdata.xlsx")
mydataset = mydataset.values
print(mydataset)
X = mydataset[:,:-1]
Y = mydataset[:,-1]
c = cross_val_score(GaussianNB(),X,Y, cv=5)
print("Cross validation value = ",c)
print("The average is: ",c.mean()," ",round(c.mean() * 100,3),"%")
# print(c.std())

      # I am using TPOT autoML library for python

# import numpy as np
# import pandas as pd
#
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.pipeline import make_pipeline, make_union
# from tpot.builtins import StackingEstimator
#
# from sklearn.preprocessing import LabelEncoder
#
# tpot_data = pd.read_csv("C:\\Users\\Vyborg\\Desktop\\MyProject\\corrected\\pdata.xlsx")
#
# tpot_data = tpot_data.apply(LabelEncoder().fit_transform)
#
# features = tpot_data.drop('species', axis=1).values
#
# training_features, testing_features, training_target, testing_target = \
#                    train_test_split(features, tpot_data['species'].values, random_state=10)
#
# exported_pipeline = make_pipeline(StackingEstimator(estimator=GaussianNB()),
#                                 MultinomialNB(alpha=0.01, fit_prior=False)
# )
#
#
# exported_pipeline.fit(training_features, training_target)
#
# results = exported_pipeline.predict(testing_features)
#
# from sklearn import metrics
# print("Accuracy:", metrics.accuracy_score(testing_target, results))
#
#
# pd.crosstab(testing_target, results, rownames=['Actual Class'], colnames=['Predicted Class'])
#
#
# from sklearn.model_selection import cross_val_score
#
# array_cross_val_score = cross_val_score(estimator=exported_pipeline, X=training_features,
#                             y=training_target, cv=10, scoring='accuracy')
#
# # I would like the confusion matrix to be based on the average cross-validation
# np.mean(array_cross_val_score)