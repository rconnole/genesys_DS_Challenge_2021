from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC

import model_utils as utils
import config
import pandas as pd
import numpy as np

trainingDS = pd.read_csv(config.data_dir + config.train_dataset, header=0)

trainingDS = utils.clean_data(trainingDS, encode_data=True)
print(trainingDS.info())
print(trainingDS["Job"].unique())

x_train, x_test, y_train, y_test = train_test_split(trainingDS.drop("Outcome", axis=1), trainingDS.Outcome,
                                                    test_size=0.3, shuffle=True, random_state=1)

log_reg_pipe = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    ('preprocessor', StandardScaler()),
    ('classification', LogisticRegression())
])

log_reg_pipe.fit(x_train, y_train)
model_score = log_reg_pipe.score(x_test, y_test)
test_model_pred = log_reg_pipe.predict(x_test)

print("Using LogisticRegression the model scored: ", model_score)
conf_matrix = confusion_matrix(y_test, test_model_pred)
print(conf_matrix)

# Model Evaluation
testingDS = pd.read_csv(config.data_dir + config.test_dataset, header=0)
testingDS = utils.clean_data(testingDS, encode_data=True)
testingDS.info()
print(testingDS.head())
x_eval = np.array(testingDS.drop("Outcome", axis=1))
y_eval = np.array(testingDS.Outcome)
eval_pred = log_reg_pipe.predict(x_eval)
eval_conf_matrix = confusion_matrix(y_eval, eval_pred)
print(eval_conf_matrix)
