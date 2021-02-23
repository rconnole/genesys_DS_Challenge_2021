import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config
import model_utils as utils

trainingDS = pd.read_csv(config.data_dir + config.train_dataset, header=0)

trainingDS = utils.clean_data(trainingDS, encode_data=True)
print(trainingDS.info())
print(trainingDS["Job"].unique())
# hold back some data for testing built model
x_train, x_test, y_train, y_test = train_test_split(trainingDS.drop("Outcome", axis=1), trainingDS.Outcome,
                                                    test_size=0.25, shuffle=True, random_state=1)
# use pipeline for building model
# I initially tried feature selection as I saw low correlation between features but it resulted in a much lower
# precision and recall with the final model. Work could be done to optimise feature selection and feature scaling
# if you would like to see the horrible result uncomment the 'feature_selection' and 'preprocessor' lines below
log_reg_pipe = Pipeline([
    # ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    # ('preprocessor', StandardScaler()),
    ('classification', LogisticRegression())
])

log_reg_pipe.fit(x_train, y_train)
model_score = log_reg_pipe.score(x_test, y_test)
pred_test = log_reg_pipe.predict(x_test)

print("Using LogisticRegression the model scored: ", model_score)
conf_matrix = confusion_matrix(y_test, pred_test)
print(conf_matrix)

# Model Evaluation
testingDS = pd.read_csv(config.data_dir + config.test_dataset, header=0)
testingDS = utils.clean_data(testingDS, encode_data=True)
testingDS.info()
print(testingDS.head())
x_eval = np.array(testingDS.drop("Outcome", axis=1))
y_eval = np.array(testingDS.Outcome)
pred_eval = log_reg_pipe.predict(x_eval)
eval_conf_matrix = confusion_matrix(y_eval, pred_eval)
print(eval_conf_matrix)

print("Precision of model: ", precision_score(y_eval, pred_eval))
print("Recall of model: ", recall_score(y_eval, pred_eval))
print("Accuracy of model: ", accuracy_score(y_eval, pred_eval))

fpr, tpr, threshold = roc_curve(y_eval, pred_eval)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(config.output_dir + "roc_curve.png")
