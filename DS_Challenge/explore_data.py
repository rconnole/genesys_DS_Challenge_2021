import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

import config as config
import model_utils as utils

# use two copies of the Dataset to allow for labeled data
# and encoded data when needed for graphs
training_DS = pd.read_csv(config.data_dir + config.train_dataset, header=0)
labeled_DS = utils.clean_data(training_DS.copy(), encode_data=False)
encoded_DS = utils.clean_data(training_DS, encode_data=True)

# print statements to help me understand the data
print(labeled_DS.describe())
print(labeled_DS.info())
print(labeled_DS.Job.describe())

jobs_outcome_df = pd.crosstab(labeled_DS["Job"], labeled_DS["Outcome"], colnames=["Successful Campaign"])
jobs_outcome_df.plot(kind="bar")
plt.xticks(rotation=30)
plt.savefig(config.output_dir + "jobs_outcome.png")
plt.clf()

# check out correlations in the data, especially the outcome of the campaign
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(encoded_DS.corr(), annot=True, linewidths=5, fmt='.1f', ax=ax, cmap='Reds')
plt.title("Heatmap of feature correlation")
plt.savefig(config.output_dir + "correlation_heatmap.png")
plt.clf()

# balance distribution
print(labeled_DS.Balance.describe())
sns.distplot(labeled_DS.Balance)
plt.title("Balance distribution")
plt.savefig(config.output_dir + "balance_distribution.png")
plt.clf()

# age distribution - value_counts works better with labeled data
labeled_DS.Age.value_counts().plot.barh()
plt.title("Age distribution")
plt.yticks(rotation=30)
plt.ylabel("Age Group")
plt.xlabel("Count")
plt.savefig(config.output_dir + "age_distribution.png")
plt.clf()

# call duration distribution
labeled_DS.CallDuration.value_counts().plot.barh()
plt.title("Call Duration distribution")
plt.yticks(rotation=30)
plt.ylabel("Call Duration")
plt.xlabel("Count")
plt.savefig(config.output_dir + "call_duration_bar.png")
plt.clf()

## Car Insruance has a higher then average correlation with the outcome
## Checking if Age or Job have any pattern to them when plotted against car insurance
# check car insurance VS age
age_car_insurance = pd.crosstab(labeled_DS["Age"], labeled_DS["CarInsurance"].apply(
    lambda x: "No Insurance" if x == 0 else "Has Insurance"), colnames=["What Age groups have car "
                                                                        "insurance?"])
age_car_insurance.plot(kind="bar")
plt.xticks(rotation=30)
plt.savefig(config.output_dir + "age_car_insurance.png")
plt.clf()

# check car insurance VS Job
job_car_insurance = pd.crosstab(labeled_DS["Job"], labeled_DS["CarInsurance"].apply(
    lambda x: "No Insurance" if x == 0 else "Has Insurance"), colnames=["What Jobs have car "
                                                                        "insurance?"])
job_car_insurance.plot(kind="bar")
plt.xticks(rotation=30)
plt.savefig(config.output_dir + "job_car_insurance.png")
plt.clf()
