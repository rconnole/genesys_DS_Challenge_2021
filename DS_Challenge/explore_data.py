import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import model_utils as utils
import config as config

trainingDS = pd.read_csv(config.data_dir + config.train_dataset, header=0)
print(trainingDS.describe())

trainingDS = utils.clean_data(trainingDS, encode_data=False)

print(trainingDS.describe())

print(trainingDS.info())
print(trainingDS.head())


jobs_outcome_df = pd.crosstab(trainingDS["Job"], trainingDS["Outcome"], colnames=["Successful Campaign"])
print(jobs_outcome_df.head())
jobs_outcome_df.plot(kind="bar")
plt.xticks(rotation=30)
plt.show()

# check out correlations in the data, especially the outcome of the campaign
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(trainingDS.corr(), annot=True, linewidths=5, fmt='.1f', ax=ax, cmap='Reds')
plt.show()


# age distribution
trainingDS.Age.value_counts().plot.barh()
plt.title("Age distribution")
plt.yticks(rotation=30)
plt.ylabel("Age Group")
plt.xlabel("Count")
plt.show()

# call duration distribution
trainingDS.CallDuration.value_counts().plot.barh()
plt.title("Call Duration distribution")
plt.yticks(rotation=30)
plt.ylabel("Call Duration")
plt.xlabel("Count")
plt.show()
print(trainingDS.head())
