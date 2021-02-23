genesys_DS_Challenge_2021

Author: Ross Connole

## Running the scripts
The scripts were written using Python 3.9.2: https://www.python.org/downloads/release/python-392/

Install the package requirements using the following command from the project home dir: `pip install -r requirements.txt`

The project is split into two main scripts (*explore_data.py*, *model_eval.py*) and two utility scripts (*config.py*, *model_utils.py*).
The main scripts can be ran directly from the python console and will create plots in the plots directory (included in the repo but can be regenerated as needed). The utility scripts provide a place to store common variables and helper code such as the code for cleaning and encoding the data. 

## Data exploration
When exploring the data I tried to find any correlation between each of the variables. I found there was very low correlation between the outcome and most of the other features, the most promising features were *PrevAttempts*, *DaysPassed* and *CarInsurance* as illustrated by the graph below. 

![correlation_heatmap](https://github.com/rconnole/genesys_DS_Challenge_2021/blob/master/plots/correlation_heatmap.png?raw=true)

Using this data I attempted to use a feature selection algorithm but that affected performance negatively, so I commented out that code in the *model_eval.py* file. 

Looking at the distribution of the data and how they affect the outcome of the campaign there is a lot of negative outcome data. 
When looking at the number of classes in the outcome field we can see the trained model will be heavily skewed to predicting a negative outcome over a positive one. This leads to the model 

As part of the exploratory data analysis I print functions which will be seen in the scripts. Using the *DataFrame.info()* and *DataFrame.describe()* gave me sights into the statistics and distribution of the numerical data, for the labeled data I used graphs to check their distribution.   


## Model Evaluation

For this project I used the SciKitLearn library as I had previous experience with it. I decided to use a Logistic Regression as the outcome of the campaign was a binary value and LR works will with binary data. 

### Logistic Regression Pipeline
The *sklearn.Pipeline* class allowed me to combine various steps for preprocessing, feature scaling, and the estimator used before fitting the data to the model. I tried building the model with the feature selector Linear SVC, while looking at the feature correlation it seemed like there was a lot of unnecessary features (Marital, CarLoan, Communication etc.) that had very little baring on the outcome. I had hoped that the feature selection would remove these, but it had a negative effect on the final model. There could be more room for optimisation here to remove those features using another selection algorithm.

The LR algorithm performed better without any feature scaling or selection. The following details show the results for the model with feature selection. The recall of the model is particularly low and means the model missed a lot of true positive and true negative outcomes. The precision is also low (sub 50%) which means this classifier would be slightly worse than tossing a coin to decide the outcome for each test sample. The ROC Curve further shows this by being mostly linear in fashion.

>Results with feature selection 
> 
>Precision of model:  0.4411764705882353
>
>Recall of model:  0.11363636363636363
>
>Accuracy of model:  0.864

![roc_curve_with_feature_selection](https://github.com/rconnole/genesys_DS_Challenge_2021/blob/master/plots/roc_curve_with_feature_selection.png?raw=true)

When removing all feature selection or scaling and using the raw LR algorithm with the test data we find more promising results. The precision is now exactly 50%, on par with the coin flip classifier mentioned earlier. The recall has also risen to 33% to allow the model to miss a smaller amount of correctly classified samples. 

>Results without feature selection or scaling
> 
>Precision of model:  0.5 
> 
>Recall of model:  0.3333333333333333
> 
>Accuracy of model:  0.868

![roc_curve](https://github.com/rconnole/genesys_DS_Challenge_2021/blob/master/plots/roc_curve.png?raw=true)

### Conclusion
Both of these approaches leave a lot to be desired as they can easily be beaten by a coin flip classifier. I would improve this by performing more work on feature selection and scaling. As part of this project I simply encoded features to allow them to be fitted to the model, but some features could have been removed. Also, manual feature selection may have yielded better results then using the algorithm. The model could also be at risk of overfitting the training data as that contains a lot more negative outcomes then positive. by removing some negative outcome data and retraining the model there could be a more balanced prediction function for positive cases (which is also rare in the testing set).

