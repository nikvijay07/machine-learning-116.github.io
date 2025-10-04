# Heart Failure Detection

## Introduction

Heart failure is a life threatening condition that plagues many of the healthcare systems and population globally. It happens when your heart is not able to pump blood properly to meet the needs of your body [1]. Early detection is key for receiving proper treatment and mitigation of consequences like strokes or heart attacks down the line. For our project, we set out to apply machine learning techniques to this field to predict if patients have a heart disease. Testing for heart disease today relies on a variety of different measures like blood tests, MRI scans, CT scans, or even exercise tests, as not one specific one can tell you the full picture [5]. Given the multimodality of the current state of testing, it highlights a tradeoff to be made for any model: less features generally allows a more widely applicable model however specialized features could produce a higher accuracy.  

Our dataset [2] is a labeled Kaggle dataset that combines multiple individual datasets internationally into 1 dataset to provide a more diverse range of information. It includes the following features:

1. Age 
2. Sex
3. Chest Pain Type:
4. Resting Blood Pressure
5. Cholesterol
6. Fasting Blood Sugar
7. Resting ElectroCardiogram Results
8. Maximum Heart Rate
9. Exercise Induced Angina
10. Oldpeak
11. ST_Slope
12. Heart Disease

## Problem Definition

Heart disease remains one of the most pressing health challenges worldwide, contributing to significant rates of mortality and long-term disability. The difficulty lies in the fact that heart disease often develops gradually and can remain undetected until severe complications arise, such as heart attacks, strokes, or organ failure. Early detection is crucial, but current diagnostic methods rely on an array of tests ranging from imaging scans to invasive procedures, which can be costly, time consuming, and inaccessible to many patients. This complexity creates barriers to timely diagnosis and effective prevention, leaving many individuals unaware of their risk until it is too late.
The problem, therefore, is not only the prevalence of heart disease but also the challenge of identifying at-risk individuals with efficiency and accuracy before critical events occur. By applying machine learning techniques to patient health data, there is an opportunity to uncover meaningful patterns and predictors of heart disease that are not easily observed through traditional testing. Such predictive tools could serve as powerful aids for clinicians and patients alike, providing earlier warnings, guiding healthier lifestyle changes, and ultimately reducing the global burden of cardiovascular disease.

## Methods

Preprocessing:
  1. Remove duplicate entries with pandas.DataFrame.drop_duplicates() from pandas
  2. Feature selection: We will determine what features we want to use based on variance in the sklearn.feature_selection module.
  3. Outlier detection and removal: We will possibly remove outliers if their z-score is too high.

Supervised learning - We are using supervised learning because the heart disease variable serves as the label for our data

We will use logarithmic regression because it’s useful for binary labels such as whether someone has heart disease or not. We can use the LogisticRegression class within scikit-learn.

We may also use random forest to get a good general model that isn’t overfit and doesn’t rely solely on any feature. Random forest will also just be helpful for prediction in general, and we can use the RandomForestRegressor class with the RandomForestClassifier class. [6]

Finally, we will implement a gradient boosting classifier for the advantages of robustness and being able to iteratively decrease the loss function as well as having strength when dealing with outliers. We can use the GradientBoostinClassifier class from scikit-learn. [7]

## Potential Results and Discussion

As we are using classification methods like logistic regression and random forest, we will use the standard set of metrics used to evaluate classification algorithms based on the frequency of true/false negatives and positives (TN, FN, TP, FP respectively), while also interpreting them in the context of medical decision-making.
Accuracy: the measure of correct classifications (TN + TP) divided by the total number (TN + TP + FN + FP). This metric tends to be dominated by the number of true negatives in a sparse dataset, but we should be able to avoid this because the number of heart disease vs. non-heart disease samples is relatively equal.
Precision: how many of our guesses actually were positive (TP / TP + FP). This metric needs to be taken with the fact that it allows us to mainly avoid false positives, which are less important than making sure there are no false negatives (using this model as a screening tool rather than a straight up diagnostic.)
Recall: the fraction of how many of the true cases we found (TP / TP + FN). On the flip side of precision, recall is very important in this context because it is typically best to ensure that the model overdiagnoses positives and finds them rather than underdiagnose. 
F1 score: the harmonic mean of precision and recall, is able to balance the amount of false positives and false negatives which is a core metric for evaluating an efficient yet comprehensive model. 
ROC-AUC (area under receiver operating characteristics curve): measures the ability of the model to distinguish between classes across thresholds. A higher AUC reduces the risk of either missing true cases or falsely alarming healthy patients.


**Expected Results**

Logistic regression will give us interpretable coefficients, which are useful for understanding risk factors, but may be outperformed by Random Forest in accuracy, precision, and recall due to its ability to capture nonlinear relationships and feature interactions.

Our goal is to achieve an F1 score above 0.75 and an AUC above 0.80, balancing both sensitivity and precision. In particular, we expect recall to be prioritized slightly higher than precision, given the importance of identifying at-risk patients in a medical setting.



