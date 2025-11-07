# Heart Failure Detection

ML Project Proposal<br />
Max Roozbahani<br />
CS 4641<br />
Group 116

## Introduction

  Heart failure is a life threatening condition that plagues our population globally. It happens when your heart is not able to pump blood properly to meet the needs of your body [1]. Early detection is key for receiving proper treatment and mitigation of consequences like heart attacks down the line. For our project, we will apply machine learning techniques to this field to predict if patients have a heart disease. Testing for heart disease today relies on a variety of different tests, as not one specific one can tell you the full picture [5]. Given the range in the current state of testing, it highlights a tradeoff to be made for any model: less features allows a more widely applicable model however specialized features could produce a higher accuracy.  
  Our dataset [2] is a labeled Kaggle dataset that combines multiple individual datasets internationally into 1 dataset to provide a more diverse range of information. It includes the following features:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting-Blood-Sugar
* Resting ElectroCardiogram Results
* Maximum Heart Rate
* Exercise Induced Angina
* Oldpeak
* ST_Slope
* Heart Disease

## Problem Definition

Heart disease remains one of the most pressing health challenges worldwide, contributing to significant rates of mortality and long-term disability. The difficulty is that heart disease often develops gradually and can remain undetected until severe complications arise, such as heart attacks, or strokes. Current diagnostic methods rely on an array of tests. This complexity creates barriers to timely diagnosis and effective prevention, leaving many individuals unaware of their risk until it is too late.
By applying machine learning techniques to patient health data, we can find meaningful patterns of heart disease that are not easily observed through traditional testing. Such predictive tools could serve as powerful aids for clinicians and patients alike, and ultimately reduce the problem of cardiovascular disease.

## Methods

The first step in creating our model was to preprocess our data. Initially, we checked for duplicates and found none. Following that, we wanted to convert qualitative data into numerical data so it was usable by our model. We used one hot encoding here to convert these features because they had relatively low cardinality and this technique works well with logistic regression. The next step was to perform outlier detection and removal. By visualizing our data with box plots for each feature, we were able to see our distributions and point out any strong outliers. In many instances, we saw individuals with a cholesterol level of 0 or resting blood pressure of 0, which is not biologically possible, making the data recording impractical. However, we did decide to keep most outliers as they were valid data points and represented people’s true data. For the biologically impossible data points, we utilized K-nearest-neighbors with K = 5 and averaged their values for that data point. We used 4 features in order to avoid the curse of dimensionality: Cholesterol, RestingBP, Age, and MaxHR. These features were also selected because their values were continuous integers which were easier to compute distance from. Below is an example of the cholesterol data before and after preprocessing.

<img width="523" height="300" alt="Screenshot 2025-11-07 at 2 11 25 PM" src="https://github.com/user-attachments/assets/1b76de14-79a9-4377-bead-11b56e5fb261" />

Figure 1: Before KNN-Imputation


<img width="523" height="300" alt="Screenshot 2025-11-07 at 2 11 38 PM" src="https://github.com/user-attachments/assets/86c2c959-ca5c-4ca0-bbc3-8d3797ecdc73" />

Figure 2: After KNN-Imputation

Finally, we decided to perform standardization on a few features to make sure that the features are on the same scale and therefore weighted similarly. 

We chose a logistic regression model to predict based on our data because it is applicable to binary classification. To adjust for overfitting, we  tested with both ridge and lasso regularization. some features weights to go to 0 if they were not relevant. During this process, we found that ​​RestingBP, Cholesterol, and RestingECG_Normal all ended up with coefficients of 0. 


## Results and Discussion

Here is the following results on accuracy:

<table>
  <tr>
    <td>
      <h3>Lasso Regularization (L1)</h3>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Accuracy</td><td>0.853</td></tr>
        <tr><td>True Positive (TP)</td><td>89</td></tr>
        <tr><td>False Positive (FP)</td><td>9</td></tr>
        <tr><td>True Negative (TN)</td><td>68</td></tr>
        <tr><td>False Negative (FN)</td><td>18</td></tr>
        <tr><td>Precision</td><td>0.908</td></tr>
        <tr><td>Recall</td><td>0.832</td></tr>
        <tr><td>F1 Score</td><td>0.868</td></tr>
      </table>
    </td>
    <td style="padding-left: 20px;">
      <h3>Ridge Regularization (L2)</h3>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Accuracy</td><td>0.859</td></tr>
        <tr><td>True Positive (TP)</td><td>90</td></tr>
        <tr><td>False Positive (FP)</td><td>9</td></tr>
        <tr><td>True Negative (TN)</td><td>68</td></tr>
        <tr><td>False Negative (FN)</td><td>17</td></tr>
        <tr><td>Precision</td><td>0.909</td></tr>
        <tr><td>Recall</td><td>0.841</td></tr>
        <tr><td>F1 Score</td><td>0.874</td></tr>
      </table>
    </td>
  </tr>
</table>

Overall, our model did achieve our goal of an F1 score of above 80%. 


## Potential Results and Discussion

As we are using classification methods like logistic regression and random forest, we will evaluate them using the standard metrics of the frequency of true/false negatives and positives (TN, FN, TP, FP respectively), while also interpreting them in a medical context.

Accuracy: the measure of correct classifications (TN + TP) divided by the total number (TN + TP + FN + FP). While accuracy can be skewed by TN in imbalanced datasets, our heart disease data is relatively balanced.

Precision: how many of our guesses actually were positive (TP / TP + FP). This metric mainly allows us to avoid false positives, though in screening, avoiding false negatives is usually more critical.

Recall: the fraction of how many of the true cases we found (TP / TP + FN). On the flip side of precision, recall is important in this context because it is typically best to ensure that the model overdiagnoses positives rather than underdiagnoses. 

F1 score: the harmonic mean of precision and recall, is able to balance the measure of false positives and false negatives which is a core metric for evaluating efficiency and coverage.

ROC-AUC: measures the ability of the model to distinguish between classes across thresholds. A higher AUC reduces the risk of either missing true cases or falsely alarming healthy patients. [3][4]



**Expected Results**

Logistic regression will give us interpretable coefficients, which are useful for understanding risk factors, but may be outperformed by Random Forest in accuracy, precision, and recall due to its ability to capture nonlinear relationships and feature interactions.

Our goal is to achieve an F1 score above 0.75 and an AUC above 0.80, balancing both sensitivity and precision. In particular, we expect recall to be prioritized slightly higher than precision, given the importance of identifying at-risk patients in a medical setting.


<img width="282" height="213" alt="Screenshot 2025-10-03 at 10 12 39 PM" src="https://github.com/user-attachments/assets/b5d9a990-da51-43ed-b1ef-f702c7f3f2d2" />

[Gantt Chart](https://docs.google.com/spreadsheets/d/12L8_VrgD5vhyxSndnmP3nrqRqYoYeFIWrzhPhH9Uay8/edit?usp=sharing)


## References



[1] Cleveland Clinic, “Congestive Heart Failure", https://www.clevelandclinic.org, para. 1, March 10, 2023. [Online]. Available: https://my.clevelandclinic.org/health/diseases/17069-heart-failure-understanding-heart-failure. [Accessed Oct. 3, 2025]. 

[2] FS. Palacios, Clinic, “Heart Failure Production Dataset", https://www.kaggle.com, 2021. [Online]. Available: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. [Accessed Oct. 3, 2025]. 

[3] M. Moffat, “Performance Metric in Medical Image Segmentation”, May 31, 2024. [Online]. Available: https://medium.com/@matthewmoffat/different-performance-metrics-in-medical-imaging-segmentation-877f3d20f350. [Accessed Oct. 3, 2025].

[4] Neptune.ai, “Performance Metrics in Machine Learning [Complete Guide]”, Neptune.ai, 2025. [Online]. Available: https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide. [Accessed Oct. 3, 2025]. 

[5] NHS Inform, “Tests for Diagnosing Heart Conditions", https://www.nhsinform.scot/, para. 3, Jan. 7, 2025. [Online]. Available: https://www.nhsinform.scot/tests-and-treatments/heart-tests/tests-for-diagnosing-heart-conditions/. [Accessed Oct. 3, 2025]. 

[6] Scikit-Learn, “Random Forests and Other Randomized Tree Ensembles”, Skikit-Learn, 2025. [Online]. Available: https://scikit-learn.org/stable/modules/ensemble.html#forest. [Accessed Oct. 3, 2025]. 

[7] Scikit-Learn, “GradientBoostingClassifier”, Skikit-Learn, 2025. [Online]. Available:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#gradientboostingclassifier. [Accessed Oct. 3, 2025]. 



