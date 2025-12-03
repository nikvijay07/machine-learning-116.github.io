# Heart Failure Detection

ML Final Project Proposal<br />
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
___
## Models

### Logistic Regression
#### Method

The first step in creating our model was to preprocess our data. Initially, we checked for duplicates and found none. Following that, we wanted to convert qualitative data into numerical data so it was usable by our model. We used one hot encoding here to convert these features because they had relatively low cardinality and this technique works well with logistic regression. The next step was to perform outlier detection and removal. By visualizing our data with box plots for each feature, we were able to see our distributions and point out any strong outliers. In many instances, we saw individuals with a cholesterol level of 0 or resting blood pressure of 0, which is not biologically possible, making the data recording impractical. However, we did decide to keep most outliers as they were valid data points and represented people’s true data. For the biologically impossible data points, we utilized K-nearest-neighbors with K = 5 and averaged their values for that data point. We used 4 features in order to avoid the curse of dimensionality: Cholesterol, RestingBP, Age, and MaxHR. These features were also selected because their values were continuous integers which were easier to compute distance from. Below is an example of the cholesterol data before and after preprocessing.

<img width="523" height="300" alt="Screenshot 2025-11-07 at 2 11 25 PM" src="https://github.com/user-attachments/assets/1b76de14-79a9-4377-bead-11b56e5fb261" />

Figure 1: Before KNN-Imputation


<img width="523" height="300" alt="Screenshot 2025-11-07 at 2 11 38 PM" src="https://github.com/user-attachments/assets/86c2c959-ca5c-4ca0-bbc3-8d3797ecdc73" />

Figure 2: After KNN-Imputation

Finally, we decided to perform standardization on a few features to make sure that the features are on the same scale and therefore weighted similarly. 

We chose a logistic regression model to predict based on our data because it is applicable to binary classification, and is a good application for our case where we do not have a huge number of data points. To adjust for overfitting, we  tested with both ridge and lasso regularization. some features weights to go to 0 if they were not relevant. During this process, we found that ​​RestingBP, Cholesterol, and RestingECG_Normal all ended up with coefficients of 0. 

To extend our analysis beyond logistic regression, we also trained and evaluated two additional supervised models: Random Forest and XGBoost (Gradient Boosting). These models were applied to the same preprocessed dataset, ensuring a consistent comparison across approaches.


### Random Forest
#### Method
For our second model, we chose to use a random forest. Random forests tend to perform well on classification tasks, and the variety of available hyperparameters gives us flexibility to tune the model for optimal performance on our data. Since we are tuning our data using cross fold validation, we also made sure to undo the standardization we performed during our preprocessing stage. This is because we did not want each training fold to contain data about the entire distribution of the data as that is counterintuitive to cross fold training. This could lead to slight overfitting as there is more information than necessary for this stage. To combat this, we standardized within each training set, so the training would only possess information about the distribution of the training data. Another improvement we made was to use grid search rather than nested for loops to tune our hyperparameters for optimized speed. This is because grid search is able to run models in parallel rather than sequentially as loops would use. 

<img width="751" height="436" alt="Screenshot 2025-12-02 at 4 26 11 PM" src="https://github.com/user-attachments/assets/075497a4-4f9a-4f92-8335-1a3cb89a9130" />
In this model, the importance for each feature is different from logistic regression. This can be attributed to the fact that logistic regression gives more weights to features with linear relationships between the features and results whereas random forest uses more of a threshold relationship (eg. Age > 60). Random Forest also does not completely eliminate features like lasso regularization might. Overall, the recall was slightly higher than before, however it came with a tradeoff of accuracy. 

On a positive note, the general limitations of random forest such as compute time and memory did not seem to affect us drastically as we were able to have enough time to tune and run our models with multiple different hyperparameters. 
More results will be discussed in the later sections. 

### XGBoost (Gradient Boosted Trees)
#### Methods

For our final model, we used extreme gradient boosted decision trees. We chose this method to compare its performance against our classic random forest model and to get a better understanding of how this type of boosting algorithm works. As opposed to random forest, this type of algorithm creates trees that sequentially improve on the error of the previous tree. For binary classification, each tree outputs a residual value which is the negative gradient of the loss function (ri = yi - pi). Where pi equals the predicted probability of the datapoint being in that classification and yi = the true label of that datapoint. Each iteration of trees outputs a logit correction which is calculated using the residuals and when testing, the sum of all residuals of the trees gives the final predicted probability of that datapoint. Our hyperparameters for this model were the following:

* number of trees
* max depth of trees
* learning rate
* number of data points per tree
* number of features per tree
* regularization lambda
* gamma (minimum loss reduction required to split node)
* minimum child weight (minimum amount of information per node before splitting)

## Results and Discussion

### Logistic Regression
Here are the core metrics for L1 vs L2 regularization on our logistic regression model:

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

For a medical model, we hoped that our false negative results would be lower as that is probably the most important statistic of our data. We would much rather overdiagnose positives that underdiagnose negatives and this is also  captured by our recall score being lower than our precision. We also noticed that ridge regularization barely improved results (only by 1 test point) which signifies that the features which were signified as irrelevant by lasso regularization did not have much weight on the decision to begin with. This is also shown in the following image as it displays the coefficients of each feature. 

<img width="944" height="545" alt="Screenshot 2025-11-07 at 4 28 21 PM" src="https://github.com/user-attachments/assets/bfaf5e85-ca0e-4841-8875-f690a95b6ef4" />
<img width="940" height="542" alt="Screenshot 2025-11-07 at 4 28 30 PM" src="https://github.com/user-attachments/assets/61b4d119-4490-46e7-b3a7-ba86e9c38324" />

Overall, we think the model performed decently well with our data, and I think we can attribute it to the data being standardized and generally low noise. Also, since the model is predicting the classification based on a linear combination of features, it seems that some features do have a fairly linear relationship with the outcome which allows us to produce a decently high accuracy rate. 

#### Why the model performed as well as it did
* Pre-processing: One-hot encoding of categorical values to reduce noise, and imputation using KNN-informed values (rather than randomized or simple mean-based imputation)
* Standardization: StandardScaler to standardize features allows for more stabilized convergence and precludes arbitrarily scaled values (like heartrate being double digit but cholesterol being in hundreds) from dominating other features, especially during L1/L2 regularization
* Linearity: The assumption of linear classification under logistic regression seemed to hold, as features like Age and MaxBP intuitively would be linearly correlated with heart disease

#### Limitations
* Model limitations: while linearity of decision boundaries seemed to hold up, it remains a limitation of the model as the relationships are not truly going to be linear; nonlinear or interacting factors may still be relevant
* Recall–precision tradeoff: The model’s conservative bias toward healthy predictions (not sacrificing precision enough to increase recall) reduces safety for clinical screening, where we want higher recall even at the cost of precision


### Random Forest

We next trained a Random Forest classifier on the same preprocessed dataset to capture non-linear relationships and feature interactions. A grid search was conducted over several hyperparameters, including the number of trees (200, 400, 600, 800), maximum tree depth (None, 8, 12, 16, 20), minimum samples required to split an internal node (2, 4, 6, 10), feature subsampling strategies (sqrt or log2), and class weighting (None or Balanced). To prevent data leakage, standardization was applied within each training fold only.

On the test set, the best Random Forest achieved the following performance:

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Accuracy</td><td>0.859</td></tr>
  <tr><td>True Positive (TP)</td><td>93</td></tr>
  <tr><td>False Positive (FP)</td><td>11</td></tr>
  <tr><td>True Negative (TN)</td><td>66</td></tr>
  <tr><td>False Negative (FN)</td><td>14</td></tr>
  <tr><td>Precision</td><td>0.89</td></tr>
  <tr><td>Recall</td><td>0.87</td></tr>
  <tr><td>F1 Score</td><td>0.88</td></tr>
</table>

The confusion matrix and feature importance plot show that the model identifies a large majority of positive heart-disease cases while still maintaining reasonable specificity. Features related to ST segment slope (such as ST_Slope_Up and ST_Slope_Flat), along with ischemia indicators like Oldpeak and ExerciseInducedAngina, emerged as the most informative predictors. Other features such as MaxHR and Chest Pain Type also contributed meaningfully.

#### Why the model performed as well as it did

* Ability to model non-linear and interacting features, which aligns with the complex nature of cardiovascular risk factors.
* The ensembling effect of averaging many decision trees reduces model variance and helps avoid overfitting.
* A balanced feature importance profile indicates robustness and avoids over-reliance on any single variable.

#### Limitations

* The model achieves high recall but slightly lower specificity, leading to more false positives compared to logistic regression.
* Reduced interpretability relative to linear models, since the decision paths of many trees are difficult to trace.
* Higher computational cost compared to logistic regression, especially during hyperparameter tuning.


### XGBoost

We also trained an XGBoost (gradient boosting) classifier to capture more complex patterns by iteratively correcting errors from previous trees. Trees are built sequentially, with each tree improving upon the residuals of the last, and logistic loss is used as the objective function. Using 5-fold cross-validation and random search, we identified a high-performing configuration with a small tree depth of 2, a learning rate of 0.027, regularization strength lambda of 1.52, and subsampling fractions of 0.85 for features and 0.87 for samples.

XGBoost achieved the following performance on the test set:

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Accuracy</td><td>0.850</td></tr>
  <tr><td>True Positive (TP)</td><td>91</td></tr>
  <tr><td>False Positive (FP)</td><td>12</td></tr>
  <tr><td>True Negative (TN)</td><td>65</td></tr>
  <tr><td>False Negative (FN)</td><td>16</td></tr>
  <tr><td>Precision</td><td>0.88</td></tr>
  <tr><td>Recall</td><td>0.85</td></tr>
  <tr><td>F1 Score</td><td>0.87</td></tr>
</table>


Its feature importance plot shows that ST_Slope_Up dominates the model’s decisions, followed by other ischemia-related and chest-pain features. The steep dropoff after the top few features suggests that XGBoost concentrates its predictive power in a smaller subset of variables compared to Random Forest.

<img width="989" height="590" alt="xgboostsignificance" src="https://github.com/user-attachments/assets/60922fed-50f0-47f8-b0db-255296d25098" />

#### Why the model performed as well as it did

* Gradient boosting can capture subtle, difficult-to-learn patterns by repeatedly correcting mistakes from earlier trees.
* The low learning rate, shallow trees, and regularization terms help prevent overfitting while maintaining strong predictive power.
* XGBoost obtained the highest ROC AUC of all models, indicating the strongest ability to rank patients by risk rather than simply classify them.

#### Limitations

* Slightly lower recall than Random Forest, which leads to more false negatives than the best-performing model.
* Sensitive to hyperparameter choices, requiring more computation and tuning effort.
* Less interpretable than logistic regression due to the sequential boosted-tree architecture.

## Comparison

When comparing the three models, several performance tradeoffs become clear from the visualizations. The bar chart comparing accuracy, recall, and F1 score shows that all three models perform similarly in overall accuracy, with logistic regression slightly ahead, but they diverge more meaningfully on recall. Random Forest achieves the highest recall, as shown in the middle bar grouping, which aligns with the confusion matrix results showing fewer false negatives. This makes Random Forest the safest model for medical screening, where missing positive cases is the most costly error. In contrast, logistic regression achieves strong accuracy and F1 score but noticeably underperforms in recall, reflecting its tendency to produce more false negatives. XGBoost sits in the middle: it is more balanced across metrics than logistic regression and performs slightly below Random Forest in recall, but it offers consistent strength across all three evaluation categories.
<img width="790" height="490" alt="metriccomparisons" src="https://github.com/user-attachments/assets/b1359cda-5596-4409-8efd-bea979550f4b" />

The ROC curves reinforce this interpretation. Logistic regression and XGBoost achieve the highest AUC values (0.921 and 0.926, respectively), indicating strong ranking ability and strong sensitivity-specificity tradeoffs across thresholds. Random Forest, while achieving the highest recall, has a slightly lower AUC of 0.918, suggesting that its overall ranking ability is marginally weaker even though it captures more positives at the default threshold. The feature importance comparison shows additional differences across models: logistic regression spreads importance more evenly, Random Forest displays balanced but more sharply defined importance values, and XGBoost exhibits the steepest drop-off, strongly emphasizing only a few features such as ST_Slope_Up and ExerciseInducedAngina. Together, these comparisons show that Random Forest is best suited for high-recall medical screening, logistic regression offers simplicity with limitations in safety due to false negatives, and XGBoost provides the strongest overall ranking performance with balanced predictive behavior.
<img width="690" height="590" alt="roc" src="https://github.com/user-attachments/assets/445e711d-4c8c-4ef3-9a15-1b7a2986425e" />

Across the three models, we observe a tradeoff between interpretability, recall, and overall predictive accuracy. Logistic regression is the easiest to interpret but suffers from higher false negatives; Random Forest offers the strongest recall and therefore the greatest safety in a clinical screening setting; and XGBoost delivers the best AUC and the most balanced set of predictive metrics. 

## Next Steps
Building on these insights, several next steps are appropriate:
* If we wanted to do some continued work on the models, we could introduce an ensemble approach where the models work together.
* It would be cool to show some of our findings to researchers of cardiovascular health or doctors in the field.
* Finish documenting and turn in our final results.



## Contribution Table and Gantt Chart
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




