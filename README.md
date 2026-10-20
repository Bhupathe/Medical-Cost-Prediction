# Medical Insurance Cost Prediction

## 1. Project Overview

The **Medical Insurance Cost Prediction** project uses Machine Learning regression techniques to estimate individual medical insurance charges based on personal, demographic, and lifestyle-related information.

The project explores relationships between age, BMI, smoking status, number of children, gender, and residential region and the medical insurance charges recorded for individuals.

Multiple regression techniques are evaluated to identify a suitable predictive model. The selected model is integrated into a Flask web application, allowing users to enter their details and receive an estimated insurance cost.

## 2. Project Objectives

- Analyze and understand the medical insurance dataset through Exploratory Data Analysis (EDA).
- Identify patterns and relationships between individual characteristics and insurance charges.
- Prepare numerical and categorical features for Machine Learning.
- Train and evaluate multiple regression models.
- Compare model performance using R², MAE, MSE, and RMSE.
- Select a suitable model based on its predictive performance.
- Deploy the selected model through a Flask application.

## 3. Prediction Objective

This is a **supervised Machine Learning regression problem**.

The target variable is `charges`, representing the medical insurance cost recorded for an individual.

The model learns from historical data to estimate insurance charges for new inputs.

### Input features

| Feature | Description |
|---|---|
| `age` | Age of the individual in years |
| `sex` | Gender category |
| `bmi` | Body Mass Index |
| `children` | Number of children or dependents covered |
| `smoker` | Smoking status: yes or no |
| `region` | Residential region |
| `charges` | Target variable: medical insurance charges |

The prediction is an estimate, not a guaranteed insurance quote.

## 4. Dataset Description

The project uses the medical insurance cost dataset commonly distributed through Kaggle.

- **Records:** 1,338
- **Columns:** 7
- **Input features:** 6
- **Target variable:** 1
- **Missing values:** None identified during the initial analysis.

The dataset contains numerical features such as age and BMI, and categorical features such as sex, smoking status, and region.

## 5. Exploratory Data Analysis (EDA)

### 5.1 Key observations

**Smoking status**
- Smokers: 274 (approximately 20.48%).
- Non-smokers: 1,064 (approximately 79.52%).
- Average charges for smokers: approximately $32,050.
- Average charges for non-smokers: approximately $8,434.

Smoking status is strongly associated with differences in average insurance charges in this dataset.

**Age**
- Age range: 18–64 years.
- Average age: approximately 39.21 years.

**BMI**
- The IQR-based analysis identified 9 potential BMI outliers.

**Gender and region**
- Gender distribution is broadly balanced.
- Average charges vary across the four regions.

**Insurance charge outliers**
- The IQR-based analysis identified 139 charge observations above the upper bound.
- Of these, 136 belonged to smokers and 3 to non-smokers.
- These observations may be genuine high costs rather than incorrect data.

### 5.2 Data preparation

- Gender encoding: `male = 0`, `female = 1`.
- Smoking encoding: `no = 0`, `yes = 1`.
- Region encoding: one-hot encoding with Southwest as the reference category.
- Training/testing split: 75% training and 25% testing.
- Random state: `42`.
- Training records: 1,003.
- Testing records: 335.
- Polynomial feature expansion: degrees 2, 3, 4, and 5.

High insurance charges should not automatically be removed or clipped, since they may contain important predictive information.

## 6. Machine Learning Models Evaluated

The `model_without_scalar.ipynb` notebook evaluates the following models:

1. Multiple Linear Regression
2. Polynomial Regression — Degree 2
3. Polynomial Regression — Degree 3
4. Polynomial Regression — Degree 4
5. Polynomial Regression — Degree 5
6. Lasso Regression
7. Ridge Regression
8. Elastic Net Regression

### Evaluation metrics

- **R² score:** Measures how much variation in the target is explained by the model. Higher is better.
- **MAE (Mean Absolute Error):** Average absolute prediction error. Lower is better.
- **MSE (Mean Squared Error):** Average squared prediction error. Lower is better.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the same units as the target. Lower is better.

## 7. Model Evaluation Results

The following values are taken from the recorded outputs in `model_without_scalar.ipynb`. All models use the notebook's test set.

| Model | R² Score | MAE | MSE | RMSE |
|---|---:|---:|---:|---:|
| Multiple Linear Regression | 0.767264 | 4,243.6541 | 35,117,755.7361 | 5,926.0236 |
| Polynomial Regression — Degree 2 | **0.854259** | **2,781.7102** | **21,990,968.1100** | **4,689.4529** |
| Polynomial Regression — Degree 3 | 0.839427 | 2,965.0862 | 24,229,055.0186 | 4,922.3018 |
| Polynomial Regression — Degree 4 | 0.717660 | 3,731.8382 | 42,602,570.5418 | 6,527.0645 |
| Polynomial Regression — Degree 5 | -0.119102 | 5,558.1456 | 168,862,648.6829 | 12,994.7162 |
| Lasso Regression | 0.767259 | 4,243.7174 | 35,118,565.4520 | 5,926.0919 |
| Ridge Regression | 0.766982 | 4,256.5435 | 35,160,344.2537 | 5,929.6159 |
| Elastic Net Regression | 0.397942 | 7,446.3115 | 90,845,289.8755 | 9,531.2796 |

*MAE, MSE, and RMSE are expressed in the dataset's charge units. The notebook does not explicitly document a currency conversion.*

### 7.1 Interpretation of the results

**Multiple Linear Regression**

The baseline model achieved an R² score of approximately 0.7673, with an MAE of 4,243.65 and an RMSE of 5,926.02.

**Polynomial Regression — Degree 2**

This model achieved the strongest overall recorded results:

- R²: **0.854259**
- MAE: **2,781.71**
- MSE: **21,990,968.11**
- RMSE: **4,689.45**

Compared with Multiple Linear Regression, the degree-2 model improved R² and reduced both MAE and RMSE.

**Polynomial Regression — Degree 3**

Degree 3 achieved an R² of 0.839427. Its results were better than the linear baseline on these metrics, but not as strong as degree 2.

**Polynomial Regression — Degrees 4 and 5**

Performance deteriorated at higher degrees. Degree 4 achieved an R² of 0.717660, while degree 5 produced a negative R² of -0.119102, indicating poor performance on the test set.

**Lasso and Ridge Regression**

Both regularized models produced results close to Multiple Linear Regression. Neither outperformed the degree-2 polynomial model in the recorded evaluation.

**Elastic Net Regression**

Elastic Net achieved an R² of approximately 0.397942, with higher MAE and RMSE than the other models in this notebook. It was not the best-performing technique in this experiment.

## 8. Selected Model

**Selected model: Polynomial Regression — Degree 2**

Degree-2 Polynomial Regression was selected because it achieved the highest recorded R² score and the lowest MAE, MSE, and RMSE among the models evaluated in `model_without_scalar.ipynb`.

Its performance suggests that polynomial feature expansion captures useful nonlinear relationships in the data that the baseline linear model does not represent as effectively.

Higher polynomial degrees did not improve the results, so degree 2 provides the strongest recorded balance of model flexibility and predictive performance.

### Important evaluation note

The adjusted R² values printed for polynomial models in the notebook are calculated using the original feature count rather than the expanded polynomial feature count. Therefore, those adjusted R² values should not be used to compare polynomial degrees without recalculation. The R², MAE, MSE, and RMSE values in the table above are the metrics directly reported by the notebook.

The model selection conclusion is based on the recorded experiments. For a rigorous final comparison, each candidate should be evaluated using an identical preprocessing and validation pipeline.

## 9. Flask Web Application

The selected model is integrated into a Flask application.

The prediction workflow is:

1. The user enters age, gender, BMI, number of children, smoking status, and region.
2. The application encodes the input values.
3. The features are arranged in the same order used during training.
4. The saved polynomial transformer expands the features.
5. The trained regression model predicts the insurance charge.
6. The predicted value is displayed to the user.

The transformer and regression model must be loaded together and applied in the same order used during training.

## 10. Project Limitations

- The dataset contains 1,338 records and may not represent all populations or insurance pricing policies.
- Individual predictions can differ from actual charges even when overall model metrics are good.
- Observed associations do not establish causation.
- High-charge observations may strongly affect model performance and should be handled carefully.
- The reported results reflect the experiments in this project and are not a guarantee of performance on other datasets.
- Predictions are intended for educational and analytical purposes, not as official insurance quotes or medical advice.

## 11. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Flask
- HTML and CSS
- Pickle for saving and loading model components

## 12. Project Workflow

1. Load the insurance dataset.
2. Inspect data quality and summarize features.
3. Perform Exploratory Data Analysis.
4. Encode categorical variables.
5. Split the dataset into training and testing sets.
6. Train multiple regression models.
7. Evaluate models using R², MAE, MSE, and RMSE.
8. Select degree-2 Polynomial Regression based on recorded results.
9. Save the model and polynomial transformer.
10. Integrate the saved components into the Flask application.

## 13. Conclusion

This project demonstrates an end-to-end Machine Learning workflow for predicting medical insurance charges, from dataset exploration and preprocessing to model evaluation and web deployment.

Eight model configurations were evaluated using the recorded test-set results. **Polynomial Regression of degree 2** performed best, achieving an R² score of approximately **0.8543**, an MAE of **2,781.71**, an MSE of **21,990,968.11**, and an RMSE of **4,689.45**.

Based on these results, degree-2 Polynomial Regression was selected for the Flask prediction application. It provides a stronger fit to the evaluation data than the other recorded models, although individual predictions remain estimates and can contain substantial errors.
