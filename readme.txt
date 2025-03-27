# Gradient Boosting Machine for Housing Price Prediction

This project implements **Gradient Boosting Regressor** to predict housing prices using the **Ames Housing dataset**. The goal is to build a predictive model, evaluate its performance, and visualize important features and actual vs. predicted values.

## Project Overview

The project uses **Gradient Boosting**, an ensemble technique, to predict housing prices based on various attributes such as neighborhood, square footage, number of rooms, and more. The model's performance is evaluated using metrics like **Mean Squared Error (MSE)** and **R-squared (R²)**. Key features contributing to the predictions are visualized using a **Feature Importance Plot**. Additionally, the relationship between actual and predicted values is displayed in a scatter plot.

### Key Steps:
1. **Data Loading**: Load the Ames Housing dataset using `fetch_openml()` from `sklearn.datasets`.
2. **Data Preprocessing**: Handle missing values, encode categorical features, and split the data into training and testing sets using `train_test_split()`.
3. **Model Training**: Train a **Gradient Boosting Regressor** model with 100 estimators and a learning rate of 0.1.
4. **Model Evaluation**: Evaluate the model using **Mean Squared Error (MSE)** and **R-squared (R²)**.
5. **Feature Importance Visualization**: Plot feature importances to identify which features are most significant in predicting housing prices.
6. **Actual vs Predicted Plot**: Visualize the relationship between actual and predicted housing prices.

## Features

- **Gradient Boosting Regressor**: An ensemble method for regression that builds trees sequentially.
- **Feature Importance Plot**: Displays which features are most influential in predicting housing prices.
- **Actual vs Predicted Plot**: Compares the model's predictions against the true values for evaluation.

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `xgboost`
  
You can install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## How to Run the Code

1. Clone this repository.
2. Install the necessary libraries (as mentioned above).
3. Run the Python script:

```bash
python gradient_boosting_housing.py
```

The script will:
- Load the Ames Housing dataset
- Preprocess the data and split it into training and testing sets
- Train the Gradient Boosting Regressor model
- Evaluate the model using MSE and R²
- Plot the feature importances and actual vs predicted values

## Results

The model’s performance is evaluated using:
- **Mean Squared Error (MSE)**: Measures prediction error; lower values indicate better performance.
- **R-squared (R²)**: Indicates the proportion of variance explained by the model; higher values represent better fit.

### Feature Importance Plot
The feature importance plot shows which features contribute most to the predictions. For this model, features like **OverallQual**, **GrLivArea**, and **TotRmsAbvGrd** are found to have significant influence on housing prices.

### Actual vs Predicted Plot
The scatter plot compares the model's predicted housing prices with the actual values. A good model should have points close to the red dashed line, representing perfect predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.