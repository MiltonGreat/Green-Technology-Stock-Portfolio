# Stock Price Movement Prediction

## Overview

This project aims to predict the direction of stock price movements (up or down) for the next day using machine learning techniques. It involves collecting historical stock data, engineering features, building and evaluating predictive models, and backtesting the strategy to assess its performance in a simulated trading environment.

### Project Goals

1. Predict Stock Price Movements:
- Predict whether the stock price will go up (1) or down (0) the next day.

2. Evaluate Model Performance:
- Compare the performance of different machine learning models (e.g., Logistic Regression, Random Forest, LSTM).

3. Backtest the Strategy:
- Simulate how the model would perform in a real-world trading environment.

4. Make Predictions:
- Predict the next day's price movement for a given set of stocks.

### Project Approach

1. Data Collection
- Fetch historical stock data using the yfinance library.
- Combine data for multiple stocks into a single DataFrame.

2. Data Preprocessing
- Clean and restructure the data.
- Add technical indicators (e.g., moving averages, RSI, MACD).
- Create a target variable for the next day's price movement.

3. Model Building
- Split the data into training and testing sets.

Train and evaluate the following models:
- Logistic Regression
- Random Forest
- LSTM

4. Model Evaluation
- Evaluate models using accuracy, F1-score, and confusion matrices.
- Perform hyperparameter tuning using GridSearchCV.

5. Backtesting
- Simulate the performance of the model in a trading environment.
- Compare strategy returns with market returns.

6. Prediction
- Predict the next day's price movement for a given set of stocks.

### Results

Logistic Regression:
- Accuracy: 52.02%
- F1-Score: 65.33%

Random Forest:
- Accuracy: 49.87%
- F1-Score: 52.85%

LSTM:
- Accuracy: 52.71%
- F1-Score: 64.11%

Backtesting
- The backtesting results show the cumulative returns of the strategy compared to the market returns.

Predictions
- The model predicts the next day's price movement for a given set of stocks.

### Future Work

1. Improve Feature Engineering:
- Add more relevant features (e.g., sentiment analysis, macroeconomic indicators).

2. Handle Class Imbalance:
- Use techniques like oversampling, undersampling, or class weights.

3. Try Advanced Models:
- Experiment with more sophisticated models like Gradient Boosting (e.g., XGBoost, LightGBM).

4. Hyperparameter Tuning:
- Optimize hyperparameters for the existing models.

5. Evaluate Data Quality:
- Check for missing values, outliers, or noisy data.

### Conclusion

This project demonstrates the challenges of predicting stock price movements using machine learning. While the models did not achieve the desired performance, the project provides valuable insights into the process of data collection, feature engineering, model building, and evaluation.
