
# Stock-Market-Forecasting




## Abstract

The stock market forecasting is one of the most challenging application of machine learning, as its historical data are naturally noisy and unstable. Most of the successful approaches act in a supervised manner, labeling training data as being of positive or negative moments of the market. However, training machine learning classifiers in such a way may suffer from over-fitting, since the market behavior depends on several external factors like other markets trends, political events, etc. In this paper, we aim at minimizing such problems by proposing an ensemble of reinforcement learning approaches which do not use annotations (i.e., market goes up or down) to learn, but rather learn how to maximize a return function over the training stage. In order to achieve this goal, we exploit a Q-learning agent trained several times with the same training data and investigate its ensemble behavior in important real-world stock markets. Experimental results in intraday trading indicate better performance than the conventional Buy-and-Hold strategy, which still behaves well in our setups. We also discuss qualitative and quantitative analyses of these results.




## Authors

- Anselmo Ferreira 
- Alessandro Sebastian Podda
- Diego Reforgiato Recupero
- Salvatore Carta
- Antonio Sanna

## INFO
- data.csv:-  It is a dataset.
- code.py:- It is the deep learning model







## Description
Ensemble of Deep Q-Learning Agents for Stock Market Forecasting:
- The code implements an ensemble of Deep Q-Learning agents to forecast stock market signals (buy, sell, none).
- It utilizes a dataset containing stock market data with features like open, high, low, close prices, and datetime information.
- The dataset is preprocessed by encoding the target variable 'signal' into numerical labels and splitting it into training and testing sets.
- Standard scaling is applied to the features for normalization.
- Multiple Deep Q-Learning agents are initialized and trained with a specified number of episodes and batch size.
- The agents learn to predict buy and sell signals based on the input features.
- The BTC close price along with the predicted buy and sell signals are plotted for all years in the dataset.
- Precision and accuracy metrics are calculated to evaluate the performance of the ensemble of agents.
- The code provides a comprehensive framework for building and evaluating deep reinforcement learning models for stock market forecasting. 

## Screenshots

![main](https://github.com/Ayush-chanchal/stock-market-forecasting/assets/103252150/e9ce2c30-1609-4978-b100-af783708bafb)


## Features Extraction

- The dataset is loaded from the 'data.csv' file.
- The datetime column is converted to the correct format for further processing.
- The features (X) are extracted by dropping the 'signal' and 'datetime' columns from the dataset.
- The target variable 'signal' is encoded into numerical labels (0 for 'buy', 1 for 'sell', 2 for 'none').
- The dataset is split into training and testing sets.
- Features are scaled using StandardScaler() to normalize the data.

### Code
data = pd.read_csv('data.csv')

data['datetime'] = pd.to_datetime(data['datetime'])

X = data.drop(['signal', 'datetime'], axis=1)  # Exclude 'datetime' for model training
y = data['signal'].map({'buy': 0, 'sell': 1, 'none': 2}).astype(int)  # Encoding signals

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


## Summary
### Objective
Ensemble of Deep Q-Learning Agents for Stock Market Forecasting.
### Deep Learning Algorithm
Deep Q-Learning (DQN).
### Purpose
 The model aims to forecast stock market signals (buy, sell, none) based on historical data. It utilizes an ensemble of Deep Q-Learning agents to learn and predict the optimal actions to take in the stock market.
### Input
The input data consists of historical stock market features such as open, high, low, close prices, and datetime information.
### Feature Extraction
Preprocessing steps include encoding the target variable ('signal') into numerical labels, splitting the dataset into training and testing sets, and standard scaling of features for normalization.
### Model Training
Multiple Deep Q-Learning agents are initialized and trained with a specified number of episodes and batch size. The agents learn to predict buy and sell signals based on the input features.
### Output
The output includes:
- Visualization: BTC close price plotted with predicted buy and sell signals for all years in the dataset.
- Evaluation Metrics: Precision and accuracy scores to evaluate the performance of the ensemble of agents in forecasting stock market signals.
