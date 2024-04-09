
# stock-market-forecasting




## Abstract

[The stock market forecasting is one of the most challenging application of machine learning, as its historical data are naturally noisy and unstable. Most of the successful approaches act in a supervised manner, labeling training data as being of positive or negative moments of the market. However, training machine learning classifiers in such a way may suffer from over-fitting, since the market behavior depends on several external factors like other markets trends, political events, etc. In this paper, we aim at minimizing such problems by proposing an ensemble of reinforcement learning approaches which do not use annotations (i.e., market goes up or down) to learn, but rather learn how to maximize a return function over the training stage. In order to achieve this goal, we exploit a Q-learning agent trained several times with the same training data and investigate its ensemble behavior in important real-world stock markets. Experimental results in intraday trading indicate better performance than the conventional Buy-and-Hold strategy, which still behaves well in our setups. We also discuss qualitative and quantitative analyses of these results.](https://linktodocumentation)




## Authors

- Anselmo Ferreira 
- Alessandro Sebastian Podda
- Diego Reforgiato Recupero
- Salvatore Carta
- Antonio Sanna

## Description
- data.csv:-  It is a dataset.
- code.py:- It is the deep learning model
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

![App Screenshot](https://drive.google.com/file/d/1llh7gLAtZN5LdSy7gKPXdAZymuXp1a64/view?usp=sharing)

