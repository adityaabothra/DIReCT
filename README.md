# DIReCT
# Overview
Algorithmic trading and artificial intelligence (AI) have transformed the field of intelligent trading. Deep reinforcement learning, we believe, has enormous potential in the finance sector of automated crypto trading. An RL agent, like a person, learns to act from its own experiences, attempting to acquire behaviour that maximises cumulative reward.

# Goal
Automate crypto trading by identifying optimal strategies guided by various technical indicators and price movement to make profit while minimizing losses in an all- day intra day market

# Background
Markov Decision Process (MDP)
Reinforcement Learning is based on MDP framework which is defined by the tuple:<S,A,R,P,𝛾>
● S: set of all states which configure the environment
● A: set of all possible actions an agent can take ● R: reward function
● P:statestransitionprobabilitydistribution
● 𝛾:discountfactorwithrange(0,1]

# Technical Indicators
Technical indicators are heuristic or pattern-based signals produced by the price, volume etc.
RSI:
The relative strength index (RSI) is a technical analysis indicator that measures the magnitude of recent price changes to determine whether a stock or other asset is
overbought or oversold
Bollinger Bands:
Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price.

# Environment
It is a space where the agent interacts, our trading environment was built from the ground up to meet OpenAI standards.
State Space:
• Stock prices, Volume Traded, Rolling Moving Average,RSI,BollingerBands,Networth, Units of Stock Held, Current Balance
Reward Metric:
• Reward: Units Held * Current Price + Balance • Cannot Trade: - Reward
• Net Worth > Initial Balance : Reward + 5
Action Space
• Action type : Buy, Sell, Hold
• Amount of stock to be traded : 0 - 100%

# Algorithm
Advantage Actor Critic(A2C)
Tricks:
• Randomise the start position instead of starting from the 1st day of the month in every episode to improve generalisation
• Prefer longer episodes, rather than training the agent for large number of episodes
• Sparse actor network and dense critic network

# Conclusion
Our agent converges at a profit range where it learns optimal strategy. It isn't always enticed by the highest profit, but it does pursue a strategy of accumulating profits over time. We wanted to teach the agent to be an intraday trader rather than a long-term investor, so holding for longer periods of time is not the best strategyforourrobot.

# References
1. https://sibanjandas.wordpress.com/2017/10/21/ reinforcement-learning-for-the-enterprise/
2. https://www.investopedia.com/terms/t/ technicalindicator.asp
3. https://www.fidelity.com/learning-center/trading-investing/ technical-analysis/technical-indicator-guide/bollinger-bands 4. https://medium.com/hackernoon/intuitive-rl-intro-to- advantage-actor-critic-a2c-4ff545978752

# TEAM3
Aditya Bothra
Vinay Kudari
