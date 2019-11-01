# Using MDL to detect Regime Changes in Stock Price Movements

Originally, this project was a Homework Project for _CME240 - Statistical and Machine Learning Approaches to Problems in Investment Management_, taught by Jeremy Evnine at the [Stanford Institute for Computational & Mathematical Engineering (ICME)](https://icme.stanford.edu/) in Spring 2019. 

The goal is to **detect regime changes in stock price movements**. The approach is to model the underlying up- and downward movements as a **binary random variable**, propose a **binomial mixture model** and determine the breakpoint in the observed time series that maximizes the quality of the fit. In a **stylized experiment on data from a simulated coin**, the model picks up a regime change with a latency of 5 periods. Applied to the **S&P 500 between 2008 and 2009**, the model suggests a regime change around June 12th, 2008 - just about two months before the financial meltdown coupled in Lehman Brother's collapse.

## How to run this notebook
The code for this project is written in **python 3.6**. Stock data is fetched from [Alpha Vantage's public API](https://www.alphavantage.co/)

```bash
# clone the git repository
git clone https://github.com/bummy93/regime-change-detection.git & cd regime-change-detection

# install all dependencies
pip install -r requirements.txt
```
