# Using MDL to detect Regime Changes in Stock Price Movements

## Summary
Originally, this project was a Homework Project for _CME240 - Statistical and Machine Learning Approaches to Problems in Investment Management_, taught by Jeremy Evnine at the [Stanford Institute for Computational & Mathematical Engineering (ICME)](https://icme.stanford.edu/) in Spring 2019. 

The goal for this little exercise is to **detect regime changes** in stock price movements using the concept of **Minimum Description Lenght (MDL)**. The underlying approach is to model up- and downward movements of closing prices as a **binary random variable**, propose a **binomial mixture model** for said variable, and determine a breakpoint in the observed time series that maximizes the quality of the fit. 

In a **stylized experiment on simulated data from a biased coin**, the model picks up a regime change with a latency of 5 periods. Applied to **S&P 500 closing prices between 2008 and 2009**, the model suggests a regime change around June 12th, 2008 - just about two months before the financial meltdown coupled in Lehman Brother's collapse.

## How to run the jupyter notebook
All code for this project is written in **python 3.6**. Stock data is fetched from [Alpha Vantage's public API](https://www.alphavantage.co/)

```bash
# clone the git repository
git clone https://github.com/bummy93/regime-change-detection.git & cd regime-change-detection

# install all dependencies
pip install -r requirements.txt
```
