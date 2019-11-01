import os
import numpy as np
import pandas as pd
import json
import urllib

def synthetic_coin_generator(probabilities, breakpoint, k):
    """
    A generator for binomial random variables which are governed by a mixture model
    
    Args:
        probabilities (Tuple): A touple of success probabilities p1 and p2,
        which are the parameters for a binomial mixture model
        breakpoint (int): The index at which the probability regime changes from p1 to p2
        k (int): The length of the generated series
    """
    
    if breakpoint <= 0 or breakpoint >= k:
        raise Exception("Breakpoint must be defined between {} and {}".format(1, k - 1))
    
    for i in range(k):
        yield (np.random.random() <= probabilities[i >= breakpoint])*1

        
class StockDataGenerator(object):
    """
    A lighweight wrapper around Alpha Vantage's marked data API (https://www.alphavantage.co/) 
    to pull a Series of daily closing prices for a particular symbol within a given date range
    """
    
    def __init__(self, symbol):
        """
        Initializes the Stock Data Generator by pulling all available data for a given symbol 
        from Alpha Vantage's TIME_SERIES_DAILY endpoint and parsing it into a data frame
        
        Make sure to export 'API_KEY' as an environment variable before invoking this function
        
        Args:
            symbol (str): Ticker symbol of an asset
            
        Returns:
            None
        """
        
        self.symbol = symbol
        request_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&"\
                      "apikey={}&outputsize=full".format(symbol, os.environ['API_KEY'])
        
        with urllib.request.urlopen(request_url) as url:
            raw_data = json.loads(url.read())
            try:
                raw_data = pd.DataFrame.from_dict(raw_data['Time Series (Daily)']).T
            except KeyError:
                raise Warning("Data for symbol {} is not available".format(symbol))
            
            raw_data.columns = ["open","high","low","close","volume"]
            raw_data["date"] = [i.date() for i in pd.to_datetime(raw_data.index)]
            self.max_date, self.min_date = raw_data.date.iloc[[0,-1]]
            self.stock_data = raw_data
    
    def closing_prices(self, start_date, end_date, real=False):
        """
        Fetches the closing prices of a particular symbol wihin a given time range
        
        Args:
            start_date (str): The earliest date for which to pull the closing prices (format: YYYY-MM-DD)
            end_date (str): The latest date for which to pull the closing prices (format: YYYY-MM-DD)
            real (bool): Determines whether the actual closing prices (real=True) are returnd or 
            a series of binary variables, indicating upward/downward price movement (defaults to False)
        
        Returns:
            closing_prices (Iterable): Either real or boolean values
            data_labels (List): A list of dates correspoinding to the closing prices
        """
        
        data_subset = self.stock_data.loc[(self.stock_data.date>=np.datetime64(start_date)) & 
                                   (self.stock_data.date<=np.datetime64(end_date)),:].sort_values(by="date")
        data_dabels = pd.to_datetime([str(i) for i in data_subset.date])
        
        if real:
            return data_subset.close.astype(float), data_dabels
        
        return ([int(data_subset.close.iloc[i]>data_subset.close.iloc[i-1]) 
                for i in range(1,data_subset.shape[0])], data_dabels)