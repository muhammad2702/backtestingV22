# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:01:50 2024

@author: atom
"""

from twelvedata import TDClient
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from bokeh.layouts import gridplot

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

st.title(" Backtest (QQQ) Release V.22 " )


    
cashval = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000)
thresholdval = st.number_input(" Select Threshold Value:", min_value=1, max_value=4, value=1, step=1)
slval = float(st.number_input("Stop Loss above/below Current Price", min_value=0.1, value=4.0, step=0.1))
tpval = float(st.number_input("Take Profit above/below Current Price", min_value=0.1, value=4.0, step=0.1))

    

class DataFeeder:
    def __init__(self, ticker, api_key, timeframe):
        self.ticker = ticker
        self.api_key = api_key
        self.timeframe = timeframe
        self.td = TDClient(apikey=self.api_key)
        try:
            self.df = self.fetch_ticker()
            print(f"Data fetched for {self.ticker}")
        except Exception as e:
            print("nope", e)

    def fetch_ticker(self):
        ts = self.td.time_series(
            symbol=self.ticker,
            interval=self.timeframe,
            outputsize=5000,
            timezone="America/New_York",
        )
        df = ts.as_pandas()
        df = df.reset_index()
        df.rename(columns={'datetime': 'date'}, inplace=True)
        
        # Sort the DataFrame by date in ascending order
        df['date'] = pd.to_datetime(df['date'])  # Ensure date column is in datetime format
        df.sort_values(by='date', inplace=True)  # Sort by date
        
        return df
    
class DataFeederML:
    def __init__(self, ticker, api_key, timeframe, output):
        self.ticker = ticker
        self.api_key = api_key
        self.timeframe = timeframe
        self.td = TDClient(apikey=self.api_key)
        self.out = output
        try:
            self.df = self.fetch_ticker()
            print(f"Data fetched for {self.ticker}")
        except Exception as e:
            print("nope", e)

    def fetch_ticker(self):
        ts = self.td.time_series(
            symbol=self.ticker,
            interval=self.timeframe,
            outputsize=self.out,
            timezone="America/New_York",
        )
        df = ts.as_pandas()
        df = df.reset_index()
        df.rename(columns={'datetime': 'date'}, inplace=True)
        return df

class DataFeeder2:
    def __init__(self, ticker, api_key, timeframe='2min'):
        if timeframe != '2min':
            raise ValueError("This class only supports a 2-minute timeframe.")
        
        self.ticker = ticker
        self.api_key = api_key
        self.timeframe = '1min'  # Use 1-minute intervals to fetch data
        self.td = TDClient(apikey=self.api_key)
        
        try:
            self.df = self.fetch_and_aggregate_data()
            print(f"Data fetched and aggregated for {self.ticker}")
        except Exception as e:
            print("Data fetching failed:", e)

    def fetch_and_aggregate_data(self):
        # Fetch 1-minute interval data
        ts = self.td.time_series(
            symbol=self.ticker,
            interval=self.timeframe,
            outputsize=5000,
            timezone="America/New_York",
        )
        df = ts.as_pandas().reset_index()
        df.rename(columns={'datetime': 'date'}, inplace=True)
        
        # Aggregate the data into 2-minute intervals
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df_agg = df.resample('2T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        return df_agg
class Indicators:
    def __init__(self, df):
        self.df = df

    def calculate_sma(self, periods):
        for period in periods:
            self.df[f'SMA_{period}'] = self.df['close'].astype(float).rolling(window=period).mean()
            self.df[f'SMA_{period}'].fillna(method='bfill', inplace=True)

    def calculate_stochastic(self, periodK=12, smoothK=3, periodD=1, periodD1=4):
        # Calculate lowest low and highest high over periodK
        self.df['lowest_low'] = self.df['low'].rolling(window=periodK).min()
        self.df['highest_high'] = self.df['high'].rolling(window=periodK).max()
        
        # Calculate %K
        self.df['%K'] = 100 * (self.df['close'] - self.df['lowest_low']) / (self.df['highest_high'] - self.df['lowest_low'])
        
        # Smooth %K with an SMA over smoothK
        self.df['%K_smooth'] = self.df['%K'].rolling(window=smoothK).mean()
        
        # Calculate %D as the SMA of %K_smooth over periodD
        self.df['%D'] = self.df['%K_smooth'].rolling(window=periodD).mean()
        
        # Calculate %D1 as the SMA of %K_smooth over periodD1
        self.df['%D1'] = self.df['%K_smooth'].rolling(window=periodD1).mean()
        
        # Calculate Double Slow K Denominator (highest - lowest over periodK)
        self.df['DoubleSlowKDen'] = self.df['%K_smooth'].rolling(window=periodK).apply(lambda x: x.max() - x.min(), raw=True)
        
        # Calculate Double Slow D Denominator (highest - lowest over periodK)
        self.df['DoubleSlowDDen'] = self.df['%D'].rolling(window=periodK).apply(lambda x: x.max() - x.min(), raw=True)
        
        # Calculate Double Slow K
        self.df['DoubleSlowK'] = 100 * (self.df['%K_smooth'] - self.df['%K_smooth'].rolling(window=periodK).min()) / self.df['DoubleSlowKDen']
        
        # Calculate Double Slow D
        self.df['%FastD'] = self.df['%D'] 
        
        self.df['DoubleSlowD'] = 100 * (self.df['%D'] - self.df['%D'].rolling(window=periodK).min()) / self.df['DoubleSlowDDen']
        self.df.fillna(method='bfill', inplace=True) 
       

class MABStrategy(Indicators):
    def __init__(self, df, df5):
        super().__init__(df)
        self.df5 = df5
    
    def mab_strategy(self, period=5):

  
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['MAB_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(period, len(self.df)):
            # Check if the price bar touches or moves within $0.10 of any one of the 21, 50, or 200 moving averages
            if self.check_price_bar_touches_moving_averages(i):
                # Check if the price bar moves in the opposite direction
                    # Check if the Fast D is in the >80 zone or coming from that zone and decreasing
                    if self.check_fast_d_conditionshort(i):
                        # Check for at least 1 additional strategy pattern found
                            # This is a placeholder for the additional pattern check
                            if self.additional_pattern_checkshort(i):
                                strategy_signals['MAB_Signal'][i] = -1  # short signal
                    elif  self.check_fast_d_conditionlong(i):
                        # Check for at least 1 additional strategy pattern found
                        # This is a placeholder for the additional pattern check
                        if self.additional_pattern_checklong(i):
                            strategy_signals['MAB_Signal'][i] = 1  # long signal
    
        return strategy_signals
        
    
    def check_price_bar_touches_moving_averages(self, i):
    # Check if the price bar touches or moves within $0.10 of any one of the 21, 50, or 200 moving averages
        if (abs(self.df5['close'][i] - self.df['SMA_21'][i]) <= 0.10 or
            abs(self.df5['close'][i] - self.df['SMA_50'][i]) <= 0.10 or
            abs(self.df5['close'][i] - self.df['SMA_200'][i]) <= 0.10):
        
        # Check if the price bar then moves in the opposite direction
            if (i + 1 < len(self.df) and  # Ensure that i+1 is within the bounds of the DataFrame
                ((self.df['close'][i] > self.df['SMA_21'][i] and self.df['close'][i+1] < self.df['SMA_21'][i+1]) or 
                 (self.df['close'][i] < self.df['SMA_21'][i] and self.df['close'][i+1] > self.df['SMA_21'][i+1]) or 
                 (self.df['close'][i] > self.df['SMA_50'][i] and self.df['close'][i+1] < self.df['SMA_50'][i+1]) or 
                 (self.df['close'][i] < self.df['SMA_50'][i] and self.df['close'][i+1] > self.df['SMA_50'][i+1]) or 
                 (self.df['close'][i] > self.df['SMA_200'][i] and self.df['close'][i+1] < self.df['SMA_200'][i+1]) or 
                 (self.df['close'][i] < self.df['SMA_200'][i] and self.df['close'][i+1] > self.df['SMA_200'][i+1]))):
                    return True
            return False

    
    def check_fast_d_conditionshort(self, i):
        """
        Checks if the Fast D is in the >80 zone or coming from that zone and decreasing.
        """
        return ( self.df['close'][i] < self.df['close'][i-1] and self.df['close'][i-2] > self.df['close'][i-3] # price up then  bounce then decrease and fast d less than 20
                and self.df['close'][i-1] < self.df['SMA_21'][i] or self.df['close'][i-1] < self.df['SMA_50'][i] or self.df['close'][i-1] < self.df['SMA_200'][i]
    and  ( self.df['%D'][i] > 80  or self.df['DoubleSlowK'][i] > 80) )
    def check_fast_d_conditionlong(self, i):
        
        return ( self.df['close'][i] > self.df['close'][i-1] and self.df['close'][i-2] < self.df['close'][i-3] # reverse of price up then  bounce then decrease and fast d less than 20
                and self.df['close'][i-1] > self.df['SMA_21'][i] or self.df['close'][i-1] > self.df['SMA_50'][i] or self.df['close'][i-1] > self.df['SMA_200'][i]
    and ( self.df['%D'][i] < 20  or self.df['DoubleSlowK'][i] < 20))
    
    def additional_pattern_checkshort(self, i):
        if  ( self.df['SMA_21'][i] < self.df['SMA_200'][i] ):
            return True
        else:
            return self.df['SMA_21'][i] < self.df['SMA_50'][i]
    def additional_pattern_checklong(self, i):
        if  ( self.df['SMA_21'][i] > self.df['SMA_200'][i] ):
            return True
        else:
            return self.df['SMA_21'][i] > self.df['SMA_50'][i]
   
   
    def run_strategy(self):
        # Calculate the indicators
        self.calculate_sma([21, 50, 200])
        self.calculate_stochastic()
    
        # Apply the MAB strategy
        strategy_signals = self.mab_strategy()
    
        return strategy_signals


class SSGStrategy(Indicators):
    def __init__(self, df):
        super().__init__(df)
    
    def ssg_strategy(self):
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['SSG_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(2, len(self.df)):
            # Check if the Fast D is in the >80 zone or coming from that zone and decreasing
            if self.df['%FastD'][i] < 80 or (self.df['%FastD'][i] < 80 and self.df['%FastD'][i-2] > self.df['%FastD'][i-3] and self.df['%FastD'][i] < self.df['%FastD'][i-1]):
                # Check if the Double Slow K is at 100 and stays there while Fast D is < 80
                if self.df['DoubleSlowK'][i] > 80 or self.df['DoubleSlowK'][i-1] > 80 or self.df['DoubleSlowK'][i-2] > 80 : #THIS CAUSES EMPTY SSG SIGNALS
                    strategy_signals['SSG_Signal'][i] = -1  # Short signal
            # Check if the Fast D is in the <20 zone or coming from that zone and increasing
            elif self.df['%FastD'][i] > 20 or self.df['%FastD'][i] < 20   and self.df['%FastD'][i-2] < self.df['%FastD'][i-3] and self.df['%FastD'][i] > self.df['%FastD'][i-1]:
                # Check if the Double Slow K is at 0 and stays there while Fast D is > 20
                if self.df['DoubleSlowK'][i] < 20 or self.df['DoubleSlowK'][i-1] < 20 or self.df['DoubleSlowK'][i-2] < 20 : #THIS CAUSES EMPTY SSG SIGNALS
                    strategy_signals['SSG_Signal'][i] = 1  # Long signal

        return strategy_signals
    
    def run_strategy(self):
        # Calculate the indicators
      
    
        # Apply the SSG strategy
        strategy_signals = self.ssg_strategy()
    
        return strategy_signals
    
    
class HPatternStrategy(Indicators):
    def __init__(self, df):
        super().__init__(df)
    def check_fast_d_conditionshort(self, i):
        """
        Checks if the Fast D is in the >80 zone or coming from that zone and decreasing.
        """
        return (self.df['%FastD'][i] > 80 and
                self.df['%FastD'][i-1] > 80  and self.df['%FastD'][i] < self.df['%FastD'][i-1]) or (self.df['DoubleSlowK'][i] > 80 and
                        self.df['DoubleSlowK'][i-1] > 80  and self.df['DoubleSlowK'][i] < self.df['DoubleSlowK'][i-1])
    def check_fast_d_conditionlong(self, i):
        """
        Checks if the Fast D is in the >80 zone or coming from that zone and decreasing.
        """
        return (self.df['%FastD'][i] <20 and
                self.df['%FastD'][i-1] <20 and self.df['%FastD'][i-2] < 20 and self.df['%FastD'][i] > self.df['%FastD'][i-1]) or (self.df['DoubleSlowK'][i] <20 and
                        self.df['DoubleSlowK'][i-1] <20 and self.df['DoubleSlowK'][i-2] < 20 and self.df['DoubleSlowK'][i] > self.df['DoubleSlowK'][i-1])
   
    def h_pattern(self):
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['HPattern_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(2, len(self.df)):
            # Check if the price hits a high point (back leg)
            if self.df['high'][i] > self.df['high'][i-1]:
                # Check if the price drops for a while then goes back up close to the same level (front leg)
                if self.df['low'][i] < self.df['low'][i-1]  and (self.df['close'][i] >= self.df['close'][i-1] or self.df['close'][i] >= self.df['close'][i-2]):
                        # Check if the Fast D is in the >80 zone or coming from that zone and decreasing
                    if self.check_fast_d_conditionshort(i):
                        strategy_signals['HPattern_Signal'][i] = -1  # Short signal
                # Check if the price hits a low point (back leg)
                elif self.df['low'][i] < self.df['low'][i-1] :
                    # Check if the price goes up for a while then goes back down close to the same level (front leg)
                    if self.df['high'][i] > self.df['high'][i-1]  and (self.df['close'][i] <= self.df['close'][i-1] or self.df['close'][i] >= self.df['close'][i-2] ):
                            # Check if the Fast D is in the <20 zone or coming from that zone and increasing
                        if self.check_fast_d_conditionlong(i):
                            strategy_signals['HPattern_Signal'][i] = 1  # Long signal

        return strategy_signals
    
    def run_strategy(self):
        # Calculate the indicators
      
    
        # Apply the H Pattern strategy
        strategy_signals = self.h_pattern()
    
        return strategy_signals
  
class NewStrategy(Indicators):
    def __init__(self, df):
        super().__init__(df)
        
    def new_strategy(self):
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['New_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(3, len(self.df)):
            # Factor 3: Highs and Lows Comparison
            if self.check_highs_before_pattern(i):
                if self.check_horizontal_highs(i):
                    if self.check_three_bars_highs(i):
                        if self.oscillator_condition(i, direction='short'):
                            strategy_signals['New_Signal'][i] = -1  # Short signal

            elif self.check_lows_before_pattern(i):
                if self.check_horizontal_lows(i):
                    if self.check_three_bars_lows(i):
                        if self.oscillator_condition(i, direction='long'):
                            strategy_signals['New_Signal'][i] = 1  # Long signal

        return strategy_signals

    def check_highs_before_pattern(self, i):
        """
        Checks if the highs of each price bar before the pattern are higher than the previous one.
        """
        return (self.df['high'][i] > self.df['high'][i-1] or self.df['high'][i] > self.df['high'][i-2])

    def check_lows_before_pattern(self, i):
        """
        Checks if the lows of each price bar before the pattern are lower than the previous one.
        """
        return (self.df['low'][i] < self.df['low'][i-1] or self.df['low'][i] < self.df['low'][i-2])

    def check_horizontal_highs(self, i):
        """
        Checks if at least 3 price bars on the 5-minute chart have highs within $0.10 of each other.
        """
        return (abs(self.df['high'][i] - self.df['high'][i-1]) <= 0.10 and
                abs(self.df['high'][i-1] - self.df['high'][i-2]) <= 0.10 and
                abs(self.df['high'][i-2] - self.df['high'][i-3]) <= 0.10)

    def check_horizontal_lows(self, i):
        """
        Checks if at least 3 price bars on the 5-minute chart have lows within $0.10 of each other.
        """
        return (abs(self.df['low'][i] - self.df['low'][i-1]) <= 0.10 and
                abs(self.df['low'][i-1] - self.df['low'][i-2]) <= 0.10 and
                abs(self.df['low'][i-2] - self.df['low'][i-3]) <= 0.10)

    def check_three_bars_highs(self, i):
        """
        Checks if the first bar where the high price is not higher than the previous bar is the first of the 3 required bars.
        """
        return (self.df['high'][i-2] > self.df['high'][i-3] and
                self.df['high'][i-1] > self.df['high'][i-2] and
                self.df['high'][i] < self.df['high'][i-1])

    def check_three_bars_lows(self, i):
        """
        Checks if the first bar where the low price is not lower than the previous bar is the first of the 3 required bars.
        """
        return (self.df['low'][i-2] < self.df['low'][i-3] and
                self.df['low'][i-1] < self.df['low'][i-2] and
                self.df['low'][i] > self.df['low'][i-1])

    def oscillator_condition(self, i, direction):
        """
        Checks if the oscillator is >80 (short) or <20 (long)
        """
        if direction == 'short':
            return ((self.df['%FastD'][i] > 80 or self.df['%FastD'][i-1] > 80 or self.df['%FastD'][i-2] > 80)
        or (self.df['DoubleSlowK'][i] > 80 or self.df['DoubleSlowK'][i-1] > 80 or self.df['DoubleSlowK'][i-2] > 80))
        elif direction == 'long':
            return ( (self.df['%FastD'][i] < 20 or self.df['%FastD'][i-1] < 20 or self.df['%FastD'][i-2] < 20)
or (self.df['DoubleSlowK'][i] < 20 or self.df['DoubleSlowK'][i-1] < 20 or self.df['DoubleSlowK'][i-2] < 20) )
    def run_strategy(self):
        # Calculate the indicators
        self.calculate_sma([21, 50, 200])
        self.calculate_stochastic()

        # Apply the new strategy
        strategy_signals = self.new_strategy()

        return strategy_signals



def combine_signals(mab_signal, ssg_signal, h_pattern_signal, trip5_signal):
    # Ensure inputs are pandas Series or DataFrames with compatible indices
    if isinstance(mab_signal, pd.DataFrame):
        mab_signal = mab_signal.iloc[:, 0]  # Convert to Series if it's a single-column DataFrame
    if isinstance(ssg_signal, pd.DataFrame):
        ssg_signal = ssg_signal.iloc[:, 0]  # Convert to Series if it's a single-column DataFrame
    if isinstance(h_pattern_signal, pd.DataFrame):
        h_pattern_signal = h_pattern_signal.iloc[:, 0]  # Convert to Series if it's a single-column DataFrame
    if isinstance(trip5_signal, pd.DataFrame):
        trip5_signal = trip5_signal.iloc[:, 0]  # Convert to Series if it's a single-column DataFrame

    # Combine the signals into a DataFrame
    combined_signal = pd.DataFrame({
        'MAB_Signal': mab_signal,
        'SSG_Signal': ssg_signal,
        'HPattern_Signal': h_pattern_signal,
        'Trip5_Signal': trip5_signal
    })

     


    return combined_signal





def calculate_signals1(signals_df, threshold):
    buy_signals = (signals_df == 1).sum(axis=1) >= threshold
    sell_signals = (signals_df == -1).sum(axis=1) >= threshold
    return buy_signals, sell_signals

class MyStrategy1(Strategy):
    # Define threshold as a class variable for optimization
    threshold = thresholdval  # Default value, will be overridden during optimization

    def init(self):
        # Calculate buy and sell signals with the current threshold
        buy_signals, sell_signals = calculate_signals1(combined_signals_1min, self.threshold)

        # Convert signals into a format usable by the backtesting library
        self.signals_buy = self.I(lambda: buy_signals)
        self.signals_sell = self.I(lambda: sell_signals)

    def next(self):
        # Execute trades based on the buy and sell signals
        if self.signals_buy[-1]:
            self.position.close()
            self.buy( sl = self.data.Close[-1]- slval , tp = self.data.Close[-1]+ tpval)
        elif self.signals_sell[-1]:
            self.position.close()
            self.sell(sl = self.data.Close[-1]+ slval , tp = self.data.Close[-1]- tpval)





def calculate_signals2(signals_df, threshold):
    buy_signals = (signals_df == 1).sum(axis=1) >= threshold
    sell_signals = (signals_df == -1).sum(axis=1) >= threshold
    return buy_signals, sell_signals

class MyStrategy2(Strategy):
    # Define threshold as a class variable for optimization
    threshold = thresholdval  # Default value, will be overridden during optimization

    def init(self):
        # Calculate buy and sell signals with the current threshold
        buy_signals, sell_signals = calculate_signals2(combined_signals_2min, self.threshold)

        # Convert signals into a format usable by the backtesting library
        self.signals_buy = self.I(lambda: buy_signals)
        self.signals_sell = self.I(lambda: sell_signals)

    def next(self):
        # Execute trades based on the buy and sell signals
        if self.signals_buy[-1]:
            self.position.close()
            self.buy(sl = self.data.Close[-1]- slval , tp = self.data.Close[-1]+ tpval)
        elif self.signals_sell[-1]:
            self.position.close()
            self.sell(sl = self.data.Close[-1]+ slval , tp = self.data.Close[-1]- tpval)

# Run the backtest with optimization





def calculate_signals5(signals_df, threshold):
    buy_signals = (signals_df == 1).sum(axis=1) >= threshold
    sell_signals = (signals_df == -1).sum(axis=1) >= threshold
    return buy_signals, sell_signals

class MyStrategy5(Strategy):
    # Define threshold as a class variable for optimization
    threshold = thresholdval # Default value, will be overridden during optimization

    def init(self):
        # Calculate buy and sell signals with the current threshold
        buy_signals, sell_signals = calculate_signals5(combined_signals_5min, self.threshold)

        # Convert signals into a format usable by the backtesting library
        self.signals_buy = self.I(lambda: buy_signals)
        self.signals_sell = self.I(lambda: sell_signals)

    def next(self):
        # Execute trades based on the buy and sell signals
        if self.signals_buy[-1]:
            self.position.close()
            self.buy(sl = self.data.Close[-1]- slval , tp = self.data.Close[-1]+ tpval)
        elif self.signals_sell[-1]:
            self.position.close()
            self.sell(sl = self.data.Close[-1]+ slval , tp = self.data.Close[-1]- tpval)
            
            

       
if st.button("Run Backtest"):
    with st.spinner('Running backtest across timeframes...'):

        
        
            # Initialize data feeders for different timeframes
            data_feeder_1min = DataFeeder('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '1min')
            data_feeder_2min = DataFeeder2('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '2min')
            data_feeder_5min = DataFeeder('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '5min')
            
            # Run each strategy on the 1-minute data
            mab_signals_1min = MABStrategy(data_feeder_1min.df, data_feeder_5min.df).run_strategy()
            ssg_signals_1min = SSGStrategy(data_feeder_1min.df).run_strategy()
            h_pattern_signals_1min = HPatternStrategy(data_feeder_1min.df).run_strategy()
            trip5_signals_1min = NewStrategy(data_feeder_1min.df).run_strategy()
            
            # Run each strategy on the 2-minute data
            mab_signals_2min = MABStrategy(data_feeder_2min.df, data_feeder_5min.df).run_strategy()
            ssg_signals_2min = SSGStrategy(data_feeder_2min.df).run_strategy()
            h_pattern_signals_2min = HPatternStrategy(data_feeder_2min.df).run_strategy()
            trip5_signals_2min = NewStrategy(data_feeder_2min.df).run_strategy()
            
            # Run each strategy on the 5-minute data
            mab_signals_5min = MABStrategy(data_feeder_5min.df , data_feeder_5min.df).run_strategy()
            ssg_signals_5min = SSGStrategy(data_feeder_5min.df).run_strategy()
            h_pattern_signals_5min = HPatternStrategy(data_feeder_5min.df).run_strategy()
            trip5_signals_5min = NewStrategy(data_feeder_5min.df).run_strategy()
            # Combine signals for each timeframe
            date_1min = data_feeder_1min.df['date']
            date_2min = data_feeder_2min.df['date']
            date_5min = data_feeder_5min.df['date']
            combined_signals_1min = combine_signals(mab_signals_1min, ssg_signals_1min, h_pattern_signals_1min, trip5_signals_1min)
            combined_signals_2min = combine_signals(mab_signals_2min, ssg_signals_2min, h_pattern_signals_2min, trip5_signals_2min)
            combined_signals_5min = combine_signals(mab_signals_5min, ssg_signals_5min, h_pattern_signals_5min, trip5_signals_5min)
            combined_signals_1min = pd.concat([date_1min, combined_signals_1min], axis=1)
            combined_signals_2min = pd.concat([date_2min, combined_signals_2min], axis=1)
            combined_signals_5min = pd.concat([date_5min, combined_signals_5min], axis=1)
            # Combine all timeframes into one DataFrame
            
            
            
            # Extract the data for each timeframe
            input_data_1min = data_feeder_1min.df[['open', 'high', 'low', 'close', 'volume']].rename(columns={
                'open': 'Open',
                'low': 'Low',
                'high': 'High',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            input_data_2min = data_feeder_2min.df[['open', 'high', 'low', 'close', 'volume']].rename(columns={
                'open': 'Open',
                'low': 'Low',
                'high': 'High',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            input_data_5min = data_feeder_5min.df[['open', 'high', 'low', 'close', 'volume']].rename(columns={
                'open': 'Open',
                'low': 'Low',
                'high': 'High',
                'close': 'Close',
                'volume': 'Volume'
            })
            print(type(mab_signals_1min))
            print(mab_signals_1min.head())
                    # Run the backtest with optimization
            bt = Backtest(input_data_1min, MyStrategy1, cash=cashval)
        
            # Optimize the threshold parameter
            stats = bt.run()
            st.title("1minute Backtest")
            trades = (stats['_trades'])
            
            #Dates
            entry_indices = trades['EntryBar'].values
            entry_dates = data_feeder_1min.df.iloc[entry_indices]['date']
            trades['EntryDate'] = entry_dates.values
            
            exit_bars = trades['ExitBar'].values
            exit_dates = data_feeder_1min.df.iloc[exit_bars]['date']
            trades['ExitDate'] = exit_dates.values
           
            
            
            
            
            order = [ 'Size', 'EntryDate' , 'ExitDate' , 'EntryPrice',
                    'ExitPrice' ,  'PnL', 'ReturnPct' , 'Duration',
                    'EntryBar', 'ExitBar' ,  'EntryTime' ,'ExitTime']
            
            
            trades = trades[order]
            st.write(trades)
            st.title("1minute Signals")
            st.write(combined_signals_1min)
            html_filename = 'backtest_plot.html'
            bt.plot(plot_volume=True, filename=html_filename,plot_pl=False, open_browser=False)

# Read the HTML file content
            with open(html_filename, 'r') as file:
                html_content = file.read()

# Display the HTML plot in Streamlit
            st.components.v1.html(html_content, height=600)
            
        
            bt2 = Backtest(input_data_2min, MyStrategy2, cash=cashval)
        
            # Optimize the threshold parameter
            stats2 = bt2.run()
            st.title("2minute Backtest")
            trades2 = (stats2['_trades'])
            
            #Dates
            entry_indices2 = trades2['EntryBar'].values
            entry_dates2 = data_feeder_2min.df.iloc[entry_indices2]['date']
            trades2['EntryDate'] = entry_dates2.values
            
            exit_bars2 = trades2['ExitBar'].values
            exit_dates2 = data_feeder_2min.df.iloc[exit_bars2]['date']
            trades2['ExitDate'] = exit_dates2.values
            
            
            
            order = [ 'Size', 'EntryDate' , 'ExitDate' , 'EntryPrice',
                     'ExitPrice' ,  'PnL', 'ReturnPct' , 'Duration',
                     'EntryBar', 'ExitBar' ,  'EntryTime' ,'ExitTime']
            
            
            trades2 = trades2[order]
            st.write(trades2)
            st.title("2minute Signals")
            st.write(combined_signals_2min)
            html_filename2 = 'backtest_plot2.html'
            bt2.plot(plot_volume=True, plot_pl=False, filename=html_filename2, open_browser=False)

# Read the HTML file content
            with open(html_filename, 'r') as file:
                html_content2 = file.read()

# Display the HTML plot in Streamlit
            st.components.v1.html(html_content2, height=600)
            
        # Run the backtest with optimization
            bt5 = Backtest(input_data_5min, MyStrategy5, cash=cashval)
            
            # Optimize the threshold parameter
            stats5 = bt5.run()
            st.title("5minute Backtest")
            trades5 = (stats5['_trades'])
            
            #Dates
            entry_indices5 = trades5['EntryBar'].values
            entry_dates5 = data_feeder_5min.df.iloc[entry_indices5]['date']
            trades5['EntryDate'] = entry_dates5.values
            
            exit_bars5 = trades5['ExitBar'].values
            exit_dates5 = data_feeder_5min.df.iloc[exit_bars5]['date']
            trades5['ExitDate'] = exit_dates5.values
            
            order = [ 'Size', 'EntryDate' , 'ExitDate' , 'EntryPrice',
                     'ExitPrice' ,  'PnL', 'ReturnPct' , 'Duration',
                     'EntryBar', 'ExitBar' ,  'EntryTime' ,'ExitTime']
            
            
            trades5 = trades5[order]
            st.write(trades5)
            st.title("5minute Signals")
            st.write(combined_signals_5min)
            html_filename5 = 'backtest_plot5.html'
            bt5.plot(plot_volume=True, filename=html_filename5,plot_pl=False, open_browser=False)

# Read the HTML file content
            with open(html_filename, 'r') as file:
                html_content5 = file.read()

# Display the HTML plot in Streamlit
            st.components.v1.html(html_content5, height=600)
        




###
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title(" Prediction ML " )


if st.button("Run Model"):
    with st.spinner('Running ML model for each timeframe...'):
        data_feeder_1min_new = DataFeeder('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '1min')
        data_feeder_2min_new = DataFeeder2('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '2min')
        data_feeder_5min_new = DataFeeder('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '5min')
    
        # Run strategies for each timeframe
        mab_strategy_1min = MABStrategy(data_feeder_1min_new.df, data_feeder_5min_new.df)
        mab_signals_1min = mab_strategy_1min.run_strategy()
    
        ssg_strategy_1min = SSGStrategy(data_feeder_1min_new.df)
        ssg_signals_1min = ssg_strategy_1min.run_strategy()
    
        h_pattern_strategy_1min = HPatternStrategy(data_feeder_1min_new.df)
        h_pattern_signals_1min = h_pattern_strategy_1min.run_strategy()
    
        trip5_strategy_1min = NewStrategy(data_feeder_1min_new.df)
        trip5_signals_1min = trip5_strategy_1min.run_strategy()
    
        mab_strategy_2min = MABStrategy(data_feeder_2min_new.df, data_feeder_5min_new.df)
        mab_signals_2min = mab_strategy_2min.run_strategy()
    
        ssg_strategy_2min = SSGStrategy(data_feeder_2min_new.df)
        ssg_signals_2min = ssg_strategy_2min.run_strategy()
    
        h_pattern_strategy_2min = HPatternStrategy(data_feeder_2min_new.df)
        h_pattern_signals_2min = h_pattern_strategy_2min.run_strategy()
    
        trip5_strategy_2min = NewStrategy(data_feeder_2min_new.df)
        trip5_signals_2min = trip5_strategy_2min.run_strategy()
    
        mab_strategy_5min = MABStrategy(data_feeder_5min_new.df, data_feeder_5min_new.df)
        mab_signals_5min = mab_strategy_5min.run_strategy()
    
        ssg_strategy_5min = SSGStrategy(data_feeder_5min_new.df)
        ssg_signals_5min = ssg_strategy_5min.run_strategy()
    
        h_pattern_strategy_5min = HPatternStrategy(data_feeder_5min_new.df)
        h_pattern_signals_5min = h_pattern_strategy_5min.run_strategy()
    
        trip5_strategy_5min = NewStrategy(data_feeder_5min_new.df)
        trip5_signals_5min = trip5_strategy_5min.run_strategy()
    
        # Define the features you want to use
        features = ['open', 'close', 'low', 'high', 'SMA_21', 'SMA_50', 'SMA_200', '%FastD', 'DoubleSlowK']
    
        # Import the 12 models
        model_1min_mab = joblib.load('1min_MAB_Signal_model.pkl')
        model_1min_ssg = joblib.load('1min_SSG_Signal_model.pkl')
        model_1min_h = joblib.load('1min_HPattern_Signal_model.pkl')
        model_1min_trip5 = joblib.load('1min_Trip5_Signal_model.pkl')
    
        model_2min_mab = joblib.load('2min_MAB_Signal_model.pkl')
        model_2min_ssg = joblib.load('2min_SSG_Signal_model.pkl')
        model_2min_h = joblib.load('2min_HPattern_Signal_model.pkl')
        model_2min_trip5 = joblib.load('2min_Trip5_Signal_model.pkl')
    
        model_5min_mab = joblib.load('5min_MAB_Signal_model.pkl')
        model_5min_ssg = joblib.load('5min_SSG_Signal_model.pkl')
        model_5min_h = joblib.load('5min_HPattern_Signal_model.pkl')
        model_5min_trip5 = joblib.load('5min_Trip5_Signal_model.pkl')
    
        # Make predictions for each model
        X_1min = data_feeder_1min_new.df[features]
        X_2min = data_feeder_2min_new.df[features]
        X_5min = data_feeder_5min_new.df[features]
    
        X_1min = StandardScaler().fit_transform(X_1min)
        X_2min = StandardScaler().fit_transform(X_2min)
        X_5min = StandardScaler().fit_transform(X_5min)
        
     
        
        
        mab_pred_1min = model_1min_mab.predict(X_1min)
        ssg_pred_1min = model_1min_ssg.predict(X_1min)
        h_pred_1min = model_1min_h.predict(X_1min)
        trip5_pred_1min = model_1min_trip5.predict(X_1min)
    
        mab_pred_2min = model_2min_mab.predict(X_2min)
        ssg_pred_2min = model_2min_ssg.predict(X_2min)
        h_pred_2min = model_2min_h.predict(X_2min)
        trip5_pred_2min = model_2min_trip5.predict(X_2min)
    
        mab_pred_5min = model_5min_mab.predict(X_5min)
        ssg_pred_5min = model_5min_ssg.predict(X_5min)
        h_pred_5min = model_5min_h.predict(X_5min)
        trip5_pred_5min = model_5min_trip5.predict(X_5min)
        
               # Convert numpy signal arrays to pandas DataFrames
        mab_pred_1min = pd.DataFrame(mab_pred_1min, columns=['MAB'])
        ssg_pred_1min = pd.DataFrame(ssg_pred_1min, columns=['SSG'])
        h_pred_1min = pd.DataFrame(h_pred_1min, columns=['HPattern'])
        trip5_pred_1min = pd.DataFrame(trip5_pred_1min, columns=['Trip5'])
        
        mab_pred_2min = pd.DataFrame(mab_pred_2min, columns=['MAB'])
        ssg_pred_2min = pd.DataFrame(ssg_pred_2min, columns=['SSG'])
        h_pred_2min = pd.DataFrame(h_pred_2min, columns=['HPattern'])
        trip5_pred_2min = pd.DataFrame(trip5_pred_2min, columns=['Trip5'])
        
        mab_pred_5min = pd.DataFrame(mab_pred_5min, columns=['MAB'])
        ssg_pred_5min = pd.DataFrame(ssg_pred_5min, columns=['SSG'])
        h_pred_5min = pd.DataFrame(h_pred_5min, columns=['HPattern'])
        trip5_pred_5min = pd.DataFrame(trip5_pred_5min, columns=['Trip5'])
        
        # Concatenate the DataFrames
        min1signals = pd.concat([mab_pred_1min, ssg_pred_1min, h_pred_1min, trip5_pred_1min], axis=1)
        min2signals = pd.concat([mab_pred_2min, ssg_pred_2min, h_pred_2min, trip5_pred_2min], axis=1)
        min5signals = pd.concat([mab_pred_5min, ssg_pred_5min, h_pred_5min, trip5_pred_5min], axis=1)
        
        # Display the predictions
        st.write("Predictions:")
        st.write("1min timeframe:")
        st.write(min1signals.tail())
        
        st.write("MAB signal:", mab_pred_1min.iloc[-1]['MAB'])
        st.write("SSG signal:", ssg_pred_1min.iloc[-1]['SSG'])
        st.write("HPattern signal:", h_pred_1min.iloc[-1]['HPattern'])
        st.write("Trip5 signal:", trip5_pred_1min.iloc[-1]['Trip5'])
        
        st.write("2min timeframe:")
        st.write(min2signals.tail())
        
        st.write("MAB signal:", mab_pred_2min.iloc[-1]['MAB'])
        st.write("SSG signal:", ssg_pred_2min.iloc[-1]['SSG'])
        st.write("HPattern signal:", h_pred_2min.iloc[-1]['HPattern'])
        st.write("Trip5 signal:", trip5_pred_2min.iloc[-1]['Trip5'])
        
        st.write("5min timeframe:")
        st.write(min5signals.tail())
        
        st.write("MAB signal:", mab_pred_5min.iloc[-1]['MAB'])
        st.write("SSG signal:", ssg_pred_5min.iloc[-1]['SSG'])
        st.write("HPattern signal:", h_pred_5min.iloc[-1]['HPattern'])
        st.write("Trip5 signal:", trip5_pred_5min.iloc[-1]['Trip5'])
      
