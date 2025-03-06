import ipywidgets as widgets
from IPython.display import display
from datetime import date, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'notebook'

import requests



def loadNupdate(tickers, from_date, today, freq, file):
    try:
        data = pd.read_csv('Tickers/'+file+'.csv', index_col=0, parse_dates=True, dayfirst=True)
        data = data.reindex(columns=tickers)
    except:
        data = load_data(tickers, from_date, today, freq)
        data.to_csv('Tickers/'+file+'.csv')
    freq = freq[0]
    delta = {'d':1,'w':5,'m':30,'q':90,'a':260}
    next_timestamp = data.index[-1]+timedelta(days=delta[freq])
    print('Next Timestamp:  '+str(next_timestamp))
    print('Today Timestamp: '+str(pd.Timestamp(today)))
    if (next_timestamp < pd.Timestamp(today)):
        df = load_data(tickers, next_timestamp.date(), today, freq)
        if df.empty:
            print('No new data')
        else:
            df.index = pd.to_datetime(df.index)
            data.index = pd.to_datetime(data.index)            
            data = data.combine_first(df)
            data.to_csv('Tickers/'+file+'.csv')
            print('\n Dataset updated and saved \n')
    return data

def load_data(tickers, from_date, today, freq):
    try:
        df = ticker_download(tickers, from_date, freq)
    except:
        df = yf.download(tickers, from_date, today, interval="1"+freq, auto_adjust=False)
        df = df[['Adj Close']].copy()
        for col in df.columns:        
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype(float)
#     df = df.reindex(columns=tickers)
    df.index = pd.to_datetime(df.index)
    return df


def ticker_download(tickers, from_date, freq):
    FRED_URL = 'https://api.stlouisfed.org/fred'
    API_KEY = '4c65ec00c7f4c4f6c8c857294433952b'
    # df = pdr.DataReader(tickers, 'fred', start=from_date, end=to_date) # Download data using DataReader
    df = pd.DataFrame()
    for ser in tickers:
        tmp_json = requests.get(f'{FRED_URL}/series/observations?series_id={ser}&api_key={API_KEY}&file_type=json&observation_start={from_date}&frequency={freq}').json()
        new_set = pd.DataFrame(tmp_json['observations'], columns=['date', 'value']).rename(columns={'value': ser}).set_index('date')
        for col in new_set.select_dtypes(exclude=['float64']).columns:        
            new_set[col] = pd.to_numeric(new_set[col], errors='coerce')
            new_set[col] = new_set[col].astype(float)
        new_set.dropna(inplace=True)
        df = pd.merge(df, new_set, how='outer', left_index=True, right_index=True)
    return df




def spread_analysis(data, tickers):
    def spread_analysis_helper(value1, value2):
        x1 = data[value1]
        x2 = data[value2]

        df = pd.DataFrame(x1-x2,columns=['Spread']).dropna()
        df['pos'] = np.maximum(df['Spread'], 0)
        df['neg'] = np.minimum(df['Spread'], 0)

        # Define the range where y > 0 and y < 0
        y_above_zero = np.maximum(df['Spread'], 0)
        y_below_zero = np.minimum(df['Spread'], 0)

        fig = go.Figure()    
        fig.add_trace(go.Scatter(x=df.index, y=df['pos'], name='positive', fill='tozeroy')) # fill down to xaxis
        fig.add_trace(go.Scatter(x=df.index, y=df['neg'], name='negative', fill='tozeroy')) # fill to trace0 y

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig.update_layout(title_text='<b>'+value1+'-'+value2+' Spread<b>', title_x=0.5)
        fig.update_layout(margin=dict(l=0, r=0, t=25, b=0))
        fig.show()

    controls = widgets.interactive(spread_analysis_helper, value1=tickers, value2=tickers)
    display(controls)

    
    
    
def returns(df, col, shift_periods):
    """ Function that takes as input a DataFrame (df), and calculates the returns of column (col) between n shifted_periods.                           
    """
    new_col = (df[col]/df[col].shift(shift_periods))-1 
    return new_col
    

    
    
def plot_select(data, tickers):
    plt_ticker_controls = widgets.interactive(lambda tic:plot_ticker(pd.DataFrame(data[tic])), tic=tickers)
    display(plt_ticker_controls)
    
def plot_ticker(df,col=0):
    subset = df.copy()
    subset.dropna(inplace=True)
    
    fig = go.Figure()    
    fig.add_trace(go.Scatter(x=subset.index, y=subset[subset.columns[col]]))
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(title_text='<b>'+df.columns[col]+'<b>', title_x=0.5)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.show()