import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import coint
import datetime

def get_pct_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with a date column and numerical columns, and returns a new DataFrame
    with the percentage weights of each numerical column per row, rounded to two decimal places.
    Args:
        df: A DataFrame containing a date column and numerical columns.
    Returns:
        A new DataFrame with the percentage weights of each numerical column per row.
    Example:
        data = {'date_time': ['2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04', '2023-04-05'],
                'external': [22, 4, 6, 82, 10],
                'cdo': [32, 6, 29, 12, 15],
                'maws': [44, 82, 12, 162, 20]}
        df = pd.DataFrame(data)
        df2 = calculate_pct_weights(df)

        # df2:
        #             external_pct  cdo_pct  maws_pct
        # date_time                                  
        # 2023-04-01         22.45    32.65     44.90
        # 2023-04-02          4.35     6.52     89.13
        # 2023-04-03         12.77    61.70     25.53
        # 2023-04-04         32.03     4.69     63.28
        # 2023-04-05         22.22    33.33     44.44
    """
    df.set_index(df.columns[0], inplace=True)
    non_date_columns = df.columns    # Get a list of non-date column names
    df['row_sum'] = df[non_date_columns].sum(axis=1)    # Calculate the row sum
    for column in non_date_columns:    # Calculate the percentage weight of each non-date column
        df[f'{column}_pct'] = (df[column] / df['row_sum']) * 100
    df_pct = df[[f'{column}_pct' for column in non_date_columns]]  # Create a new DataFrame with percentage weights
    return df_pct.round(2)

def filter_only_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with a date column and numerical columns, and returns a new DataFrame
    with the last non-null value of each quarter for each numerical column.

    Args:
        df: A DataFrame containing a date column and numerical columns.

    Returns:
        A new DataFrame with the last non-null value of each quarter for each numerical column.

    Example:
        See the sample data and result provided in the code below.
        
    eg. 
    data = {'date_time': ['2023-12-02', '2023-02-15', '2023-03-18', '2023-04-01', '2023-06-15', '2023-07-01', '2023-09-30'],
        'internal': [1, 2, 33, 4, 5, 6, 7],
        'external': [22, 4, 26, 8, 10, 12, 14],
        'cdo': [43, 63, 9, 142, 15, 18, 21],
        'maws': [54, 82, 125, 16, 20, 24, 28]}
    sample_df = pd.DataFrame(data)
    result = filter_only_quarterly(sample_df)

    """
    date_column = df.columns[0]  # Get the first column name
    df[date_column] = pd.to_datetime(df[date_column])  # Convert date column to datetime type
    df.set_index(date_column, inplace=True)  # Set date column as the index
    def last_non_null(s):  # Custom function to get the last non-null value in a group
        return s.dropna().iloc[-1]   # Custom function to get the last non-null value in a group-1]
    quarterly_data = df.resample('Q').agg(last_non_null)    # Find the last non-null value of each quarter in the DataFrame
    quarterly_data.reset_index(inplace=True)    # Reset the index to make the date column a regular column
    quarterly_data['quarter'] = quarterly_data[date_column].dt.to_period("Q").astype(str)    # Add a column indicating the calendar quarter
    return quarterly_data 

def dummy_value(): #date, value
    dates = pd.date_range(start='2023-04-24', periods=100, freq='D')
    integers = np.random.randint(0, 100, 100)
    df = pd.DataFrame({'Date': dates, 'Value': integers})
    return df

def full_join_multi_tables(dfs): #result = full_join_fill_gaps([df1, df2, df3])
    on_col = dfs[0].columns[0]    # Get the first column name from the first DataFrame
    result = dfs[0]    # Initialize the result with the first DataFrame
    for df in dfs[1:]:    # Perform a full outer join on the remaining DataFrames
        result = result.merge(df, on=on_col, how='outer')
    result = result.sort_values(by=on_col)    # Sort the result based on the joining column
    result = result.ffill()    # Use forward-fill to fill the gaps
    return result

def covariance(df, col1, col2, window=None):
    if window is not None:
        return df[[col1, col2]].rolling(window=window).cov().iloc[::window].iloc[:, 1].values
    else:
        return df[[col1, col2]].cov().iloc[0, 1]

def correlation_coefficient(df, col1, col2, window=None):
    if window is not None:
        return df[[col1, col2]].rolling(window=window).corr().iloc[::window].iloc[:, 1].values
    else:
        return df[[col1, col2]].corr().iloc[0, 1]
    
def pvalue(df, col1, col2):
    S1 = df[col1]
    S2 = df[col2]
    score, pvalues, _ = coint(S1, S2)
    return pvalues

def coefficient_of_variation(df, col, window=None):
    if window is not None:
        mean = df[col].rolling(window=window).mean()
        std_dev = df[col].rolling(window=window).std()
        return (std_dev / mean).values
    else:
        mean = df[col].mean()
        std_dev = df[col].std()
        return std_dev / mean

def z_score(df, col, value, window=None):
    if window is not None:
        mean = df[col].rolling(window=window).mean()
        std_dev = df[col].rolling(window=window).std()
        return ((value - mean) / std_dev).values
    else:
        mean = df[col].mean()
        std_dev = df[col].std()
        return (value - mean) / std_dev
    
def plot_zscore(series):
    mean = series.mean()
    std = series.std()
    zscores = (series - mean) / std

    trace_zscores = go.Scatter(x=series.index, y=zscores.values, name='Rolling Ratio z-Score')
    trace_mean = go.Scatter(x=series.index, y=[mean]*len(series), name='Mean')
    trace_upper = go.Scatter(x=series.index, y=[1.0]*len(series), name='+1', line=dict(color='red', dash='dash'))
    trace_lower = go.Scatter(x=series.index, y=[-1.0]*len(series), name='-1', line=dict(color='green', dash='dash'))

    layout = go.Layout(title='Rolling Ratio z-Score Plot', xaxis=dict(title='Date'), yaxis=dict(title='z-Score'))
    fig = go.Figure(data=[trace_zscores, trace_mean, trace_upper, trace_lower], layout=layout)

    return fig.show()

def plot_ratio(df, col1, col2):
    ratios = df[col1] / df[col2]
    mean_ratio = ratios.mean()
    data = [go.Scatter(x=ratios.index, y=ratios, name='Ratio'),
            go.Scatter(x=ratios.index, y=[mean_ratio]*len(ratios.index), name='Mean Ratio')]
    layout = go.Layout(title='Ratio Plot', xaxis=dict(title='Date'), yaxis=dict(title='Ratio'))
    fig = go.Figure(data=data, layout=layout)
    return fig.show()

def plot_ratio_ma(df, col1,col2, ma_short_window, ma_long_window):
    df['ratios'] = df[col1] / df[col2]
    df['ma_short'] = df['ratios'].rolling(window=ma_short_window).mean()
    df['ma_long'] = df['ratios'].rolling(window=ma_long_window).mean()

    trace_ratio = go.Scatter(x=df.iloc[:,0], y=df.ratios, name='Ratio')
    trace_ma_short = go.Scatter(x=df.iloc[:,0], y=df.ma_short, name='{}d Ratio MA'.format(ma_short_window))
    trace_ma_long = go.Scatter(x=df.iloc[:,0], y=df.ma_long, name='{}d Ratio MA'.format(ma_long_window))
    
    layout = go.Layout(title='Ratio MA Plot', xaxis=dict(title='Date', type='date'), yaxis=dict(title='Ratio'))
    fig = go.Figure(data=[trace_ratio, trace_ma_short, trace_ma_long], layout=layout)
    
    return fig.show()

def plot_ratio_buy_sell(series, ratios, ma_window, zscore_long_short):
    # Create the buy and sell signals from z score
    buy = series.copy()
    sell = series.copy()
    buy[zscore_long_short > -1] = 0
    sell[zscore_long_short < 1] = 0
    
    # Create a Plotly figure with the ratios, buy, and sell signals
    fig = go.Figure()

    # Add the ratio trace
    fig.add_trace(go.Scatter(x=series.index, y=series[ma_window:],
                             mode='lines',
                             name='Ratio'))

    # Add the buy signal trace
    fig.add_trace(go.Scatter(x=buy.index, y=buy[ma_window:],
                             mode='markers',
                             marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='Buy Signal'))

    # Add the sell signal trace
    fig.add_trace(go.Scatter(x=sell.index, y=sell[ma_window:],
                             mode='markers',
                             marker=dict(symbol='triangle-up', size=10, color='red'),
                             name='Sell Signal'))

    # Set the figure layout and axis range
    fig.update_layout(title='Buy and Sell Signals based on Z Score',
                      xaxis_title='Date',
                      yaxis_title='Ratio',
                      yaxis=dict(range=[ratios.min(), ratios.max()]),
                      width=800, height=400)
    fig.show()
# plot_ratio_buy_sell(train,ratios, ma_long_window, zscore_long_short)
    
# TESTING SECTION Replace with your dataframe
# df = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=10),
#                    'col2': np.random.randn(10),
#                    'col3': np.random.randn(10)})

# window_size = 3

# cov = covariance(df, 'col2', 'col3', window=window_size)
# corr_coef = correlation_coefficient(df, 'col2', 'col3', window=window_size)
# coef_of_var = coefficient_of_variation(df, 'col2', window=window_size)
# z_value = z_score(df, 'col2', 1.5, window=window_size)  # Replace 1.5 with the value you want to calculate the z-score for

# print("Covariance with window:", cov)
# print("Correlation Coefficient with window:", corr_coef)
# print("Coefficient of Variation with window:", coef_of_var)
# print("Z-Score with window:", z_value)
    

def add_slope_column(df, window):
    df['date_ordinal'] = pd.to_datetime(df.iloc[:,0]).map(datetime.datetime.toordinal)
    df.reset_index(drop=True, inplace=True)
    # df = df.drop(df.columns[0], axis=1)  #drop the date column since we will use date_ordinal instead
    x_axis = 'date_ordinal'
    df.index = df.iloc[:,1] #df[y_axis] this should be the value column
    df['slope'] = df[x_axis].rolling(window=window).apply(lambda x_axis: linregress(x_axis, x_axis.index)[0])
    return df.reset_index(drop=True).dropna()

def get_slope(df):
    df['date_ordinal'] = pd.to_datetime(df.iloc[:,0]).map(datetime.datetime.toordinal)
    slope, intercept, _, _, _ = linregress(df['date_ordinal'], df.iloc[:,1])
    return slope, intercept

def plot_multi_ts_lines(dfs, names=None):
    fig = go.Figure()

    for i, df in enumerate(dfs):
        # Assign a unique name for each trace
        if names:
            trace_name = names[i]
        else:
            trace_name = f'Time Series {i+1}'

        # Use the first and second columns by indexing
        date_col = df.columns[0]
        value_col = df.columns[1]

        fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode='lines', name=trace_name))

    fig.update_layout(title='Multiple Time Series Line Chart', xaxis_title='Date', yaxis_title='Value')
    fig.show()
# plot_multi_ts_lines([df1, df2, df3], names=['Series 1', 'Series 2', 'Series 3'])

def plot_time_series_with_slope_plotly_single(df):
    slope, intercept = get_slope(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Time Series'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['date_ordinal'].map(lambda x: slope * x + intercept), mode='lines', name='Slope', line=dict(dash='dot', color='red')))

    fig.update_layout(title='Time Series with Slope Overlay', xaxis_title='Date', yaxis_title='Value')
    fig.show()

def perc_change(df,horizon,column_index=1,keep_only_perc_change=True):
    horizon = horizon-1
    df['perc_change'] = df.iloc[:,column_index].pct_change(periods=horizon,fill_method='ffill').round(decimals=4)
    if keep_only_perc_change ==True:
        return df.drop(df.columns[1],axis=1).dropna()
    else:
        return df.dropna()
    
def describe_df(df,column_index=1):
    return df.iloc[:,column_index].describe(percentiles = [.001,.01,.05,.1,.15,.25,.5,.75,.85,.90,.95,.99,.999]).round(2).reset_index()

def describe_df_forecast(df, input_value,column_index=1):
    describe = describe_df(df,column_index)
    describe['input_value'] = input_value
    describe['forecast_value'] = input_value * (1+describe.iloc[:,column_index]/100).round(3)
    return describe

def plot_describe_df(describe_df): #will return table
    fig = go.Figure(data=[go.Table(header=dict(values=describe_df.columns.tolist()),
                    cells=dict(values=[describe_df.iloc[:,0].tolist(), 
                                       describe_df.iloc[:,1].tolist(),
                                       ]))])
    fig.show()

def plot_ts_line_chart(df, y_values = None): #assume no index was set
    if df.index.name is None: 
        trace = go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1], mode='lines')
    else:
        trace = go.Scatter(x=df.index, y=df.iloc[:,1], mode='lines')
    data = [trace]

    if y_values is not None:
        for y in y_values:
            horizontal_line = go.Scatter(x=[df.iloc[0, 0], df.iloc[-1, 0]], y=[y, y], mode='lines', line=dict(color='red', dash='dash'))
            data.append(horizontal_line)

    layout = go.Layout(title='Time Series Data Movement', xaxis_title='Date', yaxis_title='Value')
    fig = go.Figure(data=data, layout=layout)
    fig.show()
    
def plot_count_frequency_histogram(df,column_index=1,second_window = None,show_latest_value=True): #column_index=1 means 2nd column
    fig = go.Figure()
    second_volumn_dt = df.iloc[:,column_index].dtype
    if second_volumn_dt != np.dtype('int64') and second_volumn_dt != np.dtype('float64'):
        raise Exception("The data type of the second column is not integer or float. Please make sure the second column is an integer or float.")
    else:
        value_column = df.iloc[:,column_index].values.tolist() #select the 2nd column
        fig.add_trace(go.Histogram(x=value_column,
                                        name = 'all records',
                                        marker_color='#330C73',
                                        opacity=0.75))

        if second_window != None:
            latest_perc_change_l = value_column[-second_window:]
            fig.add_trace(go.Histogram(x=latest_perc_change_l,
                                    name = 'last {}'.format(second_window),
                                    marker_color='#EB89B5',
                                    opacity=0.9))
        else:
            pass
        if show_latest_value == True:
            cur_per_change = value_column[-1:]
            fig.add_trace(go.Histogram(x=cur_per_change,
                                        name = 'latest',
                                        marker_color='#FF0000',
                                        opacity=1))
        else:
            pass

        fig.update_layout(
            barmode='overlay',
            title_text='Distribution Count - (time count : {})'.format(len(df)), # title of plot
            xaxis_title_text='Value', # xaxis label
            yaxis_title_text='Frequency Count', # yaxis label
        )
        fig.show()

