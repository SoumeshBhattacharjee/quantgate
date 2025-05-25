import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests
import feedparser
import time
import random
import logging
import concurrent.futures
from functools import wraps, lru_cache
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import re

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, initial_delay: int = 1):
    """Decorator to retry functions on failure with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=2)
def search_company(query: str) -> Dict[str, str]:
    """Search for companies with enhanced error handling and exchange support"""
    try:
        # Add exchange suffixes to the search query if not present
        exchanges = {
            'US': '',  # Default
            'India': '.NS',  # NSE
            'BSE': '.BO',    # Bombay Stock Exchange
            'Japan': '.T',   # Tokyo Stock Exchange
            'Hong Kong': '.HK',
            'Korea': '.KS',
            'London': '.L',
            'Germany': '.DE',
            'Paris': '.PA',
            'Canada': '.TO',  # Toronto Stock Exchange
            'Australia': '.AX',  # Australian Securities Exchange
            'Singapore': '.SI',  # Singapore Exchange
            'Brazil': '.SA',    # B3 (Brazilian Stock Exchange)
            'Mexico': '.MX'     # Mexican Stock Exchange
        }
        
        search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        search_results = response.json()
        if 'quotes' not in search_results or not search_results['quotes']:
            logger.warning(f"No results found for query: {query}")
            return {}
            
        # Process and format results
        formatted_results = {}
        for quote in search_results['quotes']:
            if 'shortname' in quote and 'symbol' in quote:
                # Get exchange name
                exchange = 'US'  # Default
                for ex_name, suffix in exchanges.items():
                    if quote['symbol'].endswith(suffix):
                        exchange = ex_name
                        break
                
                # Format the display name
                display_name = f"{quote['symbol']} - {quote['shortname']} ({exchange})"
                formatted_results[display_name] = quote['symbol']
        
        return formatted_results
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching for company {query}: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in company search: {str(e)}")
        return {}

def plot_atr_chart(hist, ticker):
    """Plot Average True Range chart"""
    if hist is None or hist.empty or 'ATR' not in hist.columns:
        return go.Figure()  # Return empty figure instead of None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['ATR'],
        name='ATR',
        line=dict(color='orange', width=2)  # Increased line width
    ))
    
    fig.update_layout(
        title=f"{ticker} Average True Range (ATR)",
        xaxis_title="Date",
        yaxis_title="ATR Value",
        template="plotly_dark",
        height=400,  # Increased height
        margin=dict(l=50, r=50, t=50, b=50),  # Added margins
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_stochastic_chart(hist, ticker):
    """Plot Stochastic Oscillator chart"""
    if hist is None or hist.empty or '%K' not in hist.columns:
        return go.Figure()  # Return empty figure instead of None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['%K'],
        name='%K',
        line=dict(color='blue', width=2)  # Increased line width
    ))
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['%D'],
        name='%D',
        line=dict(color='red', width=2)  # Increased line width
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5)
    
    fig.update_layout(
        title=f"{ticker} Stochastic Oscillator",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        height=400,  # Increased height
        margin=dict(l=50, r=50, t=50, b=50),  # Added margins
        yaxis=dict(range=[0, 100]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_obv_chart(hist, ticker):
    """Plot On-Balance Volume chart"""
    if hist is None or hist.empty or 'OBV' not in hist.columns:
        return go.Figure()  # Return empty figure instead of None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['OBV'],
        name='OBV',
        line=dict(color='purple', width=2)  # Increased line width
    ))
    
    fig.update_layout(
        title=f"{ticker} On-Balance Volume (OBV)",
        xaxis_title="Date",
        yaxis_title="OBV Value",
        template="plotly_dark",
        height=400,  # Increased height
        margin=dict(l=50, r=50, t=50, b=50),  # Added margins
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_ad_chart(hist, ticker):
    """Plot Accumulation/Distribution Line chart"""
    if hist is None or hist.empty or 'AD' not in hist.columns:
        return go.Figure()  # Return empty figure instead of None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['AD'],
        name='A/D Line',
        line=dict(color='cyan', width=2)  # Increased line width
    ))
    
    fig.update_layout(
        title=f"{ticker} Accumulation/Distribution Line",
        xaxis_title="Date",
        yaxis_title="A/D Value",
        template="plotly_dark",
        height=400,  # Increased height
        margin=dict(l=50, r=50, t=50, b=50),  # Added margins
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_macd_histogram(hist, ticker):
    """Plot MACD Histogram chart with improved error handling"""
    if hist is None or hist.empty:
        return go.Figure()  # Return empty figure instead of None
        
    try:
        # Verify required column exists
        if 'MACD_Hist' not in hist.columns:
            logger.error("Missing MACD_Hist column")
            return go.Figure()
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['MACD_Hist'],
            name='MACD Histogram',
            marker_color=hist['MACD_Hist'].apply(lambda x: 'green' if x >= 0 else 'red')
        ))
        
        fig.update_layout(
            title=f"{ticker} MACD Histogram",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            height=400,  # Increased height
            margin=dict(l=50, r=50, t=50, b=50),  # Added margins
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating MACD histogram: {str(e)}")
        return go.Figure()

def calculate_additional_indicators(df):
    """Calculate additional technical indicators"""
    if df is None or df.empty:
        return df
        
    try:
        # Calculate ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Calculate Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Calculate OBV (On-Balance Volume)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Calculate A/D Line (Accumulation/Distribution)
        money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        money_flow_volume = money_flow_multiplier * df['Volume']
        df['AD'] = money_flow_volume.cumsum()
        
        return df
    except Exception as e:
        logger.error(f"Error calculating additional indicators: {str(e)}")
        return df

# Constants for technical analysis
TECHNICAL_INDICATORS = {
    "SMA": [20, 50, 200],  # Simple Moving Averages
    "RSI": 14,  # RSI period
    "MACD": {"fast": 12, "slow": 26, "signal": 9},  # MACD parameters
    "BB": {"period": 20, "std": 2}  # Bollinger Bands parameters
}

# Chart types and their descriptions
CHART_TYPES = {
    "Candlestick": "Traditional Japanese candlestick chart",
    "OHLC": "Open-High-Low-Close bars",
    "Hollow Candles": "Hollow candlesticks (white body for up, black for down)",
    "Heikin-Ashi": "Modified candlestick averaging price movement",
    "Line": "Simple line chart of closing prices",
    "Area": "Area chart with filled region",
    "Bar": "Regular bar chart",
    "Range Bars": "High-Low range bars"
}

def calculate_heikin_ashi(df):
    """Calculate Heikin-Ashi candlestick data"""
    ha_df = pd.DataFrame(index=df.index)
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')] = (df['Open'][0] + df['Close'][0]) / 2
    ha_df['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    return ha_df

def calculate_technical_indicators(df):
    """Calculate technical indicators with improved error handling."""
    try:
        if df is None or df.empty:
            logger.error("No data provided for technical indicators")
            return None
            
        # Calculate SMAs
        for period in TECHNICAL_INDICATORS["SMA"]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=TECHNICAL_INDICATORS["MACD"]["fast"]).mean()
        exp2 = df['Close'].ewm(span=TECHNICAL_INDICATORS["MACD"]["slow"]).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=TECHNICAL_INDICATORS["MACD"]["signal"]).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=TECHNICAL_INDICATORS["RSI"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=TECHNICAL_INDICATORS["RSI"]).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        bb_period = TECHNICAL_INDICATORS["BB"]["period"]
        bb_std = TECHNICAL_INDICATORS["BB"]["std"]
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        
        # Calculate additional indicators
        df = calculate_additional_indicators(df)
        
        # Ensure all required columns exist
        required_columns = ['MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return None

# Enhanced caching configuration
CACHE_TTL = {
    'stock_data': 300,  # 5 minutes
    'quotes': 60,       # 1 minute
    'indices': 300,     # 5 minutes
    'news': 900,        # 15 minutes
    'company_info': 3600  # 1 hour
}

def get_live_price(ticker: str) -> Optional[float]:
    """Get the current live price of a stock"""
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1d", interval="1m")['Close'].iloc[-1]
    except Exception as e:
        logger.error(f"Error fetching live price for {ticker}: {str(e)}")
        return None

def plot_live_chart(ticker: str, chart_type: str = "Bar") -> Optional[go.Figure]:
    """Create a live updating chart with better error handling and multiple chart types"""
    if 'live_data' not in st.session_state:
        st.session_state.live_data = pd.DataFrame(columns=['Time', 'Price', 'Change'])
        st.session_state.last_price = None

    try:
        # Get current price
        current_price = get_live_price(ticker)
        
        if current_price is not None:
            # Add new price to the dataframe
            current_time = pd.Timestamp.now()
            
            # Calculate price change
            price_change = 0 if st.session_state.last_price is None else current_price - st.session_state.last_price
            new_data = pd.DataFrame({
                'Time': [current_time], 
                'Price': [current_price],
                'Change': [price_change]
            })
            st.session_state.live_data = pd.concat([st.session_state.live_data, new_data], ignore_index=True)
            
            # Keep only last 30 data points for smoother visualization
            if len(st.session_state.live_data) > 30:
                st.session_state.live_data = st.session_state.live_data.iloc[-30:]
            
            st.session_state.last_price = current_price
            
            # Create figure based on chart type
            fig = go.Figure()
            
            if chart_type == "Bar":
                fig.add_trace(go.Bar(
                    x=st.session_state.live_data['Time'],
                    y=st.session_state.live_data['Price'],
                    marker_color=st.session_state.live_data['Change'].apply(
                        lambda x: 'green' if x >= 0 else 'red'
                    ),
                    text=st.session_state.live_data['Price'].round(2),
                    textposition='auto',
                ))
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=st.session_state.live_data['Time'],
                    y=st.session_state.live_data['Price'],
                    mode='lines+markers',
                    line=dict(color='white', width=2),
                    marker=dict(
                        color=st.session_state.live_data['Change'].apply(
                            lambda x: 'green' if x >= 0 else 'red'
                        ),
                        size=8
                    ),
                    text=st.session_state.live_data['Price'].round(2),
                    textposition='top center',
                ))
            elif chart_type == "Area":
                fig.add_trace(go.Scatter(
                    x=st.session_state.live_data['Time'],
                    y=st.session_state.live_data['Price'],
                    fill='tozeroy',
                    mode='lines',
                    line=dict(color='lightblue', width=2),
                    fillcolor='rgba(173, 216, 230, 0.3)',
                    text=st.session_state.live_data['Price'].round(2),
                    textposition='top center',
                ))
            elif chart_type == "Candlestick":
                # Create OHLC data for candlestick
                ohlc_data = st.session_state.live_data.set_index('Time').resample('1min').agg({
                    'Price': ['first', 'max', 'min', 'last']
                }).dropna()
                ohlc_data.columns = ['Open', 'High', 'Low', 'Close']
                
                fig.add_trace(go.Candlestick(
                    x=ohlc_data.index,
                    open=ohlc_data['Open'],
                    high=ohlc_data['High'],
                    low=ohlc_data['Low'],
                    close=ohlc_data['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
            
            # Update layout with better styling and animation settings
            fig.update_layout(
                title=dict(
                    text=f"{ticker} Live Price Movement (Current: ${current_price:.2f})",
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=500,
                yaxis=dict(
                    range=[
                        st.session_state.live_data['Price'].min() * 0.9995,
                        st.session_state.live_data['Price'].max() * 1.0005
                    ]
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 300}
                                    }
                                ]
                            )
                        ]
                    )
                ]
            )
            
            return fig
    except Exception as e:
        logger.error(f"Error creating live chart for {ticker}: {str(e)}")
        return None
    return None

def validate_ticker(ticker):
    """Validate if a ticker symbol is valid."""
    try:
        # Check if ticker is empty or None
        if not ticker or not isinstance(ticker, str):
            return False
            
        # Check if ticker contains only valid characters
        if not re.match(r'^[A-Z0-9.]+$', ticker):
            return False
            
        # Try to get basic info for the ticker
        stock = yf.Ticker(ticker)
        info = stock.info
        return bool(info and 'regularMarketPrice' in info)
    except Exception as e:
        logger.error(f"Error validating ticker {ticker}: {str(e)}")
        return False

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Stock tips for the loading screen
STOCK_TIPS = [
    "💡 Diversification is key to reducing investment risk.",
    "📈 Dollar-cost averaging can help reduce the impact of market volatility.",
    "🎯 Always have a clear investment strategy and stick to it.",
    "⏰ Time in the market beats timing the market.",
    "📊 Past performance doesn't guarantee future results.",
    "💼 Never invest more than you can afford to lose.",
    "🔍 Research companies thoroughly before investing.",
    "📱 Stay updated with market news and company announcements.",
    "🎢 Emotional trading often leads to poor investment decisions.",
    "💰 Consider dividend-paying stocks for steady income.",
    "📚 Keep learning about different investment strategies.",
    "🛡️ Use stop-loss orders to protect against significant losses.",
    "🏦 Understand the company's business model before investing.",
    "📅 Regular portfolio rebalancing helps maintain your risk level.",
    "🌍 Consider global markets for diversification opportunities."
]

# Function to show random tips during loading
def show_loading_tip():
    return random.choice(STOCK_TIPS)

# Global exchange indices with names and market caps
MARKET_CURRENCIES = {
    "US": "USD",
    "Asia": {
        "^N225": "JPY",
        "^HSI": "HKD",
        "^BSESN": "INR",
        "^NSEI": "INR"
    },
    "Europe": {
        "^FTSE": "GBP",
        "^GDAXI": "EUR",
        "^FCHI": "EUR"
    }
}

GLOBAL_INDICES = {
    "US": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite"
    },
    "Asia": {
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "^BSESN": "BSE SENSEX",
        "^NSEI": "NIFTY 50"
    },
    "Europe": {
        "^FTSE": "FTSE 100",
        "^GDAXI": "DAX",
        "^FCHI": "CAC 40"
    }
}

# Stock currencies based on region and suffix
STOCK_CURRENCIES = {
    "US": {"symbol": "$", "code": "USD"},
    "HK": {"symbol": "HK$", "code": "HKD"},
    "IN": {"symbol": "₹", "code": "INR"},
    "JP": {"symbol": "¥", "code": "JPY"},
    "GB": {"symbol": "£", "code": "GBP"},
    "EU": {"symbol": "€", "code": "EUR"},
    "KR": {"symbol": "₩", "code": "KRW"},
    "CN": {"symbol": "¥", "code": "CNY"}
}

def get_currency_for_ticker(ticker):
    """Get the currency symbol and code for a ticker based on its suffix"""
    if ticker.endswith('.NS'):
        return STOCK_CURRENCIES['IN']
    elif ticker.endswith('.HK'):
        return STOCK_CURRENCIES['HK']
    elif ticker.endswith('.T'):
        return STOCK_CURRENCIES['JP']
    elif ticker.endswith('.KS'):
        return STOCK_CURRENCIES['KR']
    elif ticker.endswith('.L'):
        return STOCK_CURRENCIES['GB']
    elif ticker.endswith('.DE') or ticker.endswith('.PA'):
        return STOCK_CURRENCIES['EU']
    else:
        return STOCK_CURRENCIES['US']  # Default to USD

# Top stocks for the navigation bar
TOP_STOCKS = {
    "US": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "UNH", "JPM",
        "V", "JNJ", "WMT", "PG", "MA"
    ],
    "Asia": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "7203.T", "9432.T", "9984.T"
    ],    "Europe": [
        "GSK.L", "RR.L", "HSBA.L", "BP.L", "SHEL.L",
        "VOW3.DE", "SAP.DE", "SIE.DE"
    ]
}

def get_next_stocks(current_stocks, count=10):
    """Get the next batch of stocks in a circular manner"""
    all_stocks = []
    for region, stocks in TOP_STOCKS.items():
        all_stocks.extend(stocks)
    
    if not current_stocks:
        return all_stocks[:count]
    
    try:
        last_index = all_stocks.index(current_stocks[-1])
    except ValueError:
        return all_stocks[:count]
    
    next_stocks = []
    total_stocks = len(all_stocks)
    for i in range(count):
        index = (last_index + i + 1) % total_stocks
        next_stocks.append(all_stocks[index])
    
    return next_stocks

@st.cache_data(ttl=60)  # Cache for 1 minute
@retry_on_failure(max_retries=2)
def get_stock_quotes(stocks: list) -> list:
    """Get stock quotes with caching and error handling"""
    stock_data = []
    
    for stock in stocks:
        try:
            # Validate ticker first
            if not validate_ticker(stock):
                logger.warning(f"Invalid ticker symbol: {stock}")
                continue

            ticker = yf.Ticker(stock)
            hist = ticker.history(period="1d")
            if not hist.empty:
                last_close = hist['Close'].iloc[-1]
                prev_close = ticker.history(period="2d")['Close'].iloc[-2]
                change_pct = ((last_close - prev_close) / prev_close) * 100
                
                currency = get_currency_for_ticker(stock)
                stock_data.append({
                    'Symbol': stock,
                    'Price': round(last_close, 2),
                    'Currency': currency,
                    'Change': round(change_pct, 2)
                })
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Ticker not found: {stock}")
            else:
                logger.warning(f"HTTP error for {stock}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error fetching data for {stock}: {str(e)}")
            continue
            
    return stock_data

def create_navigation_bar():
    """Create a fixed navigation bar with top stocks"""
    if 'current_stocks' not in st.session_state:
        st.session_state.current_stocks = []
        
    display_stocks = get_next_stocks(st.session_state.current_stocks)
    st.session_state.current_stocks = display_stocks
    
    # Create a placeholder for the ticker display
    ticker_placeholder = st.empty()
    
    # Show loading state with stock tip
    with ticker_placeholder.container():
        st.markdown("""
            <div style='text-align: center; background-color: #1E1E1E; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <div style='margin: 10px; padding: 10px; background-color: #2E2E2E; border-radius: 5px;'>
                    <p style='font-size: 14px;'>{}</p>
                </div>
                <p>Loading market data in the background...</p>
            </div>
        """.format(show_loading_tip()), unsafe_allow_html=True)
    
    # Fetch stock data
    stock_data = get_stock_quotes(display_stocks)
    
    if stock_data:
        df = pd.DataFrame(stock_data)
        df = df.sort_values('Change', ascending=False)
        
        # Update the ticker display
        with ticker_placeholder.container():
            st.write("""
                <style>
                    .stock-nav {
                        background-color: #1E1E1E;
                        padding: 10px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                        border: 1px solid #333;
                    }
                    .stock-ticker {
                        display: inline-block;
                        padding: 5px 15px;
                        margin: 0 10px;
                        border-radius: 3px;
                        background-color: #2E2E2E;
                    }
                    .stock-up { color: #00FF00; }
                    .stock-down { color: #FF0000; }
                </style>
            """, unsafe_allow_html=True)
            
            ticker_html = []
            for _, row in df.iterrows():
                color_class = "stock-up" if row['Change'] >= 0 else "stock-down"
                currency_symbol = row['Currency']['symbol']
                ticker_html.append(
                    f'<div class="stock-ticker {color_class}">{row["Symbol"]}: '
                    f'{currency_symbol}{row["Price"]:,.2f} ({row["Change"]:+.2f}%)</div>'
                )
            
            ticker_display = '<marquee scrollamount="3" onmouseover="this.stop();" onmouseout="this.start();" loop="-1">'
            ticker_display += ''.join(ticker_html)
            ticker_display += '</marquee>'
            
            st.write(f'<div class="stock-nav">{ticker_display}</div>', unsafe_allow_html=True)
        
        # Auto refresh using Streamlit's rerun mechanism
        time.sleep(30)
        st.rerun()

# Function to get global indices data with names
@st.cache_data(ttl=300)
def get_global_indices_data():
    data = []
    for region, indices in GLOBAL_INDICES.items():
        for symbol, name in indices.items():
            try:
                idx = yf.Ticker(symbol)
                info = idx.history(period="1d")
                if not info.empty:
                    last_close = info['Close'].iloc[-1]
                    prev_close = idx.history(period="2d")['Close'].iloc[-2]
                    change_pct = ((last_close - prev_close) / prev_close) * 100
                    
                    # Get the appropriate currency
                    if region == "US":
                        currency = MARKET_CURRENCIES["US"]
                    elif region == "Asia":
                        currency = MARKET_CURRENCIES["Asia"].get(symbol, "USD")
                    else:  # Europe
                        currency = MARKET_CURRENCIES["Europe"].get(symbol, "EUR")
                    
                    data.append({
                        'Region': region,
                        'Symbol': symbol,
                        'Name': name,
                        'Price': round(last_close, 2),
                        'Currency': currency,
                        'Change %': round(change_pct, 2)
                    })
            except Exception:
                continue
    return pd.DataFrame(data)

def render_global_markets():
    """Render the global markets section with index names and search"""
    indices_data = get_global_indices_data()
    if not indices_data.empty:
        # Add search functionality with currency filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input("🔍 Search Markets", "").lower()
        with col2:
            available_currencies = sorted(indices_data['Currency'].unique())
            currency_filter = st.multiselect("Filter by Currency", available_currencies, default=available_currencies)
        
        filtered_data = indices_data
        if search:
            filtered_data = indices_data[
                indices_data['Name'].str.lower().str.contains(search) |
                indices_data['Symbol'].str.lower().str.contains(search) |
                indices_data['Region'].str.lower().str.contains(search)
            ]
        
        if currency_filter:
            filtered_data = filtered_data[filtered_data['Currency'].isin(currency_filter)]
        
        # Two-column layout for regions
        cols = st.columns(2)
        regions = filtered_data['Region'].unique()
        
        for i, region in enumerate(regions):
            with cols[i % 2]:
                region_data = filtered_data[filtered_data['Region'] == region]
                st.markdown(f"### {region} Markets")
                # Format the data
                formatted_data = region_data[['Name', 'Symbol', 'Price', 'Currency', 'Change %']].copy()
                # Format price with currency
                formatted_data['Price'] = formatted_data.apply(
                    lambda x: f"{x['Price']:,.2f} {x['Currency']}", axis=1
                )
                formatted_data['Change %'] = formatted_data['Change %'].apply(
                    lambda x: f"{'🔴' if x < 0 else '🟢'} {x:+.2f}%"
                )
                
                # Remove currency column after formatting price
                formatted_data = formatted_data.drop('Currency', axis=1)
                
                # Display in a clean table
                st.dataframe(
                    formatted_data,
                    hide_index=True,
                    column_config={
                        'Name': st.column_config.TextColumn(
                            'Index Name',
                            width='medium'
                        ),
                        'Symbol': st.column_config.TextColumn(
                            'Symbol',
                            width='small'
                        ),
                        'Price': st.column_config.TextColumn(
                            'Price',
                            width='medium'
                        ),
                        'Change %': st.column_config.TextColumn(
                            'Change',
                            width='small'
                        )
                    }
                )

@lru_cache(maxsize=100)
def get_cached_stock_info(ticker: str) -> Optional[Dict[str, Any]]:
    """Cache stock info to reduce API calls with enhanced error handling"""
    try:
        if not validate_ticker(ticker):
            return None
            
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            logger.warning(f"No info available for {ticker}")
            return None
            
        # Validate essential fields
        required_fields = ['regularMarketPrice', 'regularMarketPreviousClose']
        if not all(field in info for field in required_fields):
            logger.warning(f"Missing required fields for {ticker}")
            return None
            
        return info
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching stock info for {ticker}: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching stock info for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL['stock_data'])
@retry_on_failure(max_retries=3)
def get_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Get stock data with enhanced error handling and validation"""
    try:
        if not validate_ticker(ticker):
            logger.warning(f"Invalid ticker symbol: {ticker}")
            return None
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            logger.warning(f"No data available for {ticker}")
            return None
            
        # Basic data validation
        if hist['Close'].isnull().any():
            logger.warning(f"Missing closing prices for {ticker}")
            hist = hist.dropna(subset=['Close'])
            
        if len(hist) < 2:
            logger.warning(f"Insufficient data points for {ticker}")
            return None
            
        # Calculate essential metrics
        hist['Returns'] = hist['Close'].pct_change()
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        
        return hist
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {ticker}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL['quotes'])
def get_stock_quotes(stocks: List[str]) -> List[Dict[str, Any]]:
    """Get stock quotes with optimized parallel processing and error handling"""
    if not stocks:
        return []
        
    def fetch_quote(stock: str) -> Optional[Dict[str, Any]]:
        try:
            # Validate ticker first
            if not validate_ticker(stock):
                logger.warning(f"Invalid ticker symbol: {stock}")
                return None

            # Use cached stock info if available
            stock_info = get_cached_stock_info(stock)
            if not stock_info:
                logger.warning(f"Could not fetch info for {stock}")
                return None
                
            if 'regularMarketPrice' not in stock_info:
                logger.warning(f"No market price available for {stock}")
                return None
                
            current_price = stock_info['regularMarketPrice']
            prev_close = stock_info.get('regularMarketPreviousClose', current_price)
            
            if current_price <= 0 or prev_close <= 0:
                logger.warning(f"Invalid price data for {stock}")
                return None
                
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            return {
                'Symbol': stock,
                'Price': round(current_price, 2),
                'Currency': get_currency_for_ticker(stock),
                'Change': round(change_pct, 2)
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Ticker not found: {stock}")
            else:
                logger.warning(f"HTTP error for {stock}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error fetching data for {stock}: {str(e)}")
        return None
    
    # Use ThreadPoolExecutor with a smaller number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        try:
            results = list(executor.map(fetch_quote, stocks))
            return [r for r in results if r is not None]
        except Exception as e:
            logger.error(f"Error in parallel quote fetching: {str(e)}")
            return []

def plot_stock_chart(hist, ticker, chart_type="Candlestick"):
    """Enhanced chart plotting with multiple types and better error handling"""
    if hist is None or hist.empty:
        logger.warning(f"No data available for {ticker}")
        return None
        
    try:
        currency = get_currency_for_ticker(ticker)['code']
        fig = go.Figure()
        
        # Calculate Heikin-Ashi if needed
        ha_data = calculate_heikin_ashi(hist) if chart_type == "Heikin-Ashi" else None
        
        # Add main chart trace based on type
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Candlestick',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        elif chart_type == "OHLC":
            fig.add_trace(go.Ohlc(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        elif chart_type == "Hollow Candles":
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Hollow Candles',
                increasing=dict(fillcolor='white', line_color='green'),
                decreasing=dict(fillcolor='black', line_color='red')
            ))
        elif chart_type == "Heikin-Ashi":
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=ha_data['HA_Open'],
                high=ha_data['HA_High'],
                low=ha_data['HA_Low'],
                close=ha_data['HA_Close'],
                name='Heikin-Ashi',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='Close',
                line=dict(color='white', width=2)
            ))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='Close',
                fill='tozeroy',
                fillcolor='rgba(173, 216, 230, 0.3)',
                line=dict(color='lightblue', width=2)
            ))
        elif chart_type == "Bar":
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Close'],
                name='Close',
                marker_color='lightblue'
            ))
        elif chart_type == "Range Bars":
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['High'] - hist['Low'],
                base=hist['Low'],
                name='Range',
                marker_color=hist.apply(
                    lambda x: 'green' if x['Close'] >= x['Open'] else 'red',
                    axis=1
                )
            ))
        
        # Add technical indicators if available
        for period in [20, 50, 200]:
            if f'SMA_{period}' in hist.columns:
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist[f'SMA_{period}'],
                    name=f'SMA {period}',
                    line=dict(width=1, dash='dash')
                ))
        
        # Update layout with better styling
        fig.update_layout(
            title=dict(
                text=f"{ticker} - {chart_type} Chart",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            xaxis_title="Date",
            yaxis_title=f"Price ({currency})",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating chart for {ticker}: {str(e)}")
        return None

# Additional Chart: Volume Trend
def plot_volume_chart(hist, ticker):
    """Plot Volume chart"""
    if hist is None or hist.empty:
        return go.Figure()  # Return empty figure instead of None
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f"{ticker} Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_dark",
        height=400,  # Increased height
        margin=dict(l=50, r=50, t=50, b=50),  # Added margins
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# Fetch ticker prices
def get_ticker_prices(tickers):
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            prices[ticker] = round(stock.history(period="1d")["Close"][0], 2)
        except Exception:
            prices[ticker] = "N/A"
    return prices

# Portfolio and dummy trade system
if "trades" not in st.session_state:
    st.session_state.trades = []

def place_trade():
    trade = {
        "Ticker": st.session_state.trade_ticker,
        "Type": st.session_state.trade_type,
        "Quantity": st.session_state.trade_quantity,
        "Stop Loss": st.session_state.trade_stop_loss,
        "Entry Price": get_ticker_prices([st.session_state.trade_ticker]).get(st.session_state.trade_ticker, 0)
    }
    st.session_state.trades.append(trade)

# Fetch finance news from Yahoo Finance RSS feed
@st.cache_data(ttl=CACHE_TTL['news'])
def get_finance_news() -> List[Dict[str, str]]:
    """Get finance news with caching and error handling"""
    try:
        rss_url = "https://finance.yahoo.com/rss/topstories"
        feed = feedparser.parse(rss_url)
        articles = []
        
        for entry in feed.entries[:5]:  # Limit to top 5 articles
            try:
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": entry.get("summary", "No summary available."),
                    "published": entry.get("published", "No date available.")
                })
            except Exception as e:
                logger.warning(f"Error processing news article: {str(e)}")
                continue
                
        return articles
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return [{
            "title": "Error Loading News",
            "link": "#",
            "summary": "Unable to fetch news at this time. Please try again later.",
            "published": "N/A"
        }]

# Candlestick patterns
CANDLESTICK_PATTERNS = {
    "Doji": lambda df: abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1,
    "Hammer": lambda df: (df['Low'] < df['Open']) & (df['Low'] < df['Close']) & 
                        ((df['High'] - df['Close']) < 0.1 * (df['High'] - df['Low'])) &
                        ((df['Close'] - df['Low']) > 2 * (df['Open'] - df['Close'])),
    "Engulfing Bullish": lambda df: (df['Open'].shift(1) > df['Close'].shift(1)) & 
                                   (df['Close'] > df['Open']) & 
                                   (df['Open'] <= df['Close'].shift(1)) & 
                                   (df['Close'] >= df['Open'].shift(1))
}

def plot_rsi_chart(hist, ticker):
    """Plot RSI chart"""
    if hist is None or hist.empty or 'RSI' not in hist.columns:
        return go.Figure()  # Return empty figure instead of None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['RSI'],
        name='RSI',
        line=dict(color='purple', width=2)  # Increased line width
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    
    fig.update_layout(
        title=f"{ticker} RSI",
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_dark",
        height=400,  # Increased height
        margin=dict(l=50, r=50, t=50, b=50),  # Added margins
        yaxis=dict(range=[0, 100]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_macd_chart(hist, ticker):
    """Plot MACD chart with improved error handling"""
    if hist is None or hist.empty:
        return go.Figure()  # Return empty figure instead of None
        
    try:
        # Verify required columns exist
        required_columns = ['MACD', 'MACD_Signal']
        if not all(col in hist.columns for col in required_columns):
            logger.error(f"Missing required columns for MACD chart: {required_columns}")
            return go.Figure()
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MACD'],
            name='MACD',
            line=dict(color='blue', width=2)  # Increased line width
        ))
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MACD_Signal'],
            name='Signal',
            line=dict(color='orange', width=2)  # Increased line width
        ))
        
        fig.update_layout(
            title=f"{ticker} MACD",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            height=400,  # Increased height
            margin=dict(l=50, r=50, t=50, b=50),  # Added margins
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating MACD chart: {str(e)}")
        return go.Figure()

def plot_bollinger_bands(hist, ticker):
    """Plot Bollinger Bands chart"""
    if hist is None or hist.empty or 'BB_Upper' not in hist.columns:
        return go.Figure()  # Return empty figure instead of None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        name='Price',
        line=dict(color='white', width=2)  # Increased line width
    ))
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['BB_Upper'],
        name='Upper Band',
        line=dict(color='gray', dash='dash', width=1.5)  # Increased line width
    ))
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['BB_Middle'],
        name='Middle Band',
        line=dict(color='yellow', dash='dash', width=1.5)  # Increased line width
    ))
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['BB_Lower'],
        name='Lower Band',
        line=dict(color='gray', dash='dash', width=1.5),  # Increased line width
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=f"{ticker} Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=400,  # Increased height
        margin=dict(l=50, r=50, t=50, b=50),  # Added margins
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# Fetch top movers data and create navigation bar
def create_top_movers_navbar():
    try:
        # Get top gainers and losers
        gainers = yf.Ticker("^GSPC").history(period="1d")["Close"].pct_change().nlargest(5)
        losers = yf.Ticker("^GSPC").history(period="1d")["Close"].pct_change().nsmallest(5)
        
        # Format data for display
        gainers = gainers.reset_index().rename(columns={"Close": "Change %"})
        losers = losers.reset_index().rename(columns={"Close": "Change %"})
        
        # Navigation bar
        st.sidebar.markdown("### 📈 Top Movers")
        selected_mover = st.sidebar.radio(
            "Select Gainers or Losers",
            ("Top Gainers", "Top Losers")
        )
        
        if selected_mover == "Top Gainers":
            st.sidebar.dataframe(gainers)
        else:
            st.sidebar.dataframe(losers)
    except Exception as e:
        st.sidebar.error(f"Error fetching top movers: {str(e)}")

# Enhanced portfolio class with better error handling and validation
class Portfolio:
    def __init__(self):
        self.positions: Dict[str, Dict[str, float]] = {}
        self._cache = {}
        self._last_update = 0
        self._cache_ttl = CACHE_TTL['stock_data']

    def add_position(self, ticker: str, quantity: float, price: float) -> bool:
        """Add or update a position with enhanced validation"""
        if not validate_ticker(ticker):
            logger.warning(f"Invalid ticker symbol: {ticker}")
            return False
            
        try:
            quantity = float(quantity)
            price = float(price)
            
            if quantity <= 0 or price <= 0:
                logger.warning(f"Invalid quantity or price for {ticker}")
                return False
                
            self.positions[ticker] = {
                'quantity': quantity,
                'price': price,
                'last_update': time.time()
            }
            self._invalidate_cache()
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Error adding position for {ticker}: {str(e)}")
            return False

    def _invalidate_cache(self):
        """Invalidate the portfolio cache"""
        self._cache = {}
        self._last_update = 0

    @retry_on_failure(max_retries=3)
    def get_portfolio_value(self) -> Dict[str, Any]:
        """Get portfolio value with enhanced error handling and caching"""
        current_time = time.time()
        
        if (current_time - self._last_update) < self._cache_ttl and self._cache:
            return self._cache
            
        total_value = 0.0
        position_values = {}
        errors = []
        
        for ticker, position in self.positions.items():
            try:
                stock = yf.Ticker(ticker)
                current_price = stock.info.get('regularMarketPrice', 0)
                
                if current_price <= 0:
                    errors.append(f"Invalid price for {ticker}")
                    continue
                    
                position_value = position['quantity'] * current_price
                total_value += position_value
                
                position_values[ticker] = {
                    'quantity': position['quantity'],
                    'current_price': current_price,
                    'value': position_value,
                    'currency': get_currency_for_ticker(ticker)
                }
            except Exception as e:
                errors.append(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        result = {
            'total_value': total_value,
            'positions': position_values,
            'errors': errors,
            'last_update': current_time
        }
        
        self._cache = result
        self._last_update = current_time
        
        return result

    def get_position_summary(self) -> pd.DataFrame:
        """Get a summary of all positions as a DataFrame"""
        portfolio_data = self.get_portfolio_value()
        if not portfolio_data['positions']:
            return pd.DataFrame()
            
        data = []
        for ticker, pos in portfolio_data['positions'].items():
            data.append({
                'Ticker': ticker,
                'Quantity': pos['quantity'],
                'Current Price': pos['current_price'],
                'Value': pos['value'],
                'Currency': pos['currency']['symbol']
            })
            
        return pd.DataFrame(data)

# Initialize portfolio in session state if not exists
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio()

# Initialize session state for stock rotation
if 'current_stocks' not in st.session_state:
    st.session_state.current_stocks = []

# Streamlit UI
def main():
    try:
        st.set_page_config(layout="wide", page_title="Stock Dashboard", initial_sidebar_state="collapsed")
        st.title("QUANTGATE")

        # Initialize all session state variables at the start
        session_states = {
            'show_technical': False,
            'show_global_markets': False,
            'show_portfolio': False,
            'show_news': False,
            'live_data': pd.DataFrame(columns=['Time', 'Price', 'Change']),
            'last_price': None,
            'live_updating': False,
            'current_stocks': [],
            'trades': [],
            'portfolio': Portfolio()
        }
        
        for state, default_value in session_states.items():
            if state not in st.session_state:
                st.session_state[state] = default_value

        # Create a placeholder for the navigation bar
        nav_placeholder = st.empty()

        # Main Input with Company Search
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            company_search = st.text_input("Search by Company Name or Symbol", "")
            if company_search:
                try:
                    search_results = search_company(company_search)
                    if search_results:
                        options = search_results
                        selected = st.selectbox("Select Company", options.keys())
                        ticker = options[selected]
                        st.info(f"Selected: {selected}")
                    else:
                        st.warning("No companies found matching your search. Try a different name or symbol.")
                        ticker = "AAPL"
                except Exception as e:
                    logger.error(f"Error in company search: {str(e)}")
                    st.error("Error searching for companies. Please try again.")
                    ticker = "AAPL"
            else:
                ticker = st.text_input("Or Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL").upper()
                if not validate_ticker(ticker):
                    st.warning("Invalid ticker symbol. Using default (AAPL).")
                    ticker = "AAPL"
        
        with col2:
            period = st.selectbox("Select Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=6)
        
        with col3:
            chart_type = st.selectbox("Chart Type", list(CHART_TYPES.keys()), 
                                    help="Hover over each option to see description")
        
        # Fetch and display stock data with error handling
        try:
            hist = get_stock_data(ticker, period)
            if hist is None or hist.empty:
                st.error(f"No data available for {ticker}. Please try a different ticker or time period.")
                return
                
            # Calculate technical indicators
            hist = calculate_technical_indicators(hist)
            if hist is None:
                st.error("Error calculating technical indicators. Please try again.")
                return
                
            # Create tabs for different views
            main_tab, live_tab, technical_tab, portfolio_tab, news_tab = st.tabs([
                "Main Chart", "Live View", "Technical Analysis", "Portfolio", "News"
            ])
            
            with main_tab:
                try:
                    # Show the selected chart type
                    chart_fig = plot_stock_chart(hist, ticker, chart_type)
                    if chart_fig:
                        st.plotly_chart(chart_fig, use_container_width=True, key="main_chart")
                    else:
                        st.error("Error creating chart. Please try a different chart type.")
                except Exception as e:
                    logger.error(f"Error in main chart: {str(e)}")
                    st.error("Error displaying main chart. Please try again.")
            
            with live_tab:
                try:
                    st.subheader("Live Price Movement")
                    # Initialize live data if not exists
                    if 'live_data' not in st.session_state:
                        st.session_state.live_data = pd.DataFrame(columns=['Time', 'Price', 'Change'])
                        st.session_state.last_price = None
                    
                    # Add chart type selector
                    live_chart_type = st.selectbox(
                        "Select Live Chart Type",
                        ["Bar", "Line", "Area", "Candlestick"],
                        help="Choose how to visualize the live price data"
                    )
                    
                    # Add a button to start/stop live updates
                    if 'live_updating' not in st.session_state:
                        st.session_state.live_updating = False
                    
                    if st.button("Start Live Updates" if not st.session_state.live_updating else "Stop Live Updates"):
                        st.session_state.live_updating = not st.session_state.live_updating
                    
                    if st.session_state.live_updating:
                        live_fig = plot_live_chart(ticker, live_chart_type)
                        if live_fig:
                            st.plotly_chart(live_fig, use_container_width=True, key="live_chart")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Error creating live chart. Please try again.")
                    else:
                        st.info("Click 'Start Live Updates' to begin real-time price tracking")
                except Exception as e:
                    logger.error(f"Error in live view: {str(e)}")
                    st.error("Error in live view. Please try again.")
            
            with technical_tab:
                try:
                    st.header("Technical Analysis")
                    
                    # Add spacing between sections
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # First row of charts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_volume_chart(hist, ticker), use_container_width=True, key="volume_chart")
                    with col2:
                        st.plotly_chart(plot_rsi_chart(hist, ticker), use_container_width=True, key="rsi_chart")
                    
                    # Add spacing between rows
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Second row of charts
                    col3, col4 = st.columns(2)
                    with col3:
                        st.plotly_chart(plot_macd_chart(hist, ticker), use_container_width=True, key="macd_chart")
                    with col4:
                        st.plotly_chart(plot_bollinger_bands(hist, ticker), use_container_width=True, key="bollinger_chart")
                    
                    # Add spacing between rows
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Third row of charts
                    col5, col6 = st.columns(2)
                    with col5:
                        st.plotly_chart(plot_atr_chart(hist, ticker), use_container_width=True, key="atr_chart")
                    with col6:
                        st.plotly_chart(plot_stochastic_chart(hist, ticker), use_container_width=True, key="stochastic_chart")
                    
                    # Add spacing between rows
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Fourth row of charts
                    col7, col8 = st.columns(2)
                    with col7:
                        st.plotly_chart(plot_obv_chart(hist, ticker), use_container_width=True, key="obv_chart")
                    with col8:
                        st.plotly_chart(plot_ad_chart(hist, ticker), use_container_width=True, key="ad_chart")
                    
                    # Add spacing between rows
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Fifth row - MACD Histogram
                    st.plotly_chart(plot_macd_histogram(hist, ticker), use_container_width=True, key="macd_hist_chart")
                    
                    # Add spacing at the bottom
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                except Exception as e:
                    logger.error(f"Error in technical analysis: {str(e)}")
                    st.error("Error displaying technical analysis. Please try again.")
            
            with portfolio_tab:
                try:
                    st.subheader("Portfolio Management")
                    portfolio_value = st.session_state.portfolio.get_portfolio_value()
                    
                    # Portfolio Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Portfolio Value", f"${portfolio_value['total_value']:,.2f}")
                    with col2:
                        if portfolio_value.get('errors'):
                            st.warning("Some positions could not be updated")
                    
                    # Portfolio Positions Table
                    if not portfolio_value['positions']:
                        st.info("No positions found in the portfolio.")
                    else:
                        st.dataframe(st.session_state.portfolio.get_position_summary(), use_container_width=True)
                    
                    # Trade Entry Form
                    st.subheader("Place New Trade")
                    trade_col1, trade_col2 = st.columns(2)
                    with trade_col1:
                        st.text_input("Trade Symbol", key="trade_ticker")
                        st.number_input("Quantity", min_value=1, value=100, key="trade_quantity")
                    with trade_col2:
                        st.selectbox("Trade Type", ["Buy", "Sell"], key="trade_type")
                        st.number_input("Stop Loss", min_value=0.0, value=0.0, key="trade_stop_loss")
                    
                    if st.button("Place Trade"):
                        if validate_ticker(st.session_state.trade_ticker):
                            place_trade()
                            st.success(f"Trade placed for {st.session_state.trade_ticker}")
                        else:
                            st.error("Invalid ticker symbol")
                except Exception as e:
                    logger.error(f"Error in portfolio management: {str(e)}")
                    st.error("Error in portfolio management. Please try again.")
            
            with news_tab:
                try:
                    st.subheader("Latest Market News")
                    news = get_finance_news()
                    for article in news:
                        st.markdown(f"### {article['title']}")
                        st.markdown(f"{article['summary']}")
                        st.markdown(f"[Read more]({article['link']})")
                        st.markdown("---")
                except Exception as e:
                    logger.error(f"Error in news section: {str(e)}")
                    st.error("Error loading news. Please try again.")

            # Global Markets Section
            try:
                st.subheader("Global Markets")
                render_global_markets()
            except Exception as e:
                logger.error(f"Error in global markets: {str(e)}")
                st.error("Error loading global markets. Please try again.")

        except Exception as e:
            logger.error(f"Error in main data processing: {str(e)}")
            st.error("An error occurred while processing your request. Please try again.")
            return

        # Load the navigation bar last
        try:
            with nav_placeholder:
                create_navigation_bar()
        except Exception as e:
            logger.error(f"Error in navigation bar: {str(e)}")
            st.error("Error loading navigation bar. Please refresh the page.")

    except Exception as e:
        logger.error(f"Critical error in main function: {str(e)}")
        st.error("A critical error occurred. Please refresh the page or try again later.")
        return

if __name__ == "__main__":
    main()
