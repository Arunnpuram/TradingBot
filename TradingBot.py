import alpaca_trade_api as tradeapi
import websocket
import json
import threading
import time
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Alpaca API credentials
ALPACA_API_KEY = "your_alpaca_api_key_here" #Aplaca API, set it as a environment variable and dont hard code your API key in the script
ALPACA_SECRET_KEY = "your_alpaca_secret_key_here"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Use this for paper trading

# TradingView WebSocket details
TRADINGVIEW_SOCKET = "wss://data.tradingview.com/socket.io/websocket"

# Trading parameters
SYMBOL = "AAPL"
TIMEFRAME = "1"  # 1 minute timeframe
QUANTITY = 10  # Number of shares to trade
STOP_LOSS_PERCENTAGE = 0.01  # 1% stop loss, Change accordingly 

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL, api_version='v2')

# Global variables
current_position = 0
last_trade_price = 0
stop_loss_price = 0
price_history = []
high_history = []
low_history = []

# Define trading patterns
class TradingPatterns:
    @staticmethod
    def moving_average_crossover(data, short_window=10, long_window=50):
        if len(data) < long_window:
            return 0
        short_ma = np.mean(data[-short_window:])
        long_ma = np.mean(data[-long_window:])
        if short_ma > long_ma:
            return 1  # Buy signal
        elif short_ma < long_ma:
            return -1  # Sell signal
        return 0  # Hold

    @staticmethod
    def rsi(data, period=14):
        if len(data) < period:
            return 50  # Neutral if not enough data
        delta = np.diff(data)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        if rsi > 70:
            return -1  # Overbought, sell signal
        elif rsi < 30:
            return 1  # Oversold, buy signal
        return 0  # Hold

    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        if len(data) < window:
            return 0
        rolling_mean = np.mean(data[-window:])
        rolling_std = np.std(data[-window:])
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        if data[-1] > upper_band:
            return -1  # Sell signal
        elif data[-1] < lower_band:
            return 1  # Buy signal
        return 0  # Hold

    @staticmethod
    def macd(data, short_window=12, long_window=26, signal_window=9):
        if len(data) < long_window + signal_window:
            return 0
        short_ema = np.array([sum(data[i:(i + short_window)]) / short_window for i in range(len(data) - short_window + 1)])
        long_ema = np.array([sum(data[i:(i + long_window)]) / long_window for i in range(len(data) - long_window + 1)])
        macd_line = short_ema[-signal_window:] - long_ema[-signal_window:]
        signal_line = sum(macd_line) / signal_window
        if macd_line[-1] > signal_line:
            return 1  # Buy signal
        elif macd_line[-1] < signal_line:
            return -1  # Sell signal
        return 0  # Hold

    @staticmethod
    def fibonacci_retracement(high, low, close):
        if len(high) < 2 or len(low) < 2 or len(close) < 2:
            return 0
        max_price = max(high)
        min_price = min(low)
        diff = max_price - min_price
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        fib_levels = [max_price - level * diff for level in levels]
        current_price = close[-1]
        for i in range(len(fib_levels) - 1):
            if fib_levels[i+1] <= current_price <= fib_levels[i]:
                if i < 3:
                    return 1  # Buy signal (price near lower Fib levels)
                elif i > 3:
                    return -1  # Sell signal (price near higher Fib levels)
        return 0  # Hold

    @staticmethod
    def ichimoku_cloud(high, low, close, conversion_period=9, base_period=26, leading_span_b_period=52, lagging_span_period=26):
        if len(high) < leading_span_b_period or len(low) < leading_span_b_period or len(close) < leading_span_b_period:
            return 0
        
        conversion_line = (max(high[-conversion_period:]) + min(low[-conversion_period:])) / 2
        base_line = (max(high[-base_period:]) + min(low[-base_period:])) / 2
        leading_span_a = (conversion_line + base_line) / 2
        leading_span_b = (max(high[-leading_span_b_period:]) + min(low[-leading_span_b_period:])) / 2
        
        current_price = close[-1]
        if current_price > leading_span_a and current_price > leading_span_b:
            return 1  # Buy signal
        elif current_price < leading_span_a and current_price < leading_span_b:
            return -1  # Sell signal
        return 0  # Hold

    @staticmethod
    def stochastic_oscillator(high, low, close, k_period=14, d_period=3):
        if len(high) < k_period or len(low) < k_period or len(close) < k_period:
            return 0
        
        lowest_low = min(low[-k_period:])
        highest_high = max(high[-k_period:])
        
        if highest_high - lowest_low == 0:
            return 0
        
        k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        d = sum([100 * (close[-i] - min(low[-i-k_period:-i])) / 
                 (max(high[-i-k_period:-i]) - min(low[-i-k_period:-i])) 
                 for i in range(d_period)]) / d_period
        
        if k > 80 and d > 80:
            return -1  # Overbought, sell signal
        elif k < 20 and d < 20:
            return 1  # Oversold, buy signal
        return 0  # Hold

# Update PyTorch model for pattern recognition
class PatternRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_patterns):
        super(PatternRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_patterns)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# Initialize PyTorch model
input_size = 1
hidden_size = 64
num_patterns = 7  # Updated number of trading patterns
model = PatternRecognitionModel(input_size, hidden_size, num_patterns)

# (TensorFlow model and other functions remain the same)

def on_message(ws, message):
    global current_position, last_trade_price, stop_loss_price, price_history, high_history, low_history

    data = json.loads(message)
    if "data" in data and len(data["data"]) > 0:
        candle = data["data"][0]
        close_price = candle["close"]
        high_price = candle["high"]
        low_price = candle["low"]
        
        price_history.append(close_price)
        high_history.append(high_price)
        low_history.append(low_price)
        if len(price_history) > 100:  # Keep only last 100 prices
            price_history = price_history[-100:]
            high_history = high_history[-100:]
            low_history = low_history[-100:]

        print(f"Received new candle. Close price: {close_price}")

        # Use PyTorch model to recognize pattern
        input_tensor = torch.FloatTensor(price_history).unsqueeze(0).unsqueeze(2)
        pattern_prediction = model(input_tensor)
        _, predicted_pattern = torch.max(pattern_prediction, 1)
        predicted_pattern = predicted_pattern.item()

        # Use TensorFlow model to predict next price
        next_price_prediction = predict_next_price(price_history, time_series_model, MinMaxScaler())

        # Determine trading signal based on predicted pattern
        if predicted_pattern == 0:
            signal = TradingPatterns.moving_average_crossover(price_history)
        elif predicted_pattern == 1:
            signal = TradingPatterns.rsi(price_history)
        elif predicted_pattern == 2:
            signal = TradingPatterns.bollinger_bands(price_history)
        elif predicted_pattern == 3:
            signal = TradingPatterns.macd(price_history)
        elif predicted_pattern == 4:
            signal = TradingPatterns.fibonacci_retracement(high_history, low_history, price_history)
        elif predicted_pattern == 5:
            signal = TradingPatterns.ichimoku_cloud(high_history, low_history, price_history)
        else:
            signal = TradingPatterns.stochastic_oscillator(high_history, low_history, price_history)

        print(f"Predicted pattern: {predicted_pattern}, Signal: {signal}, Next price prediction: {next_price_prediction}")

        # Trading logic
        if current_position == 0 and signal > 0 and close_price < next_price_prediction:
            print("Buy signal detected")
            try:
                order = api.submit_order(
                    symbol=SYMBOL,
                    qty=QUANTITY,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Buy order placed: {order}")
                current_position = QUANTITY
                last_trade_price = close_price
                stop_loss_price = close_price * (1 - STOP_LOSS_PERCENTAGE)
            except Exception as e:
                print(f"Error placing buy order: {e}")
        elif current_position > 0:
            if close_price < stop_loss_price:
                print("Stop loss triggered")
                try:
                    order = api.submit_order(
                        symbol=SYMBOL,
                        qty=QUANTITY,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    print(f"Sell order placed (stop loss): {order}")
                    current_position = 0
                    last_trade_price = close_price
                except Exception as e:
                    print(f"Error placing sell order: {e}")
            elif signal < 0 or (close_price > next_price_prediction and close_price > last_trade_price):
                print("Sell signal detected")
                try:
                    order = api.submit_order(
                        symbol=SYMBOL,
                        qty=QUANTITY,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    print(f"Sell order placed: {order}")
                    current_position = 0
                    last_trade_price = close_price
                except Exception as e:
                    print(f"Error placing sell order: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection opened")
    
    # Subscribe to real-time data
    subscribe_message = {
        "method": "chart.subscribe",
        "params": [f"NASDAQ:{SYMBOL}"],
        "id": "chart_subscribe"
    }
    ws.send(json.dumps(subscribe_message))

def run_websocket():
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(TRADINGVIEW_SOCKET,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()

if __name__ == "__main__":
    # Start the WebSocket connection in a separate thread
    websocket_thread = threading.Thread(target=run_websocket)
    websocket_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("TradingBot stopped by the user")
