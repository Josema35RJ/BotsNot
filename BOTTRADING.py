import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import pandas as pd
import ccxt  # Para criptomonedas
import alpaca_trade_api as tradeapi  # Para acciones y materias primas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import backtrader as bt
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import time

# ================= Configuración de APIs =================

# Configuración de Binance Testnet
binance = ccxt.binance({
    "apiKey": "YOUR_BINANCE_API_KEY",
    "secret": "YOUR_BINANCE_SECRET",
    "test": True,  # Testnet
    "enableRateLimit": True,
})

# Configuración de Alpaca Paper Trading
alpaca = tradeapi.REST(
    "YOUR_ALPACA_API_KEY",
    "YOUR_ALPACA_SECRET",
    "https://paper-api.alpaca.markets",
    api_version='v2'
)

# ================= Funciones de Utilidad =================

def get_stock_news_sentiment(ticker):
    """
    Obtiene las principales noticias de Google para el ticker indicado y calcula
    el sentimiento promedio usando TextBlob.
    """
    url = f"https://www.google.com/search?q={ticker}+stock+news"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [h.text for h in soup.find_all("h3")]
    
    sentiment_scores = []
    for headline in headlines[:5]:  # Analiza las 5 primeras noticias
        analysis = TextBlob(headline)
        sentiment_scores.append(analysis.sentiment.polarity)
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    return headlines[:5], avg_sentiment

def get_historical_data(ticker, period="1y", interval="1d"):
    """
    Descarga datos históricos de precios (Close) usando yfinance.
    """
    data = yf.download(ticker, period=period, interval=interval)
    return data["Close"].values.reshape(-1, 1)

def trade_crypto(symbol, order_type="buy", amount=0.01, log_widget=None):
    """
    Ejecuta una orden de compra/venta en Binance Testnet.
    """
    try:
        if order_type == "buy":
            binance.create_market_buy_order(symbol, amount)
        else:
            binance.create_market_sell_order(symbol, amount)
        log_message = f"Orden {order_type} ejecutada en {symbol}\n"
        log_widget.insert(tk.END, log_message)
    except Exception as e:
        log_message = f"Error en trading de {symbol}: {str(e)}\n"
        log_widget.insert(tk.END, log_message)
    log_widget.yview(tk.END)

def trade_stock(symbol, order_type="buy", qty=1, log_widget=None):
    """
    Ejecuta una orden de compra/venta en Alpaca.
    """
    try:
        alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=order_type,
            type='market',
            time_in_force='gtc'
        )
        log_message = f"Orden {order_type} ejecutada en {symbol}\n"
        log_widget.insert(tk.END, log_message)
    except Exception as e:
        log_message = f"Error en trading de {symbol}: {str(e)}\n"
        log_widget.insert(tk.END, log_message)
    log_widget.yview(tk.END)

def optimize_news_impact_model(news_data, price_changes, log_widget=None):
    """
    Optimiza un modelo Random Forest para predecir el impacto de las noticias.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(news_data)
    y = np.array(price_changes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    log_widget.insert(tk.END, f"Modelo optimizado: MSE={mse:.4f}, R2={r2:.4f}\n")
    log_widget.yview(tk.END)
    return best_model, vectorizer

# ================= Backtrader: Estrategia de Trading =================

class TradingStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=14)

    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.data.close[0] < self.sma[0]:
            self.sell()

def run_backtest(log_widget):
    """
    Ejecuta un backtest simple usando datos históricos de AAPL y la estrategia definida.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TradingStrategy)
    
    data = bt.feeds.PandasData(dataname=yf.download("AAPL", period="1y", interval="1d"))
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    
    initial_value = cerebro.broker.getvalue()
    log_widget.insert(tk.END, f"Valor inicial del portafolio: ${initial_value:.2f}\n")
    
    cerebro.run()
    final_value = cerebro.broker.getvalue()
    log_widget.insert(tk.END, f"Valor final del portafolio: ${final_value:.2f}\n")
    log_widget.insert(tk.END, "Backtest completado.\n")
    log_widget.yview(tk.END)

# ================= Ejemplo de Modelo LSTM =================

def train_lstm_model(log_widget):
    """
    Construye y entrena un modelo LSTM simple usando datos históricos de AAPL.
    """
    # Obtener datos históricos
    data = get_historical_data("AAPL", period="2y", interval="1d")
    # Normalizar los datos
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Preparar datos para LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 14
    X, y = create_dataset(data_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Dividir en train y test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Definir el modelo LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    log_widget.insert(tk.END, "Entrenando modelo LSTM...\n")
    log_widget.yview(tk.END)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    
    # Evaluar el modelo
    loss = model.evaluate(X_test, y_test, verbose=0)
    log_widget.insert(tk.END, f"Modelo LSTM entrenado. Pérdida en test: {loss:.4f}\n")
    log_widget.yview(tk.END)
    
    return model, scaler

# ================= Interfaz Gráfica (Tkinter) =================

def create_gui():
    root = tk.Tk()
    root.title("Trading Interface Completa")

    # Área de logs
    log_area = scrolledtext.ScrolledText(root, width=100, height=25, wrap=tk.WORD, font=("Arial", 10), bg="#f4f4f4", fg="black")
    log_area.pack(padx=10, pady=10)

    # Función para ejecutar las operaciones de trading y análisis
    def execute_trades():
        log_area.insert(tk.END, "Iniciando operaciones...\n")
        log_area.yview(tk.END)
        
        # Datos históricos y noticias para AAPL
        historical_prices = get_historical_data("AAPL")
        log_area.insert(tk.END, "Datos históricos obtenidos para AAPL.\n")
        
        news, sentiment = get_stock_news_sentiment("AAPL")
        log_area.insert(tk.END, f"Noticias de AAPL: {news}\n")
        log_area.insert(tk.END, f"Sentimiento promedio: {sentiment:.4f}\n")
        
        # Ejecución de operaciones en Binance y Alpaca
        trade_crypto("BTC/USDT", "buy", 0.01, log_area)
        trade_stock("AAPL", "buy", 1, log_area)
        trade_stock("OIL", "buy", 1, log_area)
        
        log_area.insert(tk.END, "Operaciones de trading ejecutadas.\n")
        log_area.yview(tk.END)
    
    def optimize_model():
        # Ejemplo de datos: usa las noticias obtenidas anteriormente y cambios de precio ficticios
        news_sample, _ = get_stock_news_sentiment("AAPL")
        price_changes_sample = [0.5, -0.3, 0.8, -0.2, 0.4]
        optimize_news_impact_model(news_sample, price_changes_sample, log_area)
    
    def execute_backtest():
        run_backtest(log_area)
    
    def train_lstm():
        train_lstm_model(log_area)
    
    # Botones para cada funcionalidad
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    
    trade_btn = tk.Button(btn_frame, text="Ejecutar Trading", command=execute_trades, font=("Arial", 12), bg="#4CAF50", fg="white", width=20)
    trade_btn.grid(row=0, column=0, padx=5, pady=5)
    
    optimize_btn = tk.Button(btn_frame, text="Optimizar Modelo Noticias", command=optimize_model, font=("Arial", 12), bg="#2196F3", fg="white", width=25)
    optimize_btn.grid(row=0, column=1, padx=5, pady=5)
    
    backtest_btn = tk.Button(btn_frame, text="Ejecutar Backtest", command=execute_backtest, font=("Arial", 12), bg="#FF9800", fg="white", width=20)
    backtest_btn.grid(row=1, column=0, padx=5, pady=5)
    
    lstm_btn = tk.Button(btn_frame, text="Entrenar Modelo LSTM", command=train_lstm, font=("Arial", 12), bg="#9C27B0", fg="white", width=20)
    lstm_btn.grid(row=1, column=1, padx=5, pady=5)
    
    root.mainloop()

# ================= Ejecución de la GUI =================

create_gui()
