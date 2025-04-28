# Trading Bot

A sophisticated algorithmic trading bot that combines technical analysis with sentiment analysis to make trading decisions. The bot uses EMA (Exponential Moving Average) crossover strategy combined with FinBERT sentiment analysis to trade options on Alpaca's platform.

## Features

- **Technical Analysis**: Implements EMA crossover strategy (9 and 21 periods)
- **Sentiment Analysis**: Utilizes FinBERT for real-time sentiment analysis of financial news
- **Options Trading**: Automatically trades ATM (At-The-Money) calls and puts
- **Risk Management**: Implements stop-loss protection
- **Paper Trading**: Supports both paper and live trading modes
- **Comprehensive Logging**: Detailed logging of all trading activities and decisions

## Prerequisites

- Python 3.7+
- Alpaca Trading Account (Paper or Live)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading_bot.git
cd trading_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the root directory with your Alpaca API credentials:
```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
```

2. Configure trading parameters in `main.py`:
   - TICKER: The stock symbol to trade
   - FAST_WINDOW: Fast EMA period (default: 9)
   - SLOW_WINDOW: Slow EMA period (default: 21)
   - PAPER: Set to True for paper trading, False for live trading
   - STOP_LOSS_PCT: Stop loss percentage (default: 15%)

> **Important**: Never commit your `.env` file to version control. Add it to your `.gitignore` file.

## Usage

Run the trading bot:
```bash
python main.py
```

The bot will:
1. Monitor the specified stock's price movements
2. Analyze news sentiment using FinBERT
3. Execute trades based on EMA crossover and sentiment signals
4. Log all activities to the `logs` directory

## Trading Strategy

The bot implements the following strategy:
- Buys 1 ATM CALL when:
  - Fast EMA > Slow EMA AND
  - Sentiment score ≥ +0.05
- Buys 1 ATM PUT when:
  - Fast EMA < Slow EMA AND
  - Sentiment score ≤ -0.05
- Flattens to cash when signals flip or become neutral

## Dependencies

- alpaca-py: Alpaca trading API client
- numpy: Numerical computing
- pandas: Data manipulation
- yfinance: Yahoo Finance data
- feedparser: RSS feed parsing
- torch: PyTorch for FinBERT
- transformers: Hugging Face Transformers for FinBERT
- python-dotenv: Environment variable management

## Logging

All trading activities and decisions are logged to:
- Console output
- `logs/trading_bot.log` file

## Disclaimer

This trading bot is for educational purposes only. Trading options involves significant risk and is not suitable for all investors. Past performance is not indicative of future results. Always test thoroughly in paper trading mode before using real money.
