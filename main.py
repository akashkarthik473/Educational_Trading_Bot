"""
trading_bot_options.py
──────────────────────
1-minute EMA-crossover + FinBERT sentiment bot that
 • buys 1 ATM CALL when fast-EMA > slow-EMA AND sentiment ≥ +0.05
 • buys 1 ATM PUT  when fast-EMA < slow-EMA AND sentiment ≤ –0.05
Flatten-to-cash whenever the signal flips or goes neutral.
Runs on Alpaca PAPER by default.
"""

from __future__ import annotations
import asyncio
import datetime as dt
from datetime import date, timedelta
from typing import Optional, List
import logging
import sys
import os
import time
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import urllib.parse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ───────── Alpaca SDK ─────────
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()

# ╭───────────────────────────── CONFIG ────────────────────────────╮
TICKER = "AAPL"
FAST_WINDOW = 9
SLOW_WINDOW = 21
PAPER = True  # False → live
STOP_LOSS_PCT = 0.15  # 15% stop loss

# Get API credentials from environment variables
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")

# Logging configuration
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'trading_bot.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ╰────────────────────────────────────────────────────────────────╯

# ───── Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
stock_data = StockHistoricalDataClient(API_KEY, API_SECRET)
option_data = OptionHistoricalDataClient(API_KEY, API_SECRET)

# ───── FinBERT sentiment
logger.info("Loading FinBERT model...")
_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_nlp = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
logger.info("FinBERT model loaded successfully")

# ───── helpers
def latest_closes(sym: str, n: int) -> pd.Series:
    try:
        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Minute,
            limit=n
        )
        df = stock_data.get_stock_bars(req).df
        if df.empty:
            logger.warning("No data returned. The market may be closed or there is no recent data available.")
            return pd.Series(dtype=float)
        if "close" not in df.columns:
            logger.error(f"DataFrame columns: {df.columns}")
            logger.error(f"DataFrame head:\n{df.head()}")
            raise ValueError("Column 'close' not found in stock bars data")
        return df["close"]
    except Exception as e:
        logger.error(f"Error getting latest closes: {e}")
        return pd.Series(dtype=float)

def rss_headlines(sym: str, hrs: int = 6, limit: int = 25) -> list[str]:
    try:
        q = urllib.parse.quote_plus(f"{sym} when:{hrs}h site:finance.yahoo.com")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        return [e.title for e in feedparser.parse(url).entries[:limit]]
    except Exception as e:
        logger.error(f"Error fetching RSS headlines: {e}")
        return []

def finbert_score(texts: List[str]) -> float:
    if not texts:
        return 0.0
    try:
        def s(t):
            toks = _tok(t, return_tensors="pt", truncation=True, max_length=96)
            with torch.no_grad():
                p = torch.softmax(_nlp(**toks).logits, 1)[0]
            return (p[2]-p[0]).item()  # pos – neg
        return float(np.mean([s(t) for t in texts]))
    except Exception as e:
        logger.error(f"Error calculating FinBERT score: {e}")
        return 0.0

def pick_atm_contract(sym: str, typ: str) -> str:
    """
    Return OCC symbol of nearest-expiry (≥3 days) ATM call/put.
    `typ` = "call" | "put"
    """
    try:
        today = date.today()
        window_min = today + timedelta(days=3)  # Minimum 3 days to expiration
        window_max = today + timedelta(days=30)  # Maximum 30 days to expiration
        
        req = GetOptionContractsRequest(
            underlying_symbols=[sym],
            option_type=typ,
            expiration_date_gte=window_min.isoformat(),
            expiration_date_lte=window_max.isoformat(),
            tradable=True,
            limit=100,
        )
        contracts = trading_client.get_option_contracts(req).option_contracts
        if not contracts:
            raise RuntimeError("No tradable option found")
        spot = yf.Ticker(sym).fast_info["lastPrice"]
        return min(contracts,
                   key=lambda c: abs(float(c.strike_price) - spot)).symbol
    except Exception as e:
        logger.error(f"Error picking ATM contract: {e}")
        raise

# ───── trading state
_pos: Optional[str] = None  # "long", "short", None
_sent_cache_ts: Optional[dt.datetime] = None
_sent_val: float = 0.0

def check_options_buying_power() -> float:
    """Get current options buying power in dollars"""
    try:
        account = trading_client.get_account()
        return float(account.options_buying_power)
    except Exception as e:
        logger.error(f"Error checking options buying power: {e}")
        return 0.0

def is_market_open() -> bool:
    """Check if the market is currently open"""
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return False

async def check_stop_loss():
    """Check if any positions have hit stop loss"""
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            loss_pct = (entry_price - current_price) / entry_price
            
            if loss_pct >= STOP_LOSS_PCT:
                logger.warning(f"Stop loss triggered for {pos.symbol} at {loss_pct:.1%} loss")
                await rebalance(None)  # Close position
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking stop loss: {e}")
        return False

async def rebalance(target: Optional[str]):
    try:
        trading_client.close_all_positions(cancel_orders=True)
        if target is None:
            logger.info("→ Flat")
            return

        typ = "call" if target == "long" else "put"
        occ = pick_atm_contract(TICKER, typ)
        
        # Check options buying power before placing order
        options_bp = check_options_buying_power()
        logger.info(f"Current options buying power: ${options_bp:.2f}")
        
        if options_bp < 500:  # Buffer for fees and price movements
            logger.warning(f"⚠️ Insufficient options buying power (${options_bp:.2f}). Skipping trade.")
            return

        leg = OptionLegRequest(
            symbol=occ,
            side=OrderSide.BUY,
            qty=5,
            ratio_qty=1
        )

        order = MarketOrderRequest(
            symbol=occ,
            side=OrderSide.BUY,
            order_class="simple",
            time_in_force=TimeInForce.DAY,
            qty=5
        )
        try:
            resp = trading_client.submit_order(order)
            logger.info(f"→ Entered {target.upper()} via {occ}  id={resp.id}")
        except Exception as e:
            logger.error(f"[ORDER ERR] {e}")
    except Exception as e:
        logger.error(f"Error in rebalance: {e}")

async def trade_loop():
    global _pos, _sent_cache_ts, _sent_val
    logger.info("Starting trading bot...")

    while True:
        try:
            now = dt.datetime.now(dt.UTC)
            
            # Check if market is open
            if not is_market_open():
                logger.info("Market is closed. Waiting...")
                await asyncio.sleep(300)  # Check every 5 minutes
                continue

            # Check stop loss first
            if await check_stop_loss():
                await asyncio.sleep(60)
                continue

            # refresh sentiment every 15 min
            if not _sent_cache_ts or (now - _sent_cache_ts).total_seconds() > 900:
                _sent_val = finbert_score(rss_headlines(TICKER))
                _sent_cache_ts = now
                logger.info(f"Sentiment: {round(_sent_val, 3)}")

            closes = latest_closes(TICKER, SLOW_WINDOW+5)
            fast, slow = closes.ewm(span=FAST_WINDOW).mean().iloc[-1], \
                         closes.ewm(span=SLOW_WINDOW).mean().iloc[-1]
            raw_sig = np.sign(fast - slow)
            bias = 1 if _sent_val > 0.05 else -1 if _sent_val < -0.05 else 0
            sig = raw_sig * bias

            desired = "long" if sig > 0 else "short" if sig < 0 else None
            if desired != _pos:
                await rebalance(desired)
                _pos = desired

            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in trade loop: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying

# ───── entry-point
if __name__ == "__main__":
    try:
        logger.info("Initializing trading bot...")
        asyncio.run(trade_loop())
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
        trading_client.close_all_positions()
        logger.info("Stopped – positions closed.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        trading_client.close_all_positions()
        logger.info("Bot stopped due to error – positions closed.")
