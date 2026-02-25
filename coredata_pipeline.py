"""
Real-time data pipeline with edge case handling and multiple data sources
"""
import asyncio
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import yfinance as yf
from collections import deque
import talib

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    spreads: Optional[Dict[str, float]] = None
    
class DataPipeline:
    """Robust data pipeline with failover mechanisms"""
    
    def __init__(self, symbols: List[str], timeframe: str = "15m"):
        self.symbols = symbols
        self.timeframe = timeframe
        self.exchange = self._initialize_exchange()
        self.data_buffer = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.last_update = {symbol: None for symbol in symbols}
        self._validate_symbols()
        
    def _initialize_exchange(self):
        """Initialize exchange connection with error handling"""
        try:
            exchange = ccxt.binance({
                'apiKey': os.getenv('EXCHANGE_API_KEY', ''),
                'secret': os.getenv('EXCHANGE_SECRET', ''),
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Test connectivity
            exchange.fetch_status()
            logger.info