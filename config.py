"""config.py — настройки псевдо-опционного скальпера Bybit (USDT-Perpetual)."""
import os
from dotenv import load_dotenv

load_dotenv()

BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_SECRET = os.getenv('BYBIT_SECRET', '')
TESTNET = os.getenv('TESTNET', 'true').lower() in ('1', 'true', 'yes')

DEFAULT_SYMBOLS = (
    'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,DOGE/USDT:USDT,'
    'SUI/USDT:USDT,AVAX/USDT:USDT,LINK/USDT:USDT,ADA/USDT:USDT,BNB/USDT:USDT,'
    'NEAR/USDT:USDT,APT/USDT:USDT,ARB/USDT:USDT,OP/USDT:USDT,SEI/USDT:USDT'
)
SYMBOLS = [s.strip() for s in os.getenv('SYMBOLS', DEFAULT_SYMBOLS).split(',') if s.strip()]

TF_ENTRY = '1m'
TF_TREND = '5m'
CANDLE_LIMIT = 100

ATR_N = 14
ATR_MIN_PCT = float(os.getenv('ATR_MIN_PCT', '0.12'))
BB_N = 20
BB_K = 2.0
RSI_N = 14
RSI_LONG = (55.0, 78.0)
RSI_SHORT = (22.0, 45.0)
VOL_SMA_N = 20
VOL_MULT = float(os.getenv('VOL_MULT', '1.8'))
BREAK_N = 5
OB_DEPTH = 5
OB_IMB = float(os.getenv('OB_IMB', '1.2'))

SL_ATR_MULT = float(os.getenv('SL_ATR_MULT', '1.5'))
SL_MIN_PCT = 0.0015
SL_MAX_PCT = 0.0060
RR = float(os.getenv('RR', '2.5'))

RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', '1.0')) / 100.0
LEVERAGE = int(os.getenv('LEVERAGE', '10'))
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '4'))
DAY_LOSS_PCT = float(os.getenv('DAY_LOSS_PCT', '4.0')) / 100.0

ENTRY_TIMEOUT_S = float(os.getenv('ENTRY_TIMEOUT_S', '8'))
BE_TRIGGER = float(os.getenv('BE_TRIGGER', '1.2'))
BE_OFFSET_PCT = float(os.getenv('BE_OFFSET_PCT', '0.0006'))
TIME_STOP_MIN = float(os.getenv('TIME_STOP_MIN', '20'))
POLL_SEC = float(os.getenv('POLL_SEC', '2'))

TG_TOKEN = os.getenv('TG_TOKEN', '')
TG_CHAT_ID = os.getenv('TG_CHAT_ID', '')

STATE_FILE = os.getenv('STATE_FILE', 'state.json')
LOG_FILE = os.getenv('LOG_FILE', 'bot.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
