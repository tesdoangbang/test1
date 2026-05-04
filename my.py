# /opt/xauusd_signal_bot/xauusd_signal_bot.py
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


@dataclass(frozen=True)
class Config:
    ticker: str = "GC=F"
    interval: str = "15m"
    days_back: int = 10

    ema_short: int = 9
    ema_long: int = 21
    ema_trend: int = 50
    rsi_period: int = 9

    use_atr_filter: bool = True
    atr_multiplier: float = 1.30

    min_score: int = 5
    cooldown_bars: int = 1

    sl_atr_mult: float = 0.90
    tp_atr_mult: float = 5.20

    poll_seconds: int = 30
    timezone_name: str = "Asia/Jakarta"

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    state_file: str = "state.json"
    log_file: str = "bot.log"


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, timeout: int = 15) -> None:
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_message(self, text: str) -> None:
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok", False):
            raise RuntimeError(f"Telegram API error: {data}")


class StateStore:
    def __init__(self, state_file: str) -> None:
        self.path = Path(state_file)
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "last_processed_candle": None,
                "last_signal_candle": None,
                "last_signal_type": None,
                "last_signal_score": None,
                "last_signal_time": None,
                "last_heartbeat_hour": None,
            }
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "last_processed_candle": None,
                "last_signal_candle": None,
                "last_signal_type": None,
                "last_signal_score": None,
                "last_signal_time": None,
                "last_heartbeat_hour": None,
            }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.save()


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_config_from_env() -> Config:
    def env_str(name: str, default: str) -> str:
        return os.getenv(name, default).strip()

    def env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        return int(value) if value not in (None, "") else default

    def env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        return float(value) if value not in (None, "") else default

    def env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    return Config(
        ticker=env_str("TICKER", "GC=F"),
        interval=env_str("INTERVAL", "15m"),
        days_back=env_int("DAYS_BACK", 10),
        ema_short=env_int("EMA_SHORT", 9),
        ema_long=env_int("EMA_LONG", 21),
        ema_trend=env_int("EMA_TREND", 50),
        rsi_period=env_int("RSI_PERIOD", 9),
        use_atr_filter=env_bool("USE_ATR_FILTER", True),
        atr_multiplier=env_float("ATR_MULTIPLIER", 1.30),
        min_score=env_int("MIN_SCORE", 5),
        cooldown_bars=env_int("COOLDOWN_BARS", 1),
        sl_atr_mult=env_float("SL_ATR_MULT", 0.90),
        tp_atr_mult=env_float("TP_ATR_MULT", 5.20),
        poll_seconds=env_int("POLL_SECONDS", 30),
        timezone_name=env_str("TIMEZONE_NAME", "Asia/Jakarta"),
        telegram_bot_token=env_str("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=env_str("TELEGRAM_CHAT_ID", ""),
        state_file=env_str("STATE_FILE", "state.json"),
        log_file=env_str("LOG_FILE", "bot.log"),
    )


def validate_config(config: Config) -> None:
    if not config.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN belum diisi")
    if not config.telegram_chat_id:
        raise ValueError("TELEGRAM_CHAT_ID belum diisi")
    if config.ema_short >= config.ema_long:
        raise ValueError("EMA_SHORT harus lebih kecil dari EMA_LONG")
    if config.min_score < 1:
        raise ValueError("MIN_SCORE minimal 1")
    if config.poll_seconds < 10:
        raise ValueError("POLL_SECONDS terlalu kecil")
    if config.interval not in {"15m", "30m", "1h", "1d"}:
        raise ValueError("INTERVAL harus salah satu dari: 15m, 30m, 1h, 1d")


def fetch_ohlcv(config: Config) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - pd.Timedelta(days=config.days_back)

    df = yf.download(
        config.ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=config.interval,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return df


def calculate_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()

    out["EMA_Short"] = out["Close"].ewm(span=config.ema_short, adjust=False).mean()
    out["EMA_Long"] = out["Close"].ewm(span=config.ema_long, adjust=False).mean()
    out["EMA_Trend"] = out["Close"].ewm(span=config.ema_trend, adjust=False).mean()

    out["EMA_Long_Slope"] = out["EMA_Long"].diff()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD_Line"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD_Line"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD_Line"] - out["MACD_Signal"]
    out["MACD_Hist_Change"] = out["MACD_Hist"].diff()

    out["BB_Mid"] = out["Close"].rolling(window=20).mean()
    out["BB_Std"] = out["Close"].rolling(window=20).std()
    out["BB_Up"] = out["BB_Mid"] + 2 * out["BB_Std"]
    out["BB_Low"] = out["BB_Mid"] - 2 * out["BB_Std"]

    hl = out["High"] - out["Low"]
    hc = (out["High"] - out["Close"].shift(1)).abs()
    lc = (out["Low"] - out["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["ATR_Median_20"] = out["ATR"].rolling(20).median()

    out["Body"] = (out["Close"] - out["Open"]).abs()
    out["Range"] = (out["High"] - out["Low"]).replace(0, np.nan)
    out["Body_Ratio"] = out["Body"] / out["Range"]

    out["Rolling_High_5"] = out["High"].shift(1).rolling(5).max()
    out["Rolling_Low_5"] = out["Low"].shift(1).rolling(5).min()

    return out


def apply_cooldown(signal_series: pd.Series, cooldown_bars: int) -> pd.Series:
    filtered = signal_series.copy()
    last_signal_index = -1000000

    for i in range(len(filtered)):
        if filtered.iat[i] == 0:
            continue
        if i - last_signal_index <= cooldown_bars:
            filtered.iat[i] = 0
            continue
        last_signal_index = i

    return filtered


def generate_signals(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = df.copy()

    cross_up = (out["EMA_Short"] > out["EMA_Long"]) & (
        out["EMA_Short"].shift(1) <= out["EMA_Long"].shift(1)
    )
    cross_down = (out["EMA_Short"] < out["EMA_Long"]) & (
        out["EMA_Short"].shift(1) >= out["EMA_Long"].shift(1)
    )

    trend_long = (
        (out["Close"] > out["EMA_Trend"])
        & (out["EMA_Short"] > out["EMA_Long"])
        & (out["EMA_Long_Slope"] > 0)
    )
    trend_short = (
        (out["Close"] < out["EMA_Trend"])
        & (out["EMA_Short"] < out["EMA_Long"])
        & (out["EMA_Long_Slope"] < 0)
    )

    atr_ok = (
        out["ATR"] >= (out["ATR_Median_20"] * config.atr_multiplier)
        if config.use_atr_filter
        else pd.Series(True, index=out.index)
    )
    body_ok = out["Body_Ratio"] >= 0.45
    bullish_break = out["Close"] > out["Rolling_High_5"]
    bearish_break = out["Close"] < out["Rolling_Low_5"]

    long_score = (
        cross_up.astype(int)
        + trend_long.astype(int)
        + (out["MACD_Hist"] > 0).astype(int)
        + (out["MACD_Hist_Change"] > 0).astype(int)
        + ((out["RSI"] >= 52) & (out["RSI"] <= 68)).astype(int)
        + (out["Close"] > out["EMA_Short"]).astype(int)
        + bullish_break.astype(int)
        + atr_ok.astype(int)
        + body_ok.astype(int)
    )

    short_score = (
        cross_down.astype(int)
        + trend_short.astype(int)
        + (out["MACD_Hist"] < 0).astype(int)
        + (out["MACD_Hist_Change"] < 0).astype(int)
        + ((out["RSI"] >= 32) & (out["RSI"] <= 48)).astype(int)
        + (out["Close"] < out["EMA_Short"]).astype(int)
        + bearish_break.astype(int)
        + atr_ok.astype(int)
        + body_ok.astype(int)
    )

    invalid_long = (
        (out["Close"] >= out["BB_Up"])
        | (out["RSI"] > 70)
        | (out["MACD_Line"] < out["MACD_Signal"])
    )
    invalid_short = (
        (out["Close"] <= out["BB_Low"])
        | (out["RSI"] < 30)
        | (out["MACD_Line"] > out["MACD_Signal"])
    )

    raw_signal = pd.Series(0, index=out.index, dtype=int)
    raw_signal.loc[(long_score >= config.min_score) & (~invalid_long)] = 1
    raw_signal.loc[(short_score >= config.min_score) & (~invalid_short)] = -1
    raw_signal = apply_cooldown(raw_signal, config.cooldown_bars)

    out["Signal"] = raw_signal
    out["Signal_Type"] = np.where(
        out["Signal"] == 1,
        "BUY",
        np.where(out["Signal"] == -1, "SELL", "HOLD"),
    )
    out["Signal_Score"] = 0
    out.loc[out["Signal"] == 1, "Signal_Score"] = long_score[out["Signal"] == 1]
    out.loc[out["Signal"] == -1, "Signal_Score"] = short_score[out["Signal"] == -1]

    out["SL_Price"] = np.nan
    out["TP_Price"] = np.nan
    out.loc[out["Signal"] == 1, "SL_Price"] = out["Close"] - (out["ATR"] * config.sl_atr_mult)
    out.loc[out["Signal"] == 1, "TP_Price"] = out["Close"] + (out["ATR"] * config.tp_atr_mult)
    out.loc[out["Signal"] == -1, "SL_Price"] = out["Close"] + (out["ATR"] * config.sl_atr_mult)
    out.loc[out["Signal"] == -1, "TP_Price"] = out["Close"] - (out["ATR"] * config.tp_atr_mult)

    return out


def candle_to_iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").isoformat()


def format_price(value: float) -> str:
    return f"{value:.2f}"


def build_signal_message(row: pd.Series, config: Config) -> str:
    candle_time = row["Date"].tz_convert(config.timezone_name).strftime("%Y-%m-%d %H:%M:%S")
    atr_text = format_price(float(row["ATR"])) if pd.notna(row["ATR"]) else "-"
    rsi_text = f"{float(row['RSI']):.1f}" if pd.notna(row["RSI"]) else "-"
    score_text = str(int(row["Signal_Score"]))

    return (
        f"<b>🚨 XAUUSD SIGNAL {row['Signal_Type']}</b>\n"
        f"Symbol: <b>{config.ticker}</b>\n"
        f"TF: <b>{config.interval}</b>\n"
        f"Waktu: <b>{candle_time} {config.timezone_name}</b>\n\n"
        f"Entry: <b>{format_price(float(row['Close']))}</b>\n"
        f"SL: <b>{format_price(float(row['SL_Price']))}</b>\n"
        f"TP: <b>{format_price(float(row['TP_Price']))}</b>\n\n"
        f"Score: <b>{score_text}</b>\n"
        f"RSI: <b>{rsi_text}</b>\n"
        f"ATR: <b>{atr_text}</b>\n"
        f"EMA: <b>{config.ema_short}/{config.ema_long}</b>\n"
        f"ATR Filter: <b>{'ON' if config.use_atr_filter else 'OFF'}</b> ({config.atr_multiplier:.2f})\n"
        f"Cooldown: <b>{config.cooldown_bars}</b>\n"
        f"Risk:Reward ATR = <b>{config.sl_atr_mult:.2f}:{config.tp_atr_mult:.2f}</b>"
    )


def build_startup_message(config: Config) -> str:
    cfg = asdict(config)
    safe_cfg = {k: v for k, v in cfg.items() if "token" not in k.lower()}
    formatted = "\n".join(f"- {k}: <b>{v}</b>" for k, v in safe_cfg.items())
    return f"<b>✅ XAUUSD signal bot started</b>\n{formatted}"


def build_heartbeat_message(config: Config, row: pd.Series) -> str:
    candle_time = row["Date"].tz_convert(config.timezone_name).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"<b>💓 Bot heartbeat</b>\n"
        f"Symbol: <b>{config.ticker}</b>\n"
        f"TF: <b>{config.interval}</b>\n"
        f"Last closed candle: <b>{candle_time} {config.timezone_name}</b>\n"
        f"Close: <b>{format_price(float(row['Close']))}</b>\n"
        f"RSI: <b>{float(row['RSI']):.1f}</b>\n"
        f"ATR: <b>{format_price(float(row['ATR']))}</b>"
    )


def should_send_heartbeat(state: StateStore) -> bool:
    current_hour = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    if state.get("last_heartbeat_hour") == current_hour:
        return False
    state.set("last_heartbeat_hour", current_hour)
    return True


def process_once(config: Config, notifier: TelegramNotifier, state: StateStore) -> None:
    raw = fetch_ohlcv(config)
    if raw.empty or len(raw) < 80:
        logging.warning("Data kosong atau belum cukup untuk indikator")
        return

    df = calculate_indicators(raw, config)
    df = generate_signals(df, config)

    closed_df = df.iloc[:-1].copy()
    if closed_df.empty:
        logging.warning("Belum ada candle closed")
        return

    last_closed = closed_df.iloc[-1]
    candle_iso = candle_to_iso(last_closed["Date"])

    if should_send_heartbeat(state):
        try:
            notifier.send_message(build_heartbeat_message(config, last_closed))
        except Exception as exc:
            logging.exception("Gagal kirim heartbeat: %s", exc)

    if state.get("last_processed_candle") == candle_iso:
        logging.info("Candle %s sudah diproses", candle_iso)
        return

    state.set("last_processed_candle", candle_iso)

    if int(last_closed["Signal"]) == 0:
        logging.info("Tidak ada sinyal pada candle %s", candle_iso)
        return

    signal_type = str(last_closed["Signal_Type"])
    signal_score = int(last_closed["Signal_Score"])

    if (
        state.get("last_signal_candle") == candle_iso
        and state.get("last_signal_type") == signal_type
        and state.get("last_signal_score") == signal_score
    ):
        logging.info("Sinyal candle %s sudah pernah dikirim", candle_iso)
        return

    message = build_signal_message(last_closed, config)
    notifier.send_message(message)

    state.set("last_signal_candle", candle_iso)
    state.set("last_signal_type", signal_type)
    state.set("last_signal_score", signal_score)
    state.set("last_signal_time", datetime.now(timezone.utc).isoformat())

    logging.info(
        "Signal sent | candle=%s type=%s score=%s",
        candle_iso,
        signal_type,
        signal_score,
    )


def main() -> None:
    config = load_config_from_env()
    setup_logging(config.log_file)
    validate_config(config)

    notifier = TelegramNotifier(
        token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )
    state = StateStore(config.state_file)

    logging.info("Bot started with config: %s", {k: v for k, v in asdict(config).items() if "token" not in k.lower()})

    try:
        notifier.send_message(build_startup_message(config))
    except Exception as exc:
        logging.exception("Gagal kirim startup message: %s", exc)

    while True:
        try:
            process_once(config, notifier, state)
        except Exception as exc:
            logging.exception("Unhandled error: %s", exc)
            try:
                notifier.send_message(f"<b>⚠️ Bot error</b>\n<code>{str(exc)[:3500]}</code>")
            except Exception:
                logging.exception("Gagal kirim error message ke Telegram")
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
