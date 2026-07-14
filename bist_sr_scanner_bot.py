import os
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- Sabit Parametreler ---
LOOKBACK_PERIOD = 20
ATR_LEN = 14
BOX_WIDTH = 1.5
VOL_LEN = 20
MIN_BARS = 100


def rma_atr(df, length=14):
    """
    Pine Script'in ta.rma ve ta.atr mantığına sadık kalınarak
    RMA (Running Moving Average) bazlı ATR hesabı yapar.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    rma = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return rma.values


def calculate_pivots(close, lookback=20):
    """
    Belirli bir lookback periyodundaki tepe ve dip pivot noktalarını hesaplar.
    """
    n = len(close)
    piv_highs = np.full(n, np.nan)
    piv_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        is_high = True
        is_low = True
        for j in range(1, lookback + 1):
            if close[i] <= close[i - j] or close[i] <= close[i + j]:
                is_high = False
            if close[i] >= close[i - j] or close[i] >= close[i + j]:
                is_low = False

        if is_high:
            piv_highs[i] = close[i]
        if is_low:
            piv_lows[i] = close[i]

    return piv_highs, piv_lows


def drop_incomplete_today(df):
    """
    Türkiye saatiyle borsa kapanışından (TSİ 18:00 / UTC 15:00) önce tarama yapılıyorsa
    günlük bar henüz tamamlanmadığı için son barı eler. Kapanıştan sonra ise barı korur.
    """
    if len(df):
        now_utc = datetime.utcnow()
        if now_utc.hour < 15:  # UTC 15:00 öncesi seans açık demektir
            if df.index[-1].date() == now_utc.date():
                df = df.iloc[:-1]
    return df


def check_chartprime_sr(df: pd.DataFrame):
    """
    ChartPrime Support & Resistance indikatörünün hacim onaylı
    destek/direnç mantığını simüle eden ana tarama motoru.
    """
    if df is None or len(df) < MIN_BARS:
        return False, {}

    df = drop_incomplete_today(df)
    n = len(df)
    if n < MIN_BARS:
        return False, {}

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    volume = df["Volume"].values

    # Kutu genişliği için ATR hesabı
    atr = rma_atr(df, ATR_LEN)
    withd = atr * BOX_WIDTH

    # Pivot tespiti (20 bar geride kesinleşen seviyeleri döner)
    piv_highs, piv_lows = calculate_pivots(close, LOOKBACK_PERIOD)

    # Düzeltilmiş Hacim Filtresi (Mutlak hacim üzerinden rolling eşik değeri)
    vol_threshold = (
        pd.Series(volume).rolling(VOL_LEN, min_periods=1).max().values / 2.5
    )

    supportLevel = np.nan
    supportLevel_1 = np.nan
    resistanceLevel = np.nan
    resistanceLevel_1 = np.nan
    prev_supportLevel = np.nan
    prev_resistanceLevel = np.nan
    prev_resistanceLevel_1 = np.nan
    res_is_sup = False
    last_signal = {}

    for i in range(1, n):
        pivot_idx = i - LOOKBACK_PERIOD

        # Destek Seviyesi Doğrulaması (Geriye dönük pivot mumunun hacmine bakıyoruz)
        if pivot_idx >= 0 and not np.isnan(piv_lows[i]):
            is_green = close[pivot_idx] > df["Open"].values[pivot_idx]
            if is_green and volume[pivot_idx] > vol_threshold[pivot_idx]:
                supportLevel = piv_lows[i]
                supportLevel_1 = supportLevel - withd[i]

        # Direnç Seviyesi Doğrulaması (Geriye dönük pivot mumunun hacmine bakıyoruz)
        if pivot_idx >= 0 and not np.isnan(piv_highs[i]):
            is_red = close[pivot_idx] < df["Open"].values[pivot_idx]
            if is_red and volume[pivot_idx] > vol_threshold[pivot_idx]:
                resistanceLevel = piv_highs[i]
                resistanceLevel_1 = resistanceLevel + withd[i]

        brekout_res = False
        res_holds = False
        sup_holds = False

        if (
            not np.isnan(prev_resistanceLevel_1)
            and not np.isnan(resistanceLevel_1)
        ):
            brekout_res = (low[i - 1] < prev_resistanceLevel_1) and (
                low[i] > resistanceLevel_1
            )
        if not np.isnan(prev_resistanceLevel) and not np.isnan(
            resistanceLevel
        ):
            res_holds = (high[i - 1] > prev_resistanceLevel) and (
                high[i] < resistanceLevel
            )
        if not np.isnan(prev_supportLevel) and not np.isnan(supportLevel):
            sup_holds = (low[i - 1] < prev_supportLevel) and (
                low[i] > supportLevel
            )

        prev_res_is_sup = res_is_sup
        if brekout_res:
            res_is_sup = True
        elif res_holds:
            res_is_sup = False

        # Sadece en son bar için sinyal kontrolü yapıyoruz
        if i == n - 1:
            signal_type = None
            if sup_holds:
                signal_type = "Destekten Dönüş (Bounce)"
            elif brekout_res and prev_res_is_sup:
                signal_type = "Kırılan Dirençten Dönüş (Retest)"
            elif brekout_res and not prev_res_is_sup:
                signal_type = "Taze Direnç Kırılımı (Breakout)"

            if signal_type:
                seviye = (
                    supportLevel
                    if "Destek" in signal_type
                    else resistanceLevel_1
                )
                last_signal = {
                    "tarih": str(df.index[i].date()),
                    "kapanis": round(float(close[i]), 2),
                    "seviye": (
                        round(float(seviye), 2)
                        if not np.isnan(seviye)
                        else None
                    ),
                    "sinyal_tipi": signal_type,
                }

        prev_supportLevel = supportLevel
        prev_resistanceLevel = resistanceLevel
        prev_resistanceLevel_1 = resistanceLevel_1

    if last_signal:
        return True, last_signal
    return False, {}


def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram API bağlantı hatası: {e}")
        return False


def main():
    telegram_token = os.environ.get("TELEGRAM_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not telegram_token or not telegram_chat_id:
        print("Hata: Telegram token veya chat id çevre değişkeni eksik!")
        return

    # Tarama yapılacak BIST Listesi (Dilediğin gibi genişletebilirsin)
    bist_tickers = [
        "THYAO.IS",
        "ASELS.IS",
        "EREGL.IS",
        "TUPRS.IS",
        "KCHOL.IS",
        "SAHOL.IS",
        "AKBNK.IS",
        "GARAN.IS",
        "SISE.IS",
        "BIMAS.IS",
        "YKBNK.IS",
        "ISCTR.IS",
        "SASA.IS",
        "HEKTS.IS",
        "PGSUS.IS",
        "EKGYO.IS",
        "PETKM.IS",
        "TOASO.IS",
        "FROTO.IS",
        "ARCLK.IS",
    ]

    signals_sent = 0
    for ticker in bist_tickers:
        print(f"-> {ticker} taranıyor...")
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                continue

            has_signal, signal_data = check_chartprime_sr(df)
            if has_signal:
                hisse_adi = ticker.replace(".IS", "")
                emoji = "🟢" if "Destek" in signal_data["sinyal_tipi"] else "🔴"

                mesaj = (
                    f"⚠️ <b>SİNYAL TESPİT EDİLDİ</b> {emoji}\n\n"
                    f"📈 <b>Hisse:</b> #{hisse_adi}\n"
                    f"📅 <b>Tarih:</b> {signal_data['tarih']}\n"
                    f"⚡ <b>Sinyal Tipi:</b> {signal_data['sinyal_tipi']}\n"
                    f"💵 <b>Kapanış Fiyatı:</b> {signal_data['kapanis']} TL\n"
                    f"🎯 <b>Referans Seviye:</b> {signal_data['seviye']} TL"
                )

                send_telegram_message(telegram_token, telegram_chat_id, mesaj)
                signals_sent += 1

        except Exception as e:
            print(f"Hata ({ticker}): {e}")

    print(f"Tarama başarıyla bitti. Gönderilen sinyal: {signals_sent}")


if __name__ == "__main__":
    main()
