import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import telebot  # pip install pyTelegramBotAPI

# --- Sabit Parametreler ---
LOOKBACK_PERIOD = 20
ATR_LEN = 14
BOX_WIDTH = 1.5
VOL_LEN = 20
MIN_BARS = 100

# Telegram Bot Kurulumu
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Hata: TELEGRAM_TOKEN çevre değişkeni tanımlanmamış!")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Varsayılan Tarama Listesi
BIST_TICKERS = [
    "THYAO.IS", "ASELS.IS", "EREGL.IS", "TUPRS.IS", "KCHOL.IS",
    "SAHOL.IS", "AKBNK.IS", "GARAN.IS", "SISE.IS", "BIMAS.IS",
    "YKBNK.IS", "ISCTR.IS", "SASA.IS", "HEKTS.IS", "PGSUS.IS",
    "EKGYO.IS", "PETKM.IS", "TOASO.IS", "FROTO.IS", "ARCLK.IS"
]

# --- Teknik Analiz Fonksiyonları ---

def rma_atr(df, length=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean().values

def calculate_pivots(close, lookback=20):
    n = len(close)
    piv_highs, piv_lows = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        is_high = is_low = True
        for j in range(1, lookback + 1):
            if close[i] <= close[i - j] or close[i] <= close[i + j]:
                is_high = False
            if close[i] >= close[i - j] or close[i] >= close[i + j]:
                is_low = False
        if is_high: piv_highs[i] = close[i]
        if is_low:  piv_lows[i] = close[i]
    return piv_highs, piv_lows

def drop_incomplete_today(df):
    """
    Eğer aktif seans saatlerinde (/hisse komutuyla canlı) sorgulanıyorsa 
    ve son bar henüz kapanmadıysa, analizin hatalı olmaması için son barı eler.
    """
    if len(df):
        now_utc = datetime.utcnow()
        if now_utc.hour < 15: # UTC 15:00 (TSİ 18:00) öncesi seans açık demektir
            if df.index[-1].date() == now_utc.date():
                df = df.iloc[:-1]
    return df

def check_chartprime_sr(df: pd.DataFrame):
    if df is None or len(df) < MIN_BARS:
        return False, {}
    
    df = drop_incomplete_today(df)
    n = len(df)
    if n < MIN_BARS:
        return False, {}

    high, low, close, volume = df["High"].values, df["Low"].values, df["Close"].values, df["Volume"].values
    atr = rma_atr(df, ATR_LEN)
    withd = atr * BOX_WIDTH
    piv_highs, piv_lows = calculate_pivots(close, LOOKBACK_PERIOD)
    vol_threshold = pd.Series(volume).rolling(VOL_LEN, min_periods=1).max().values / 2.5

    supportLevel = supportLevel_1 = resistanceLevel = resistanceLevel_1 = np.nan
    prev_supportLevel = prev_resistanceLevel = prev_resistanceLevel_1 = np.nan
    res_is_sup = False
    last_signal = {}

    for i in range(1, n):
        pivot_idx = i - LOOKBACK_PERIOD
        if pivot_idx >= 0 and not np.isnan(piv_lows[i]):
            is_green = close[pivot_idx] > df["Open"].values[pivot_idx]
            if is_green and volume[pivot_idx] > vol_threshold[pivot_idx]:
                supportLevel = piv_lows[i]
                supportLevel_1 = supportLevel - withd[i]

        if pivot_idx >= 0 and not np.isnan(piv_highs[i]):
            is_red = close[pivot_idx] < df["Open"].values[pivot_idx]
            if is_red and volume[pivot_idx] > vol_threshold[pivot_idx]:
                resistanceLevel = piv_highs[i]
                resistanceLevel_1 = resistanceLevel + withd[i]

        brekout_res = res_holds = sup_holds = False
        if not np.isnan(prev_resistanceLevel_1) and not np.isnan(resistanceLevel_1):
            brekout_res = (low[i - 1] < prev_resistanceLevel_1) and (low[i] > resistanceLevel_1)
        if not np.isnan(prev_resistanceLevel) and not np.isnan(resistanceLevel):
            res_holds = (high[i - 1] > prev_resistanceLevel) and (high[i] < resistanceLevel)
        if not np.isnan(prev_supportLevel) and not np.isnan(supportLevel):
            sup_holds = (low[i - 1] < prev_supportLevel) and (low[i] > supportLevel)

        prev_res_is_sup = res_is_sup
        if brekout_res: res_is_sup = True
        elif res_holds: res_is_sup = False

        if i == n - 1:
            signal_type = None
            if sup_holds:
                signal_type = "Destekten Dönüş (Bounce)"
            elif brekout_res and prev_res_is_sup:
                signal_type = "Kırılan Dirençten Dönüş (Retest)"
            elif brekout_res and not prev_res_is_sup:
                signal_type = "Taze Direnç Kırılımı (Breakout)"

            if signal_type:
                seviye = supportLevel if "Destek" in signal_type else resistanceLevel_1
                last_signal = {
                    "tarih": str(df.index[i].date()),
                    "kapanis": round(float(close[i]), 2),
                    "seviye": round(float(seviye), 2) if not np.isnan(seviye) else None,
                    "sinyal_tipi": signal_type
                }

        prev_supportLevel = supportLevel
        prev_resistanceLevel = resistanceLevel
        prev_resistanceLevel_1 = resistanceLevel_1

    if last_signal:
        return True, last_signal
    return False, {}

# --- Telegram Komut İşleyicileri ---

@bot.message_handler(commands=['start', 'help', 'yardim'])
def send_welcome(message):
    welcome_text = (
        "🔍 <b>BIST Hacim-Destek-Direnç Tarayıcı Botu</b>\n\n"
        "Kullanabileceğiniz komutlar:\n"
        "⚡ /tarama - Tüm hisse listesini tarar ve aktif sinyalleri listeler.\n"
        "📈 /hisse [HİSSE_KODU] - Belirtilen hisseyi anlık analiz eder. (Örn: <code>/hisse THYAO</code>)\n\n"
        "<i>Not: Analizler günlük grafik (1D) periyodundadır.</i>"
    )
    bot.reply_to(message, welcome_text, parse_mode="HTML")


@bot.message_handler(commands=['tarama'])
def handle_full_scan(message):
    bot.reply_to(message, "🔄 <b>BIST Listesi taranıyor...</b>\nBu işlem yaklaşık 10-15 saniye sürebilir.", parse_mode="HTML")
    
    signals_found = []
    for ticker in BIST_TICKERS:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                continue
            
            has_signal, signal_data = check_chartprime_sr(df)
            if has_signal:
                hisse_adi = ticker.replace(".IS", "")
                emoji = "🟢" if "Destek" in signal_data["sinyal_tipi"] else "🔴"
                
                msg = (
                    f"⚠️ <b>SİNYAL TESPİT EDİLDİ</b> {emoji}\n"
                    f"📈 <b>Hisse:</b> #{hisse_adi}\n"
                    f"📅 <b>Tarih:</b> {signal_data['tarih']}\n"
                    f"⚡ <b>Sinyal Tipi:</b> {signal_data['sinyal_tipi']}\n"
                    f"💵 <b>Kapanış:</b> {signal_data['kapanis']} TL\n"
                    f"🎯 <b>Referans Seviye:</b> {signal_data['seviye']} TL"
                )
                signals_found.append(msg)
        except Exception as e:
            print(f"Hata ({ticker}): {e}")

    if signals_found:
        for sig in signals_found:
            bot.send_message(message.chat.id, sig, parse_mode="HTML")
    else:
        bot.send_message(message.chat.id, "❌ Şu anda listedeki hiçbir hissede aktif kırılım/dönüş sinyali bulunmuyor.")


@bot.message_handler(commands=['hisse'])
def handle_single_ticker(message):
    # Komuttan sonra hisse ismini ayıkla (Örn: /hisse THYAO -> THYAO)
    text_parts = message.text.split()
    if len(text_parts) < 2:
        bot.reply_to(message, "⚠️ Lütfen bir hisse kodu belirtin. Örn: <code>/hisse THYAO</code>", parse_mode="HTML")
        return

    raw_ticker = text_parts[1].upper().strip()
    ticker = f"{raw_ticker}.IS" if not raw_ticker.endswith(".IS") else raw_ticker

    bot.reply_to(message, f"🔍 <b>#{raw_ticker}</b> analiz ediliyor...", parse_mode="HTML")

    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            bot.send_message(message.chat.id, f"❌ <b>{raw_ticker}</b> verisi Yahoo Finance üzerinden çekilemedi. Kodun doğruluğunu kontrol edin.", parse_mode="HTML")
            return

        has_signal, signal_data = check_chartprime_sr(df)
        if has_signal:
            emoji = "🟢" if "Destek" in signal_data["sinyal_tipi"] else "🔴"
            msg = (
                f"⚠️ <b>SİNYAL TESPİT EDİLDİ</b> {emoji}\n\n"
                f"📈 <b>Hisse:</b> #{raw_ticker}\n"
                f"📅 <b>Tarih:</b> {signal_data['tarih']}\n"
                f"⚡ <b>Sinyal Tipi:</b> {signal_data['sinyal_tipi']}\n"
                f"💵 <b>Kapanış Fiyatı:</b> {signal_data['kapanis']} TL\n"
                f"🎯 <b>Referans Seviye:</b> {signal_data['seviye']} TL"
            )
            bot.send_message(message.chat.id, msg, parse_mode="HTML")
        else:
            bot.send_message(message.chat.id, f"ℹ️ <b>#{raw_ticker}</b> için şu anda aktif bir destek/direnç kırılım veya retest sinyali bulunmuyor.", parse_mode="HTML")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Analiz sırasında bir hata oluştu: {str(e)}")


if __name__ == "__main__":
    print("Bot aktif... Telegram üzerinden komutlar dinleniyor!")
    bot.infinity_polling()
