# ==============================================================================
# CHARTPRIME "SUPPORT & RESISTANCE (HIGH VOLUME BOXES)" TARAYICISI - v2 + TELEGRAM
# ==============================================================================
# GitHub Actions ile zamanlanmis olarak calisir, sonuclari Telegram'a gonderir.
# Ortam degiskenleri (GitHub Secrets uzerinden gelir):
#   TELEGRAM_BOT_TOKEN
#   TELEGRAM_CHAT_ID
# ==============================================================================

import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1) INDIKATOR PARAMETRELERI
# ------------------------------------------------------------------------------
LOOKBACK_PERIOD = 20
VOL_LEN         = 2
BOX_WIDTH       = 1.0
ATR_LEN         = 200
DATA_PERIOD     = "3y"
MIN_BARS        = 260

# ------------------------------------------------------------------------------
# 2) BIST HISSE LISTESI
# ------------------------------------------------------------------------------
BIST_TICKERS = [
    "A1CAP","A1YEN","AAGYO","ACSEL","ADEL","ADESE","ADGYO","AEFES","AFYON","AGESA",
    "AGHOL","AGROT","AGYO","AHGAZ","AHSGY","AKBNK","AKCNS","AKENR","AKFGY","AKFIS",
    "AKFYE","AKGRT","AKHAN","AKMGY","AKSA","AKSEN","AKSGY","AKSUE","AKYHO","ALARK",
    "ALBRK","ALCAR","ALCTL","ALFAS","ALGYO","ALKA","ALKIM","ALKLC","ALTNY","ALVES",
    "ANELE","ANGEN","ANHYT","ANSGR","ARASE","ARCLK","ARDYZ","ARENA","ARFYE","ARMGD",
    "ARSAN","ARTMS","ARZUM","ASELS","ASGYO","ASTOR","ASUZU","ATAGY","ATAKP","ATATP",
    "ATATR","ATEKS","ATLAS","ATSYH","AVGYO","AVHOL","AVOD","AVPGY","AVTUR","AYCES",
    "AYDEM","AYEN","AYES","AYGAZ","AZTEK","BAGFS","BAHKM","BAKAB","BALAT","BALSU",
    "BANVT","BARMA","BASCM","BASGZ","BAYRK","BEGYO","BERA","BESLR","BESTE","BETAE",
    "BEYAZ","BFREN","BIENY","BIGCH","BIGEN","BIGTK","BIMAS","BINBN","BINHO","BIOEN",
    "BIZIM","BJKAS","BLCYT","BLUME","BMSCH","BMSTL","BNTAS","BOBET","BORLS","BORSK",
    "BOSSA","BRISA","BRKO","BRKSN","BRKVY","BRLSM","BRMEN","BRSAN","BRYAT","BSOKE",
    "BTCIM","BUCIM","BULGS","BURCE","BURVA","BVSAN","BYDNR","CANTE","CASA","CATES",
    "CCOLA","CELHA","CEMAS","CEMTS","CEMZY","CEOEM","CGCAM","CIMSA","CLEBI","CMBTN",
    "CMENT","CONSE","COSMO","CRDFA","CRFSA","CUSAN","CVKMD","CWENE","DAGI","DAPGM",
    "DARDL","DCTTR","DENGE","DERHL","DERIM","DESA","DESPC","DEVA","DGATE","DGGYO",
    "DGNMO","DIRIT","DITAS","DMRGD","DMSAS","DNISI","DOAS","DOCO","DOFER","DOFRB",
    "DOGUB","DOHOL","DOKTA","DSTKF","DUNYH","DURDO","DURKN","DYOBY","DZGYO","EBEBK",
    "ECILC","ECOGR","ECZYT","EDATA","EDIP","EFOR","EGEEN","EGEGY","EGEPO","EGGUB",
    "EGPRO","EGSER","EKDMR","EKGYO","EKIM","EKIZ","EKOS","EKSUN","ELITE","EMKEL",
    "EMNIS","EMPAE","ENDAE","ENERY","ENJSA","ENKAI","ENPRA","ENSRI","ENTRA","EPLAS",
    "ERBOS","ERCB","EREGL","ERSU","ESCAR","ESCOM","ESEN","ETILR","ETYAT","EUHOL",
    "EUKYO","EUPWR","EUREN","EUYO","EYGYO","FADE","FENER","FLAP","FMIZP","FONET",
    "FORMT","FORTE","FRIGO","FRMPL","FROTO","FZLGY","GARAN","GARFA","GATEG","GEDIK",
    "GEDZA","GENIL","GENKM","GENTS","GEREL","GESAN","GIPTA","GLBMD","GLCVY","GLRMK",
    "GLRYH","GLYHO","GMTAS","GOKNR","GOLDA","GOLTS","GOODY","GOZDE","GRNYO","GRSEL",
    "GRTHO","GSDDE","GSDHO","GSRAY","GUBRF","GUNDG","GWIND","GZNMI","HALKB","HATEK",
    "HATSN","HDFGS","HEDEF","HEKTS","HKTM","HLGYO","HOROZ","HRKET","HTTBT","HUBVC",
    "HUNER","HURGZ","ICBCT","ICUGS","IDGYO","IEYHO","IHAAS","IHEVA","IHGZT","IHLAS",
    "IHLGM","IHYAY","IMASM","INDES","INFO","INGRM","INTEK","INTEM","INVEO","INVES",
    "ISATR","ISBIR","ISBTR","ISCTR","ISDMR","ISFIN","ISGSY","ISGYO","ISKPL","ISKUR",
    "ISMEN","ISSEN","ISYAT","IZENR","IZFAS","IZINV","IZMDC","JANTS","KAPLM","KAREL",
    "KARSN","KARTN","KATMR","KAYSE","KBORU","KCAER","KCHOL","KENT","KERVN","KFEIN",
    "KGYO","KIMMR","KLGYO","KLKIM","KLMSN","KLNMA","KLRHO","KLSER","KLSYN","KLYPV",
    "KMPUR","KNFRT","KOCMT","KONKA","KONTR","KONYA","KOPOL","KORDS","KOTON","KRDMA",
    "KRDMB","KRDMD","KRGYO","KRONT","KRPLS","KRSTL","KRTEK","KRVGD","KSTUR","KTLEV",
    "KTSKR","KUTPO","KUVVA","KUYAS","KZBGY","KZGYO","LIDER","LIDFA","LILAK","LINK",
    "LKMNH","LMKDC","LOGO","LRSHO","LUKSK","LXGYO","LYDHO","LYDYE","MAALT","MACKO",
    "MAGEN","MAKIM","MAKTK","MANAS","MARBL","MARKA","MARMR","MARTI","MAVI","MCARD",
    "MEDTR","MEGAP","MEGMT","MEKAG","MEPET","MERCN","MERIT","MERKO","METRO","MEYSU",
    "MGROS","MHRGY","MIATK","MMCAS","MNDRS","MNDTR","MOBTL","MOGAN","MOPAS","MPARK",
    "MRGYO","MRSHL","MSGYO","MTRKS","MTRYO","MZHLD","NATEN","NETAS","NETCD","NIBAS",
    "NTGAZ","NTHOL","NUGYO","NUHCM","OBAMS","OBASE","ODAS","ODINE","OFSYM","ONCSM",
    "ONRYT","ORCAY","ORGE","ORMA","ORZAX","OSMEN","OSTIM","OTKAR","OTTO","OYAKC",
    "OYAYO","OYLUM","OYYAT","OZATD","OZGYO","OZKGY","OZRDN","OZSUB","OZYSR","PAGYO",
    "PAHOL","PAMEL","PAPIL","PARSN","PASEU","PATEK","PCILT","PEKGY","PENGD","PENTA",
    "PETKM","PETUN","PGSUS","PINSU","PKART","PKENT","PLTUR","PNLSN","PNSUT","POLHO",
    "POLTK","PRDGS","PRKAB","PRKME","PRZMA","PSDTC","PSGYO","QNBFK","QNBTR","QUAGR",
    "RALYH","RAYSG","REEDR","RGYAS","RNPOL","RODRG","RTALB","RUBNS","RUZYE","RYGYO",
    "RYSAS","SAFKR","SAHOL","SAMAT","SANEL","SANFM","SANKO","SARKY","SASA","SAYAS",
    "SDTTR","SEGMN","SEGYO","SEKFK","SEKUR","SELEC","SELVA","SERNT","SEYKM","SILVR",
    "SISE","SKBNK","SKTAS","SKYLP","SKYMD","SMART","SMRTG","SMRVA","SNGYO","SNICA",
    "SNPAM","SODSN","SOHOE","SOKE","SOKM","SONME","SRVGY","SUMAS","SUNTK","SURGY",
    "SUWEN","SVGYO","TABGD","TARKM","TATEN","TATGD","TAVHL","TBORG","TCELL","TCKRC",
    "TDGYO","TEHOL","TEKTU","TERA","TEZOL","TGSAS","THYAO","TKFEN","TKNSA","TLMAN",
    "TMPOL","TMSN","TNZTP","TOASO","TRALT","TRCAS","TRENJ","TRGYO","TRHOL","TRILC",
    "TRMET","TSGYO","TSKB","TSPOR","TTKOM","TTRAK","TUCLK","TUKAS","TUPRS","TUREX",
    "TURGG","TURSG","UCAYM","UFUK","ULAS","ULKER","ULUFA","ULUSE","ULUUN","UMPAS",
    "UNLU","USAK","VAKBN","VAKFA","VAKFN","VAKKO","VANGD","VBTYZ","VERTU","VERUS",
    "VESBE","VESTL","VKFYO","VKGYO","VKING","VRGYO","VSNMD","YAPRK","YATAS","YAYLA",
    "YBTAS","YEOTK","YESIL","YGGYO","YIGIT","YKBNK","YKSLN","YONGA","YUNSA","YYAPI",
    "YYLGD","ZEDUR","ZERGY","ZGYO","ZOREN","ZRGYO",
]

def get_symbols():
    return [t.strip().upper() + ".IS" for t in BIST_TICKERS]

# ------------------------------------------------------------------------------
# 3) MATEMATIKSEL FONKSIYONLAR
# ------------------------------------------------------------------------------
def rma_atr(df, length=200):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean().values


def get_delta_volume(df):
    direction = np.sign(df["Close"] - df["Open"]).values
    last_dir = 1
    out = np.empty(len(direction))
    for i in range(len(direction)):
        if direction[i] > 0:
            last_dir = 1
        elif direction[i] < 0:
            last_dir = -1
        out[i] = last_dir
    return df["Volume"].values * out


def calculate_pivots(src, lookback):
    n = len(src)
    piv_highs = np.full(n, np.nan)
    piv_lows = np.full(n, np.nan)
    for i in range(lookback * 2, n):
        window = src[i - (lookback * 2): i + 1]
        center = src[i - lookback]
        if center == np.max(window) and np.sum(window == center) == 1:
            piv_highs[i] = center
        if center == np.min(window) and np.sum(window == center) == 1:
            piv_lows[i] = center
    return piv_highs, piv_lows


def drop_incomplete_today(df):
    if len(df) and df.index[-1].date() == datetime.now().date():
        df = df.iloc[:-1]
    return df


# ------------------------------------------------------------------------------
# 4) ANA INDIKATOR MANTIGI & SINYAL KONTROLU
# ------------------------------------------------------------------------------
def check_chartprime_sr(df: pd.DataFrame):
    if df is None or len(df) < MIN_BARS:
        return False, {}

    df = drop_incomplete_today(df)
    n = len(df)
    if n < MIN_BARS:
        return False, {}

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    vol = get_delta_volume(df)
    vol_25 = vol / 2.5
    vol_hi = pd.Series(vol_25).rolling(VOL_LEN, min_periods=1).max().values
    vol_lo = pd.Series(vol_25).rolling(VOL_LEN, min_periods=1).min().values

    atr = rma_atr(df, ATR_LEN)
    withd = atr * BOX_WIDTH

    piv_highs, piv_lows = calculate_pivots(close, LOOKBACK_PERIOD)

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
        if not np.isnan(piv_lows[i]) and vol[i] > vol_hi[i]:
            supportLevel = piv_lows[i]
            supportLevel_1 = supportLevel - withd[i]

        if not np.isnan(piv_highs[i]) and vol[i] < vol_lo[i]:
            resistanceLevel = piv_highs[i]
            resistanceLevel_1 = resistanceLevel + withd[i]

        brekout_res = False
        res_holds = False
        sup_holds = False

        if not np.isnan(prev_resistanceLevel_1) and not np.isnan(resistanceLevel_1):
            brekout_res = (low[i - 1] < prev_resistanceLevel_1) and (low[i] > resistanceLevel_1)
        if not np.isnan(prev_resistanceLevel) and not np.isnan(resistanceLevel):
            res_holds = (high[i - 1] > prev_resistanceLevel) and (high[i] < resistanceLevel)
        if not np.isnan(prev_supportLevel) and not np.isnan(supportLevel):
            sup_holds = (low[i - 1] < prev_supportLevel) and (low[i] > supportLevel)

        prev_res_is_sup = res_is_sup
        if brekout_res:
            res_is_sup = True
        elif res_holds:
            res_is_sup = False

        if i == n - 1:
            signal_type = None
            if sup_holds:
                signal_type = "Destekten Donus"
            elif brekout_res and prev_res_is_sup:
                signal_type = "Kirilan Direncten Donus (Retest)"
            elif brekout_res and not prev_res_is_sup:
                signal_type = "Taze Direnc Kirilimi"

            if signal_type:
                seviye = supportLevel if "Destek" in signal_type else resistanceLevel_1
                last_signal = {
                    "tarih": str(df.index[i].date()),
                    "kapanis": round(float(close[i]), 4),
                    "seviye": round(float(seviye), 4) if not np.isnan(seviye) else None,
                    "sinyal_tipi": signal_type,
                }

        prev_supportLevel = supportLevel
        prev_resistanceLevel = resistanceLevel
        prev_resistanceLevel_1 = resistanceLevel_1

    if last_signal:
        return True, last_signal
    return False, {}


# ------------------------------------------------------------------------------
# 5) TELEGRAM GONDERIM
# ------------------------------------------------------------------------------
def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("UYARI: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tanimli degil, mesaj gonderilmiyor.")
        print(text)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram mesaj limiti ~4096 karakter, uzun mesajlari parcala
    max_len = 3500
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [text]
    for chunk in chunks:
        try:
            r = requests.post(url, data={"chat_id": chat_id, "text": chunk}, timeout=20)
            if r.status_code != 200:
                print(f"Telegram gonderim hatasi: {r.text}")
        except Exception as e:
            print(f"Telegram gonderim istisnasi: {e}")


# ------------------------------------------------------------------------------
# 6) TOPLU BIST TARAMASI
# ------------------------------------------------------------------------------
def scan_chartprime_bist():
    symbols = get_symbols()
    results = []

    print(f"Toplam {len(symbols)} hisse indiriliyor...")
    data = yf.download(symbols, period=DATA_PERIOD, interval="1d",
                        threads=True, progress=True, auto_adjust=False)

    for sym in symbols:
        try:
            if sym not in data["Close"].columns:
                continue
            df = pd.DataFrame({
                "Open": data["Open"][sym],
                "High": data["High"][sym],
                "Low": data["Low"][sym],
                "Close": data["Close"][sym],
                "Volume": data["Volume"][sym],
            }).dropna()

            is_signal, detail = check_chartprime_sr(df)
            if is_signal:
                results.append({
                    "Hisse": sym.replace(".IS", ""),
                    "Tarih": detail["tarih"],
                    "Kapanis": detail["kapanis"],
                    "Seviye": detail["seviye"],
                    "Sinyal": detail["sinyal_tipi"],
                })
        except Exception:
            continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_result = scan_chartprime_bist()

    today_str = datetime.now().strftime("%d.%m.%Y")
    if df_result.empty:
        msg = f"📊 BIST ChartPrime Taramasi ({today_str})\n\nBugun alim seviyesinde hisse bulunamadi."
    else:
        df_result = df_result.sort_values(by=["Sinyal", "Hisse"])
        lines = [f"📊 BIST ChartPrime Taramasi ({today_str})",
                 f"{len(df_result)} hisse sinyal verdi:\n"]
        for _, row in df_result.iterrows():
            lines.append(
                f"• {row['Hisse']} | {row['Sinyal']} | Kapanis: {row['Kapanis']} | Seviye: {row['Seviye']}"
            )
        msg = "\n".join(lines)

    print(msg)
    send_telegram(msg)
