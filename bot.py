import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
import io
import os
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ================= CONFIG =================
BOT_TOKEN = ("AAFn9xj_uKBOhGBBf9vSf2FOR11AxFdq5TU")
SUBSCRIBERS_FILE = "subscribers.txt"

# ----- PDL/PDH CONFIG -----
ZONE_PERCENT = 1
COOLDOWN = 1800
TIMEFRAMES_PDL = ["15m", "30m", "1h"]
LOOKBACK_CANDLES = 3

# ----- SUPPLY BOT CONFIG -----
TIMEFRAMES_SUPPLY = ["30m", "1h", "2h", "4h", "1d"]
LOOKBACK_CANDLES_TF = {
    "30m": 100, "1h": 80, "2h": 60, "4h": 50, "1d": 50
}
PUMP_THRESHOLDS = {
    "30m": 6.0, "1h": 8.0, "2h": 8.0, "4h": 10.0, "1d": 15.0
}
SUPPLY_ZONE_HEIGHTS = {
    "30m": 0.03, "1h": 0.05, "2h": 0.06, "4h": 0.07, "1d": 0.08
}
CHART_CANDLES_TF = LOOKBACK_CANDLES_TF
SCAN_INTERVAL = 60
ALERT_COOLDOWN_SUPPLY = 1800
HEARTBEAT_INTERVAL = 180

# =========================================
exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})

# ----- PDL/PDH MEMORY -----
ALERT_MEMORY_PDL = {}
LAST_MANUAL_SCAN = {}

# ----- SUPPLY MEMORY -----
ALERT_MEMORY_SUPPLY = {}
SUPPLY_ZONES = {}
LAST_HEARTBEAT = 0

# ================= HELPERS =================
# --- PDL/PDH ---
def in_zone(price, level):
    return abs(price - level) / level * 100 <= ZONE_PERCENT

def bullish_engulfing(prev, curr):
    return prev["close"] < prev["open"] and curr["close"] > curr["open"] and curr["open"] <= prev["close"] and curr["close"] >= prev["open"]

def bearish_engulfing(prev, curr):
    return prev["close"] > prev["open"] and curr["close"] < curr["open"] and curr["open"] >= prev["close"] and curr["close"] <= prev["open"]

def make_chart_pdl(df, pdh, pdl):
    df = df.tail(70).reset_index(drop=True)
    candle_width = 0.5; wick_width = 0.8
    fig_width, fig_height = 10, 5
    fig, (ax_candle, ax_vol) = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=200,
                                           gridspec_kw={'height_ratios':[3,1]}, sharex=True)
    fig.patch.set_facecolor("#0b0b0b"); ax_candle.set_facecolor("#0b0b0b"); ax_vol.set_facecolor("#0b0b0b")
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    price_range = df['high'].max() - df['low'].min(); price_range = price_range if price_range!=0 else 1

    for i,row in df.iterrows():
        color = "#00ff00" if row["close"]>=row["open"] else "#ff0000"
        ax_candle.plot([i,i],[row["low"],row["high"]], color=color, linewidth=wick_width, zorder=1)
        lower=min(row["open"],row["close"]); height=max(abs(row["close"]-row["open"]), price_range*0.002)
        ax_candle.add_patch(plt.Rectangle((i-candle_width/2, lower), candle_width, height, color=color, zorder=2))
    ax_candle.plot(df.index, df['ema20'], color='blue', linewidth=1.0, zorder=3)
    ax_candle.plot(df.index, df['ema50'], color='yellow', linewidth=1.0, zorder=3)
    ax_candle.axhline(pdh, color="white", linestyle="-", linewidth=0.6)
    ax_candle.axhline(pdl, color="white", linestyle="-", linewidth=0.6)
    max_vol = df['vol'].max()
    scaled_vol = df['vol']/max_vol*0.1*price_range
    colors = ["#00ff00" if df.loc[i,"close"]>=df.loc[i,"open"] else "#ff0000" for i in df.index]
    ax_vol.bar(df.index, scaled_vol, color=colors, width=candle_width, align='center')
    ax_candle.set_xticks([]); ax_candle.set_yticks([]); ax_vol.set_xticks([]); ax_vol.set_yticks([])
    for ax in [ax_candle,ax_vol]:
        for spine in ax.spines.values():
            spine.set_visible(False)
    buf=io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

# --- SUPPLY BOT ---
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def is_bullish_engulfing(prev,curr):
    return prev["close"]<prev["open"] and curr["close"]>curr["open"] and curr["open"]<=prev["close"] and curr["close"]>=prev["open"]

def is_dragonfly_doji(candle):
    body=abs(candle["close"]-candle["open"]); range_=candle["high"]-candle["low"]
    if range_==0: return False
    return body/range_<=0.2 and abs(candle["close"]-candle["high"])/range_<=0.1

def clean_symbol(symbol): return symbol.split(":")[0]

def compute_rsi(series, period=14):
    delta=series.diff(); gain=delta.where(delta>0,0); loss=-delta.where(delta<0,0)
    avg_gain=gain.rolling(period).mean(); avg_loss=loss.rolling(period).mean()
    rs=avg_gain/avg_loss.replace(0,1e-6)
    return 100-(100/(1+rs))

def make_chart_supply(df, zone_low, zone_high, candles=50):
    df=df.tail(candles).reset_index(drop=True)
    fig,(ax,axv)=plt.subplots(2,1,figsize=(5,3),dpi=150,gridspec_kw={'height_ratios':[3,1]},sharex=True)
    fig.patch.set_facecolor("#0b0b0b"); ax.set_facecolor("#0b0b0b"); axv.set_facecolor("#0b0b0b")
    for i,r in df.iterrows():
        c="#00ff00" if r["close"]>=r["open"] else "#ff0000"
        ax.plot([i,i],[r["low"],r["high"]], color=c, linewidth=1)
        ax.add_patch(plt.Rectangle((i-0.25,min(r["open"],r["close"])),0.5,abs(r["close"]-r["open"]) or 0.0001,color=c))
    ax.add_patch(plt.Rectangle((0,zone_low), len(df), zone_high-zone_low,color="orange",alpha=0.1))
    ax.axis("off"); axv.axis("off")
    buf=io.BytesIO(); plt.tight_layout(pad=0); plt.savefig(buf,format="png"); plt.close(); buf.seek(0)
    return buf

# ================= SCANNERS =================
# --- PDL/PDH SCAN ---
async def scan_market(context: ContextTypes.DEFAULT_TYPE, ignore_cooldown=False):
    print(f"\n=== Scan {time.strftime('%H:%M:%S')} ===")
    if not os.path.exists(SUBSCRIBERS_FILE): return
    with open(SUBSCRIBERS_FILE) as f: subscribers=[x.strip() for x in f if x.strip()]
    tickers = exchange.fetch_tickers()
    futures = [t for s,t in tickers.items() if "/USDT" in s and ":USDT" in s and t.get("percentage") is not None]
    gainers=sorted(futures,key=lambda x:x["percentage"],reverse=True)[:30]
    losers=sorted(futures,key=lambda x:x["percentage"])[:30]
    targets=gainers+losers

    for t in targets:
        symbol=t["symbol"]
        try:
            daily=exchange.fetch_ohlcv(symbol,"1d",limit=2)
            pdh=daily[0][2]; pdl=daily[0][3]
            signal_found=False
            for tf in TIMEFRAMES_PDL:
                if signal_found: break
                ohlcv=exchange.fetch_ohlcv(symbol, tf, limit=70)
                df=pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                for i in range(-LOOKBACK_CANDLES,-1):
                    prev=df.iloc[i-1]; curr=df.iloc[i]
                    near_pdh=in_zone(curr["high"],pdh) or in_zone(curr["low"],pdh)
                    near_pdl=in_zone(curr["high"],pdl) or in_zone(curr["low"],pdl)
                    if not (near_pdh or near_pdl): continue
                    zone_text="Near Previous Daily HIGH" if near_pdh else "Near Previous Daily LOW"
                    pattern_text=None; pattern_key=None
                    if bullish_engulfing(prev,curr): pattern_text="🟢⬆️ Bullish Engulfing"; pattern_key="BULL"
                    elif bearish_engulfing(prev,curr): pattern_text="🔴⬇️ Bearish Engulfing"; pattern_key="BEAR"
                    if not pattern_text: continue
                    alert_key=f"{symbol}_{zone_text}_{pattern_key}"
                    if not ignore_cooldown and alert_key in ALERT_MEMORY_PDL and time.time()-ALERT_MEMORY_PDL[alert_key]<COOLDOWN: continue
                    text=f"{symbol.replace(':USDT','')}\n⏱ TF: {tf}\n\n📍 {zone_text}\n{pattern_text}"
                    for chat in subscribers:
                        chart=make_chart_pdl(df,pdh,pdl)
                        try: await context.bot.send_photo(chat_id=chat, photo=chart, caption=text); await asyncio.sleep(1.5)
                        except Exception as e: print(f"Send error {symbol}: {e}")
                    ALERT_MEMORY_PDL[alert_key]=time.time(); signal_found=True; break
        except Exception as e: print(f"Error {symbol}: {e}")
    print("=== Cycle finished ===")

# --- SUPPLY SCAN ---
async def supply_scan(context: ContextTypes.DEFAULT_TYPE):
    global LAST_HEARTBEAT
    log("🤖 Supply scan started")
    if not os.path.exists(SUBSCRIBERS_FILE): return
    with open(SUBSCRIBERS_FILE) as f: subscribers=[x.strip() for x in f if x.strip()]
    while True:
        try:
            now=time.time()
            if now-LAST_HEARTBEAT>HEARTBEAT_INTERVAL: log("🟢 Bot alive | scanning markets"); LAST_HEARTBEAT=now
            tickers=exchange.fetch_tickers()
            futures=[t for s,t in tickers.items() if "/USDT" in s and ":USDT" in s and t.get("percentage") is not None]
            raw_targets=sorted(futures,key=lambda x:x["percentage"],reverse=True)[:30]+sorted(futures,key=lambda x:x["percentage"])[:30]
            targets={t["symbol"]:t for t in raw_targets}.values()
            for t in targets:
                symbol=t["symbol"]; clean=clean_symbol(symbol)
                for tf in TIMEFRAMES_SUPPLY:
                    try: lookback=LOOKBACK_CANDLES_TF.get(tf,50); df=pd.DataFrame(exchange.fetch_ohlcv(symbol,tf,limit=lookback),columns=["time","open","high","low","close","vol"]); await asyncio.sleep(0.15)
                    except Exception as e: log(f"Skipping {symbol} {tf}: {e}"); continue
                    if (symbol,tf) not in SUPPLY_ZONES:
                        pump_start_idx=None
                        for i in range(len(df)-10):
                            start_price=df["low"].iloc[i]; max_close=df["close"].iloc[i:i+10].max()
                            pump_pct=(max_close-start_price)/start_price*100
                            if pump_pct>=PUMP_THRESHOLDS[tf]: pump_start_idx=i; break
                        if pump_start_idx is not None:
                            pump_low=df["low"].iloc[pump_start_idx]; zone_high=pump_low*(1+SUPPLY_ZONE_HEIGHTS[tf])
                            SUPPLY_ZONES[(symbol,tf)]={"low":pump_low,"high":zone_high,"pattern_sent":False}
                            log(f"📌 Supply zone created for {clean} {tf}: {pump_low:.6f} → {zone_high:.6f}")
                    zone=SUPPLY_ZONES.get((symbol,tf))
                    if zone and not zone["pattern_sent"] and len(df)>=3:
                        curr=df.iloc[-2]; prev=df.iloc[-3]
                        if zone["low"]<=curr["close"]<=zone["high"]:
                            if is_bullish_engulfing(prev,curr): pattern="Bullish Engulfing"
                            elif is_dragonfly_doji(curr): pattern="Dragonfly Doji"
                            else: continue
                            key=f"{symbol}_{tf}_{pattern}"
                            if key in ALERT_MEMORY_SUPPLY and time.time()-ALERT_MEMORY_SUPPLY[key]<ALERT_COOLDOWN_SUPPLY: continue
                            rsi=compute_rsi(df["close"]).iloc[-2]; rsi_text=f"{rsi:.2f}" if not pd.isna(rsi) else "N/A"
                            log(f"🚨 ALERT | {clean} | {tf} | {pattern} | RSI {rsi_text}")
                            chart=make_chart_supply(df,zone["low"],zone["high"],candles=CHART_CANDLES_TF.get(tf,50))
                            golden_emoji=""; 
                            if tf in ["2h","4h","1d"]: golden_emoji="💎✨ "
                            text=f"{golden_emoji}🔥 SUPPLY PULLBACK ALERT 🔥\nCoin: {clean}\nTimeframe: {tf}\nSupply zone: {zone['low']:.6f} → {zone['high']:.6f}\nPattern: {pattern}\nRSI: {rsi_text}"
                            for chat in subscribers:
                                try: await context.bot.send_photo(chat_id=chat, photo=chart, caption=text); await asyncio.sleep(1.5)
                                except Exception as e: log(f"Send error {symbol}: {e}")
                            ALERT_MEMORY_SUPPLY[key]=time.time(); zone["pattern_sent"]=True
        except Exception as e: log(f"❌ ERROR: {e}")
        await asyncio.sleep(SCAN_INTERVAL)

# ================= TELEGRAM COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid=str(update.effective_chat.id)
    os.makedirs(os.path.dirname(SUBSCRIBERS_FILE) or ".",exist_ok=True)
    with open(SUBSCRIBERS_FILE,"a+") as f:
        f.seek(0)
        if cid not in f.read(): f.write(cid+"\n")
    await update.message.reply_text("✅ Bot ready. Use /scan for PDL/PDH or /supply for Supply Bot")

async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid=str(update.effective_chat.id)
    if cid in LAST_MANUAL_SCAN and time.time()-LAST_MANUAL_SCAN[cid]<300: await update.message.reply_text("⏱ Please wait before using /scan again."); return
    LAST_MANUAL_SCAN[cid]=time.time()
    await update.message.reply_text("⚡ Manual PDL/PDH scan started! Please wait...")
    await scan_market(context, ignore_cooldown=True)
    await update.message.reply_text("✅ Manual scan finished!")

async def start_supply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚡ Supply scan started")
    asyncio.create_task(supply_scan(context))

# ================= MAIN =================
def main():
    log("🤖 Bot launched")
    app=Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("supply", start_supply))
    app.run_polling()

if __name__=="__main__":
    main()

