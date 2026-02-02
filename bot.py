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
BOT_TOKEN = "8221222358:AAGVqmSx1b4LnCxiRyzU8bKEqEqKDny8-BM"

# ----- PDL/PDH CONFIG -----
ZONE_PERCENT = 1
COOLDOWN = 1800
TIMEFRAMES_PDL = ["15m", "30m", "1h"]
LOOKBACK_CANDLES = 3

# ----- SUPPLY BOT CONFIG -----
TIMEFRAMES_SUPPLY = ["30m", "1h", "2h", "4h", "1d"]
LOOKBACK_CANDLES_TF = {"30m":100,"1h":80,"2h":60,"4h":50,"1d":50}
PUMP_THRESHOLDS = {"30m":6.0,"1h":8.0,"2h":8.0,"4h":10.0,"1d":15.0}
SUPPLY_ZONE_HEIGHTS = {"30m":0.03,"1h":0.05,"2h":0.06,"4h":0.07,"1d":0.08}
SCAN_INTERVAL = 60
ALERT_COOLDOWN_SUPPLY = 1800
HEARTBEAT_INTERVAL = 180

# =========================================
exchange = ccxt.bybit({
    "enableRateLimit": True,
    "options": {"defaultType": "linear"}
})

# ================= GLOBAL MEMORY =================
SUBSCRIBERS = set()

ALERT_MEMORY_PDL = {}
ALERT_MEMORY_SUPPLY = {}
SUPPLY_ZONES = {}

SUPPLY_TASK_RUNNING = False
LAST_HEARTBEAT = 0
LAST_MANUAL_SCAN = {}

# ================= HELPERS =================
def register_chat(update: Update):
    SUBSCRIBERS.add(update.effective_chat.id)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ---------- PDL / PDH ----------
def in_zone(price, level):
    return abs(price - level)/level*100 <= ZONE_PERCENT

def bullish_engulfing(prev,curr):
    return prev["close"]<prev["open"] and curr["close"]>curr["open"] and curr["open"]<=prev["close"] and curr["close"]>=prev["open"]

def bearish_engulfing(prev,curr):
    return prev["close"]>prev["open"] and curr["close"]<curr["open"] and curr["open"]>=prev["close"] and curr["close"]<=prev["open"]

def make_chart_pdl(df,pdh,pdl):
    df=df.tail(70).reset_index(drop=True)
    fig,(ax,axv)=plt.subplots(2,1,figsize=(10,5),dpi=200,gridspec_kw={'height_ratios':[3,1]},sharex=True)
    fig.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    axv.set_facecolor("#0b0b0b")

    df['ema20']=df['close'].ewm(span=20).mean()
    df['ema50']=df['close'].ewm(span=50).mean()

    for i,r in df.iterrows():
        c="#00ff00" if r["close"]>=r["open"] else "#ff0000"
        ax.plot([i,i],[r["low"],r["high"]],color=c,linewidth=0.8)
        ax.add_patch(plt.Rectangle((i-0.3,min(r["open"],r["close"])),0.6,abs(r["close"]-r["open"]) or 0.0001,color=c))

    ax.plot(df.index,df["ema20"],color="blue",linewidth=1.0)
    ax.plot(df.index,df["ema50"],color="yellow",linewidth=1.0)

    ax.axhline(pdh,color="white",linewidth=0.6)
    ax.axhline(pdl,color="white",linewidth=0.6)

    ax.axis("off")
    axv.axis("off")

    buf=io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf,format="png")
    plt.close()
    buf.seek(0)
    return buf

# ---------- SUPPLY ----------
def compute_rsi(series,period=14):
    delta=series.diff()
    gain=delta.where(delta>0,0)
    loss=-delta.where(delta<0,0)
    rs=gain.rolling(period).mean() / loss.rolling(period).mean().replace(0,1e-6)
    return 100-(100/(1+rs))

def make_chart_supply(df,zone_low,zone_high,candles=50):
    df=df.tail(candles).reset_index(drop=True)
    fig,(ax,axv)=plt.subplots(2,1,figsize=(6,4),dpi=180,gridspec_kw={'height_ratios':[3,1]},sharex=True)
    fig.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    axv.set_facecolor("#0b0b0b")

    for i,r in df.iterrows():
        c="#00ff00" if r["close"]>=r["open"] else "#ff0000"
        ax.plot([i,i],[r["low"],r["high"]],color=c,linewidth=1)
        ax.add_patch(plt.Rectangle((i-0.25,min(r["open"],r["close"])),0.5,abs(r["close"]-r["open"]) or 0.0001,color=c))

    ax.add_patch(plt.Rectangle((0,zone_low),len(df),zone_high-zone_low,color="orange",alpha=0.15))
    ax.axis("off")
    axv.axis("off")

    buf=io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf,format="png")
    plt.close()
    buf.seek(0)
    return buf

# ================= SCANNERS =================
async def scan_market(context: ContextTypes.DEFAULT_TYPE, ignore_cooldown=False):
    tickers=exchange.fetch_tickers()
    futures=[t for s,t in tickers.items() if "/USDT" in s and ":USDT" in s and t.get("percentage") is not None]
    targets=sorted(futures,key=lambda x:x["percentage"],reverse=True)[:30] + \
            sorted(futures,key=lambda x:x["percentage"])[:30]

    for t in targets:
        symbol=t["symbol"]
        daily=exchange.fetch_ohlcv(symbol,"1d",limit=2)
        pdh,pdl=daily[0][2],daily[0][3]

        for tf in TIMEFRAMES_PDL:
            df=pd.DataFrame(exchange.fetch_ohlcv(symbol,tf,limit=70),
                columns=["time","open","high","low","close","vol"])

            for i in range(-LOOKBACK_CANDLES,-1):
                prev,curr=df.iloc[i-1],df.iloc[i]
                if not (in_zone(curr["high"],pdh) or in_zone(curr["low"],pdl)):
                    continue

                if bullish_engulfing(prev,curr):
                    pattern="🟢⬆️ Bullish Engulfing"
                elif bearish_engulfing(prev,curr):
                    pattern="🔴⬇️ Bearish Engulfing"
                else:
                    continue

                key=f"{symbol}_{tf}_{pattern}"
                if not ignore_cooldown and key in ALERT_MEMORY_PDL and time.time()-ALERT_MEMORY_PDL[key]<COOLDOWN:
                    continue

                chart=make_chart_pdl(df,pdh,pdl)
                for chat in SUBSCRIBERS:
                    await context.bot.send_photo(chat_id=chat,photo=chart,
                        caption=f"{symbol.replace(':USDT','')}\nTF: {tf}\n{pattern}")

                ALERT_MEMORY_PDL[key]=time.time()
                break

# ---------- SUPPLY LOOP ----------
async def supply_scan(context: ContextTypes.DEFAULT_TYPE):
    global LAST_HEARTBEAT
    while True:
        now=time.time()
        if now-LAST_HEARTBEAT>HEARTBEAT_INTERVAL:
            log("🟢 Supply bot alive")
            LAST_HEARTBEAT=now

        tickers=exchange.fetch_tickers()
        futures=[t for s,t in tickers.items() if "/USDT" in s and ":USDT" in s and t.get("percentage") is not None]
        targets={t["symbol"]:t for t in sorted(futures,key=lambda x:x["percentage"],reverse=True)[:30] +
                               sorted(futures,key=lambda x:x["percentage"])[:30]}.values()

        for t in targets:
            symbol=t["symbol"]
            clean=symbol.split(":")[0]

            for tf in TIMEFRAMES_SUPPLY:
                df=pd.DataFrame(exchange.fetch_ohlcv(symbol,tf,limit=LOOKBACK_CANDLES_TF[tf]),
                    columns=["time","open","high","low","close","vol"])

                if (symbol,tf) not in SUPPLY_ZONES:
                    low=df["low"].min()
                    high=low*(1+SUPPLY_ZONE_HEIGHTS[tf])
                    SUPPLY_ZONES[(symbol,tf)]={"low":low,"high":high,"sent":False}

                zone=SUPPLY_ZONES[(symbol,tf)]
                curr,prev=df.iloc[-2],df.iloc[-3]

                if zone["low"]<=curr["close"]<=zone["high"] and not zone["sent"]:
                    rsi=compute_rsi(df["close"]).iloc[-2]
                    chart=make_chart_supply(df,zone["low"],zone["high"])
                    emoji="💎✨ " if tf in ["2h","4h","1d"] else ""
                    for chat in SUBSCRIBERS:
                        await context.bot.send_photo(
                            chat_id=chat,
                            photo=chart,
                            caption=f"{emoji}SUPPLY ALERT\n{clean}\nTF: {tf}\nRSI: {rsi:.2f}"
                        )
                    zone["sent"]=True
        await asyncio.sleep(SCAN_INTERVAL)

# ================= COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    register_chat(update)
    await update.message.reply_text("✅ Bot ready\n/scan → PDL/PDH\n/supply → Supply bot")

async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    register_chat(update)
    await update.message.reply_text("🔍 Manual scan started")
    await scan_market(context,ignore_cooldown=True)
    await update.message.reply_text("✅ Scan finished")

async def start_supply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SUPPLY_TASK_RUNNING
    register_chat(update)
    if not SUPPLY_TASK_RUNNING:
        SUPPLY_TASK_RUNNING=True
        asyncio.create_task(supply_scan(context))
    await update.message.reply_text("🔥 Supply bot running 24/7")

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
