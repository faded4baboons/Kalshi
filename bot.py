# bot.py - Kalshi weather team: news agent + betting agent + risk referee
# PAPER ONLY. This file contains no code that places real orders.
# Runs once per call. GitHub Actions calls it every 30 minutes.

import json, math, os, re, datetime as dt
import requests

# ---------- SETTINGS ----------
SERIES = "KXHIGHTPHX"            # Kalshi Phoenix daily high temperature
LAT, LON = 33.4342, -112.0116    # Phoenix Sky Harbor, the settlement station
NWS_OFFICE = "PSR"               # NWS Phoenix office (forecast discussions)
TZ = "America/Phoenix"
MIN_EDGE = 0.08                  # 8 cents of edge after fees
QTY = 10                         # contracts per paper trade
MAX_EXPOSURE = 50.0              # max $ in open paper trades
DAILY_LOSS_LIMIT = 30.0          # referee stops new trades after this much settled loss in 24h
MAX_TRADES_PER_RUN = 3
START_BANK = 1000.0
BASE_SPREAD_F = 2.0              # default extra forecast uncertainty (unvalidated)
BIAS_CAP_F = 3.0                 # news agent can shift the forecast at most this much
SPREAD_MIN_F, SPREAD_MAX_F = 1.5, 4.0
NEWS_EVERY_HOURS = 3
NEWS_MODEL = "claude-haiku-4-5-20251001"
STATE_FILE = "state.json"

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
UA = {"User-Agent": "kalshi-paper-bot (personal research)"}

def utcnow():
    return dt.datetime.now(dt.timezone.utc)

def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_iso(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)

def phx_today():
    return (utcnow() - dt.timedelta(hours=7)).date()

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"bank": START_BANK, "open": [], "closed": [], "log": [], "equity": [],
            "news": None, "board": [], "status": "", "last_run": None}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=1)

def log(s, agent, text, amt="", cls=""):
    s["log"].insert(0, {"t": iso(utcnow()), "agent": agent, "text": text, "amt": amt, "cls": cls})
    s["log"] = s["log"][:60]

# ================= NEWS AGENT (AI) =================
def fetch_afd():
    r = requests.get(f"https://api.weather.gov/products/types/AFD/locations/{NWS_OFFICE}", headers=UA, timeout=20)
    r.raise_for_status()
    items = r.json().get("@graph", [])
    if not items:
        return None, ""
    pid = items[0]["id"]
    r2 = requests.get(f"https://api.weather.gov/products/{pid}", headers=UA, timeout=20)
    r2.raise_for_status()
    return pid, r2.json().get("productText", "")

def fetch_alerts():
    r = requests.get(f"https://api.weather.gov/alerts/active?point={LAT},{LON}", headers=UA, timeout=20)
    r.raise_for_status()
    return [f["properties"].get("headline", "") for f in r.json().get("features", [])][:5]

def clean_afd(text):
    """Keep the readable discussion: join wrapped lines, drop headers, cap length."""
    t = text.replace("\r", "")
    i = t.find(".DISCUSSION")
    if i == -1:
        i = t.find(".KEY MESSAGES")
    t = t[i:] if i != -1 else t
    j = t.find(".AVIATION")
    t = t[:j] if j != -1 else t
    paras = [" ".join(p.split()) for p in t.split("\n\n")]
    return "\n\n".join(p for p in paras if p)[:6000]

def ask_claude(afd_text, alerts, days):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    system = ("You are a weather analyst helping score Kalshi markets on the official daily HIGH temperature "
              "at Phoenix Sky Harbor. You read the NWS Area Forecast Discussion and alerts, and compare what "
              "forecasters say with raw computer models. Reply with ONLY a JSON object, no other text.")
    prompt = (f"Days to score (Phoenix local): {', '.join(days)}\n\n"
              f"Active alerts: {alerts if alerts else 'none'}\n\n"
              f"NWS Area Forecast Discussion:\n{afd_text[:12000]}\n\n"
              "Return JSON exactly like:\n"
              '{"days": {"YYYY-MM-DD": {"bias_f": 0.0, "spread_f": 2.0, "reason": "short"}}, '
              '"summary": "2-3 plain sentences", "pause_trading": false, '
              '"highlights": [{"quote": "exact words copied from the discussion", "topic": "storms", "lean": "cooler"}]}\n'
              "highlights = 4 to 10 short phrases (2-8 words) copied EXACTLY from the discussion that matter for "
              "the daily high. topic is one of: storms, heat, clouds, wind, smoke, models, uncertainty. "
              "lean is one of: cooler, warmer, uncertain.\n"
              "bias_f = how many degrees F the official high will likely differ from raw global model output "
              "(negative = cooler). Use 0 if the discussion gives no clear reason. "
              "spread_f = uncertainty in degrees F, 1.5 (very confident) to 4 (very unsure). "
              "pause_trading = true only for truly unusual situations (major storms, dust storms, "
              "forecasters saying models are unreliable).")
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": NEWS_MODEL, "max_tokens": 600, "system": system,
                            "messages": [{"role": "user", "content": prompt}]}, timeout=60)
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    text = re.sub(r"```json|```", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None

def news_agent(s):
    last = s.get("news") or {}
    if last.get("ran_at") and utcnow() - parse_iso(last["ran_at"]) < dt.timedelta(hours=NEWS_EVERY_HOURS):
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        s["news"] = {"ran_at": iso(utcnow()), "summary": "News agent is off: add the ANTHROPIC_API_KEY secret to turn it on.",
                     "days": {}, "pause_trading": False, "source": None}
        return
    pid, afd = fetch_afd()
    if pid and pid == last.get("source"):
        last["ran_at"] = iso(utcnow())
        return   # same discussion as last time, no need to pay for another read
    alerts = fetch_alerts()
    days = [(phx_today() + dt.timedelta(days=i)).isoformat() for i in (1, 2)]
    doc = clean_afd(afd)
    out = ask_claude(doc, alerts, days)
    if not out:
        log(s, "News", "Could not read the AI response this round, using plain model settings")
        return
    clean = {}
    for d, v in (out.get("days") or {}).items():
        b = max(-BIAS_CAP_F, min(BIAS_CAP_F, num(v.get("bias_f")) or 0.0))
        sp = max(SPREAD_MIN_F, min(SPREAD_MAX_F, num(v.get("spread_f")) or BASE_SPREAD_F))
        clean[d] = {"bias_f": b, "spread_f": sp, "reason": str(v.get("reason", ""))[:160]}
    low = doc.lower()
    topics = {"storms", "heat", "clouds", "wind", "smoke", "models", "uncertainty"}
    leans = {"cooler", "warmer", "uncertain"}
    highlights = []
    for h in out.get("highlights") or []:
        q = " ".join(str(h.get("quote", "")).split())
        if 3 <= len(q) <= 80 and q.lower() in low:   # drop anything the AI didn't copy exactly
            highlights.append({"quote": q, "topic": h.get("topic") if h.get("topic") in topics else "models",
                               "lean": h.get("lean") if h.get("lean") in leans else "uncertain"})
    s["news"] = {"ran_at": iso(utcnow()), "source": pid, "alerts": alerts, "days": clean,
                 "summary": str(out.get("summary", ""))[:500], "pause_trading": bool(out.get("pause_trading")),
                 "doc": doc, "highlights": highlights[:12], "dropped": max(0, len(out.get("highlights") or []) - len(highlights))}
    hist = s.get("news_history", [])
    hist.append({"t": iso(utcnow()), "days": {d: v["bias_f"] for d, v in clean.items()}, "n": len(highlights)})
    s["news_history"] = hist[-40:]
    log(s, "News", "New forecast discussion read: " + s["news"]["summary"][:140])

# ================= BETTING AGENT =================
def quotes(m):
    ya, na_ = num(m.get("yes_ask_dollars")), num(m.get("no_ask_dollars"))
    yb, nb = num(m.get("yes_bid_dollars")), num(m.get("no_bid_dollars"))
    if ya is None and m.get("yes_ask") is not None: ya = num(m["yes_ask"]) / 100
    if na_ is None and m.get("no_ask") is not None: na_ = num(m["no_ask"]) / 100
    if yb is None and m.get("yes_bid") is not None: yb = num(m["yes_bid"]) / 100
    if nb is None and m.get("no_bid") is not None: nb = num(m["no_bid"]) / 100
    if ya is None and nb is not None: ya = 1 - nb
    if na_ is None and yb is not None: na_ = 1 - yb
    return ya, na_

def market_date(ticker):
    try:
        return dt.datetime.strptime(ticker.split("-")[1], "%y%b%d").date()
    except Exception:
        return None

def get_ensemble():
    p = {"latitude": LAT, "longitude": LON, "daily": "temperature_2m_max", "temperature_unit": "fahrenheit",
         "timezone": TZ, "models": "gfs_seamless", "forecast_days": 4}
    r = requests.get(ENSEMBLE, params=p, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    keys = [k for k in d if k.startswith("temperature_2m_max")]
    return {dt.date.fromisoformat(day): [d[k][i] for k in keys if d[k][i] is not None] for i, day in enumerate(d["time"])}

def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def prob_yes(m, members, bias, spread):
    st, lo, hi = m.get("strike_type"), num(m.get("floor_strike")), num(m.get("cap_strike"))
    def hit(t):
        if st == "greater": return lo is not None and t > lo
        if st == "less": return hi is not None and t < hi
        if st == "between": return lo is not None and hi is not None and lo <= t <= hi
        return False
    tot = 0.0
    for v in members:
        mu = v + bias
        for t in range(int(mu - 15), int(mu + 16)):
            if hit(t):
                tot += phi((t + 0.5 - mu) / spread) - phi((t - 0.5 - mu) / spread)
    return min(max(tot / max(len(members), 1), 0.01), 0.99)

def fee(price):
    return 0.07 * price * (1 - price)

def settle(s):
    still = []
    for p in s["open"]:
        try:
            r = requests.get(f"{KALSHI}/markets/{p['ticker']}", timeout=20)
            r.raise_for_status()
            res = (r.json().get("market", {}).get("result") or "").lower()
        except Exception:
            still.append(p)
            continue
        if res not in ("yes", "no"):
            still.append(p)
            continue
        won = (p["side"] == "YES") == (res == "yes")
        pnl = p["qty"] * ((1.0 if won else 0.0) - p["cost"] - p["fee"])
        s["bank"] += pnl
        p.update({"won": won, "pnl": pnl, "result": res, "closed_at": iso(utcnow())})
        s["closed"].append(p)
        log(s, "Betting", f"{'Won' if won else 'Lost'} {p['side']} on {p['label']}", f"{'+' if pnl >= 0 else '-'}${abs(pnl):.2f}", "pos" if won else "neg")
    s["open"] = still

# ================= RISK REFEREE (rules only) =================
def exposure(s):
    return sum(p["qty"] * p["cost"] for p in s["open"])

def recent_loss(s):
    cutoff = utcnow() - dt.timedelta(hours=24)
    return -sum(c["pnl"] for c in s["closed"] if parse_iso(c["closed_at"]) >= cutoff and c["pnl"] < 0)

def referee_blocks(s):
    if os.environ.get("BOT_PAUSED", "").lower() == "true":
        return "Paused by you (BOT_PAUSED variable)"
    if (s.get("news") or {}).get("pause_trading"):
        return "Paused: news agent flagged unusual weather"
    if recent_loss(s) >= DAILY_LOSS_LIMIT:
        return f"Paused: ${recent_loss(s):.2f} lost in 24h (limit ${DAILY_LOSS_LIMIT:.0f})"
    return None

# ================= RUN ONE CYCLE =================
def run():
    s = load_state()
    status = []
    try:
        news_agent(s)
    except Exception as e:
        status.append(f"News agent error: {e}")
    try:
        settle(s)
    except Exception as e:
        status.append(f"Settlement error: {e}")
    block = referee_blocks(s)
    board = []
    try:
        r = requests.get(f"{KALSHI}/markets", params={"series_ticker": SERIES, "status": "open", "limit": 200}, timeout=20)
        r.raise_for_status()
        markets = r.json().get("markets", [])
        ens = get_ensemble()
        news_days = (s.get("news") or {}).get("days", {})
        held = {p["ticker"] for p in s["open"]}
        trades = 0
        for m in markets:
            tk = m.get("ticker", "")
            d = market_date(tk)
            if d is None or d <= phx_today() or not ens.get(d):
                continue
            ya, na_ = quotes(m)
            if not ya or not na_:
                continue
            adj = news_days.get(d.isoformat(), {})
            bias, spread = adj.get("bias_f", 0.0), adj.get("spread_f", BASE_SPREAD_F)
            p = prob_yes(m, ens[d], bias, spread)
            ey, en = p - ya - fee(ya), (1 - p) - na_ - fee(na_)
            label = f"{d.strftime('%a %b %d')}: {m.get('yes_sub_title') or m.get('subtitle') or tk}"
            board.append({"label": label, "p": round(p, 4), "ya": ya, "na": na_, "ey": round(ey, 4), "en": round(en, 4),
                          "held": tk in held, "bias": bias, "spread": spread})
            if tk in held or block or trades >= MAX_TRADES_PER_RUN:
                continue
            side, cost, edge = (("YES", ya, ey) if ey >= MIN_EDGE else ("NO", na_, en) if en >= MIN_EDGE else (None, None, None))
            if not side:
                continue
            if exposure(s) + QTY * cost > MAX_EXPOSURE:
                continue
            s["open"].append({"ticker": tk, "label": label, "side": side, "cost": cost, "fee": fee(cost), "qty": QTY,
                              "model_p": p, "edge": edge, "bias": bias, "spread": spread, "opened_at": iso(utcnow())})
            held.add(tk)
            trades += 1
            log(s, "Betting", f"Paper bought {QTY} {side} at {cost*100:.0f}¢, {label} (edge {edge*100:+.1f}¢)", f"-${QTY*cost:.2f}", "pos" if side == "YES" else "neg")
    except Exception as e:
        status.append(f"Market scan error: {e}")
    board.sort(key=lambda b: -max(b["ey"], b["en"]))
    s["board"] = board[:15]
    s["equity"].append({"t": iso(utcnow()), "v": round(s["bank"], 2)})
    s["equity"] = s["equity"][-500:]
    s["status"] = " ".join(status) if status else (block or "Running normally")
    s["last_run"] = iso(utcnow())
    save_state(s)
    print(s["status"])

if __name__ == "__main__":
    run()
