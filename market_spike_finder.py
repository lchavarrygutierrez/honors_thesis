import argparse
import datetime as dt
import time
from typing import List
import pandas as pd
import requests

UTC = dt.timezone.utc

def dt_utc(date_str: str) -> dt.datetime:
    return dt.datetime.fromisoformat(date_str).replace(tzinfo=UTC, microsecond=0)

def iso_z(d: dt.datetime) -> str:
    return d.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def fetch_coinbase_hourly(start_dt: dt.datetime, end_dt_excl: dt.datetime, product_id: str = "ETH-USD") -> pd.DataFrame:
    """
    Coinbase candles endpoint caps at 300 candles per request.
    Hourly => must chunk (<300 hours per request).
    """
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    headers = {"User-Agent": "pagaya-spike-finder", "Accept": "application/json"}

    if end_dt_excl <= start_dt:
        return pd.DataFrame()

    end_incl = end_dt_excl - dt.timedelta(seconds=1)

    rows: List[list] = []
    cur = start_dt
    CHUNK_HOURS = 280  # safely below 300
    session = requests.Session()

    while cur <= end_incl:
        nxt_excl = min(cur + dt.timedelta(hours=CHUNK_HOURS), end_incl + dt.timedelta(seconds=1))
        req_end = nxt_excl - dt.timedelta(seconds=1)

        params = {"granularity": 3600, "start": iso_z(cur), "end": iso_z(req_end)}
        print(f"Coinbase chunk: {params['start']} -> {params['end']}")

        r = session.get(url, params=params, headers=headers, timeout=60)

        if r.status_code == 429:
            time.sleep(1)
            continue

        if r.status_code != 200:
            raise RuntimeError(f"Coinbase HTTP {r.status_code}\nURL: {r.url}\nBODY: {r.text}")

        data = r.json()
        if isinstance(data, list) and data:
            rows.extend(data)

        cur = nxt_excl
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()

    # Coinbase schema: [time, low, high, open, close, volume]
    cols = ["time", "low", "high", "open", "close", "volume"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("time")

    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = df["time"].astype(int)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)

def spike_windows(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    s = df.loc[df[flag_col].fillna(False), ["time","time_utc","dev_pct"]]
    if s.empty:
        return pd.DataFrame(columns=["start","end","duration_sec","max_dev_pct"])

    wins = []
    start_idx = 0
    for i in range(1, len(s)):
        if s["time"].iloc[i] - s["time"].iloc[i-1] > 3600:
            wins.append(s.iloc[start_idx:i])
            start_idx = i
    wins.append(s.iloc[start_idx:])

    out = []
    for seg in wins:
        out.append({
            "start": seg["time_utc"].iloc[0],
            "end": seg["time_utc"].iloc[-1],
            "duration_sec": int(seg["time"].iloc[-1] - seg["time"].iloc[0]),
            "max_dev_pct": float(seg["dev_pct"].abs().max()),
        })
    return pd.DataFrame(out).sort_values("max_dev_pct", ascending=False).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD UTC inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD UTC exclusive")
    ap.add_argument("--twap_hours", type=int, default=10)
    ap.add_argument("--thresh", type=float, nargs="+", default=[0.005, 0.01])
    ap.add_argument("--out_prefix", default="market")
    args = ap.parse_args()

    start_dt = dt_utc(args.start)
    end_dt_excl = dt_utc(args.end)

    print(f"Fetching Coinbase ETH-USD hourlies {args.start} -> {args.end}")
    df = fetch_coinbase_hourly(start_dt, end_dt_excl, "ETH-USD")

    if df.empty:
        print("No data returned.")
        return

    df["twap"] = df["close"].rolling(args.twap_hours, min_periods=args.twap_hours).mean()
    df["dev_pct"] = (df["close"] - df["twap"]).abs() / df["twap"]

    for th in args.thresh:
        df[f"spike_gt_{int(round(th*10000))}bps"] = df["dev_pct"] > th

    out_raw = f"{args.out_prefix}_hourly_{args.start}_{args.end}.csv"
    df.to_csv(out_raw, index=False)
    print(f"Saved: {out_raw}")

    pref = f"spike_gt_{int(round(max(args.thresh)*10000))}bps"
    spikes = spike_windows(df.dropna(subset=["twap"]), pref)
    out_spk = f"{args.out_prefix}_spikes_{args.start}_{args.end}.csv"
    spikes.to_csv(out_spk, index=False)
    print(f"Saved: {out_spk}")

    print("DONE")

if __name__ == "__main__":
    main()
