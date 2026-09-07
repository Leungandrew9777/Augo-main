#!/usr/bin/env python3
"""
notify.py — Telegram notifications for Augo.

Usage:
    python notify.py                 # weekly summary (predictions + value bets)
    python notify.py --results       # daily grading summary (evaluate.py metrics)
    python notify.py --error "..."   # send an error message
    python notify.py --dry-run       # print the message instead of sending

Env (in .env):
    TELEGRAM_BOT_TOKEN=<bot token from @BotFather>
    TELEGRAM_CHAT_ID=<chat id (user or group)>
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(APP_DIR, "predictions_cache.json")
BANKROLL_FILE = os.path.join(APP_DIR, "bankroll.json")

API = "https://api.telegram.org/bot{token}/sendMessage"


def _cfg() -> tuple[str, str] | None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return None
    return token, chat


def send(text: str, *, dry_run: bool = False) -> bool:
    cfg = _cfg()
    if cfg is None:
        print("ℹ️  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — message printed instead:\n")
        print(text)
        return False
    if dry_run:
        print(text)
        return False
    token, chat = cfg
    try:
        resp = requests.post(API.format(token=token), json={"chat_id": chat, "text": text}, timeout=15)
        resp.raise_for_status()
        print("✓ Telegram message sent.")
        return True
    except Exception as exc:
        print(f"⚠️  Telegram send failed: {exc}")
        return False


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_weekly_summary() -> str:
    cache = _load_cache()
    preds = cache.get("predictions", [])
    gw = str(cache.get("gameweek", "?"))
    if not preds:
        return "Augo: no predictions in cache yet — run run_pipeline.py."

    lines = [f"⚽ Augo · {gw} predictions"]
    safe = []
    for p in preds:
        h, d, a = _f(p.get("prob_home")), _f(p.get("prob_draw")), _f(p.get("prob_away"))
        pick = str(p.get("model_pick", "?"))
        line = (f"  {p.get('home_team', '?')} v {p.get('away_team', '?')} → "
                f"[{pick}]  H {h*100:.0f}% · D {d*100:.0f}% · A {a*100:.0f}%")
        lines.append(line)
        if max(h, d, a) >= 0.65:
            safe.append(f"  • {p.get('home_team')} v {p.get('away_team')} → {pick} ({max(h,d,a)*100:.0f}%)")

    if safe:
        lines.append("")
        lines.append("💪 Safe picks (>65%):")
        lines.extend(safe)

    try:
        from persistence import load_bankroll
        from bankroll import build_current_suggestions
        settings = load_bankroll()
        sugg = build_current_suggestions(preds, float(settings.get("current_bankroll", 1000.0)),
                                         float(settings.get("risk_cap", 0.05)))
        if sugg:
            lines.append("")
            lines.append("💰 Value bets (Kelly):")
            for s in sugg[:8]:
                lines.append(f"  • {s['match']} → {s['bet_outcome']} @ {s['odds']} "
                             f"(edge {s['edge_disp']}, stake {s['stake_disp']})")
    except Exception:
        pass
    return "\n".join(lines)


def build_results_summary() -> str:
    try:
        import evaluate as ev
        records, _ = ev._collect()
        if not records:
            return "Augo: no graded matches yet (predictions_history + results.csv)."
        import pandas as pd
        df = pd.DataFrame(records)
        m = ev._metrics(df)
        return ("📊 Augo grading update\n"
                f"  settled matches: {m['matches']}\n"
                f"  model accuracy: {m['accuracy']:.1%}\n"
                f"  log-loss: {m['log_loss']:.3f} · Brier: {m['brier']:.4f} · ECE: {m['ece']:.3f}\n"
                f"  ROI (fair odds): {m['roi_fair']:+.1%} · ROI (book odds): {m['roi_book']:+.1%}"
                + (f"\n  value bets: {m['value_bets']} · ROI: {m['roi_value']:+.1%}" if m["value_bets"] else ""))
    except Exception as exc:
        return f"Augo: could not build grading summary ({exc})."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true", help="daily grading summary")
    parser.add_argument("--error", default=None, help="send an error message")
    parser.add_argument("--dry-run", action="store_true", help="print only")
    args = parser.parse_args()

    if args.error:
        send(f"⚠️ Augo error:\n{args.error}", dry_run=args.dry_run)
    elif args.results:
        send(build_results_summary(), dry_run=args.dry_run)
    else:
        send(build_weekly_summary(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
