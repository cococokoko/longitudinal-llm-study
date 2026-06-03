#!/usr/bin/env python3
"""
notify.py — Send a daily summary email for today's pipeline run.

Checks study.db for today's wave, counts OK vs error responses,
and emails a short report. Exits with code 1 if no wave ran today.

Usage:
  python notify.py                          # uses .env for credentials
  python notify.py --date 2026-06-03       # check a specific date
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import smtplib
import sqlite3
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import urllib.request

from dotenv import load_dotenv

load_dotenv()

LOW_BALANCE_THRESHOLD = 30.0   # warn if limit_remaining drops below this


# ── OpenRouter balance ────────────────────────────────────────────────────────

def fetch_balance() -> dict | None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
            return {
                "limit":           data.get("limit"),
                "limit_remaining": data.get("limit_remaining"),
                "usage":           data.get("usage"),
                "usage_monthly":   data.get("usage_monthly"),
            }
    except Exception:
        return None


# ── DB helpers ────────────────────────────────────────────────────────────────

def check_run(db_path: str, date: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id FROM study_waves WHERE name = ?", (date,))
    wave = cur.fetchone()
    if not wave:
        conn.close()
        return {"ran": False, "date": date}

    wave_id = wave["id"]

    cur.execute("""
        SELECT
            json_extract(call_params, '$.model_id_requested') AS model,
            COUNT(*) AS total,
            SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END) AS ok,
            SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
            SUM(COALESCE(input_tokens,  0)) AS in_tok,
            SUM(COALESCE(output_tokens, 0)) AS out_tok,
            ROUND(SUM(COALESCE(cost_usd, 0)), 4) AS cost
        FROM response_records
        WHERE wave_id = ?
        GROUP BY model
        ORDER BY model
    """, (wave_id,))
    rows = [dict(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT error, COUNT(*) AS cnt
        FROM response_records
        WHERE wave_id = ? AND error IS NOT NULL
        GROUP BY
            CASE
                WHEN error LIKE '%403%' THEN '403 Forbidden'
                WHEN error LIKE '%404%' THEN '404 Not Found'
                WHEN error LIKE '%402%' THEN '402 Payment Required'
                WHEN error LIKE '%429%' THEN '429 Rate Limited'
                ELSE 'Other'
            END
        ORDER BY cnt DESC
        LIMIT 5
    """, (wave_id,))
    error_types = [dict(r) for r in cur.fetchall()]

    conn.close()

    total  = sum(r["total"]  for r in rows)
    ok     = sum(r["ok"]     for r in rows)
    errors = sum(r["errors"] for r in rows)
    cost   = sum(r["cost"]   for r in rows)

    return {
        "ran":         True,
        "date":        date,
        "wave_id":     wave_id,
        "models":      rows,
        "total":       total,
        "ok":          ok,
        "errors":      errors,
        "cost":        round(cost, 4),
        "error_types": error_types,
    }


# ── Email formatting ──────────────────────────────────────────────────────────

def _status_emoji(data: dict, balance: dict | None) -> str:
    low = balance and balance["limit_remaining"] is not None and balance["limit_remaining"] < LOW_BALANCE_THRESHOLD
    if not data["ran"]:
        return "❌"
    if data["errors"] == 0:
        return "⚠️ 💰" if low else "✅"
    if data["ok"] == 0:
        return "❌"
    return "⚠️"


def build_subject(data: dict, balance: dict | None) -> str:
    emoji = _status_emoji(data, balance)
    if not data["ran"]:
        return f"{emoji} LLM Study — no run on {data['date']}"
    pct = round(100 * data["ok"] / data["total"]) if data["total"] else 0
    return f"{emoji} LLM Study {data['date']} — {data['ok']}/{data['total']} OK ({pct}%)"


def build_body(data: dict, balance: dict | None) -> str:
    # ── Balance block ─────────────────────────────────────────────────────────
    if balance and balance["limit_remaining"] is not None:
        remaining = balance["limit_remaining"]
        limit     = balance["limit"] or 0
        used      = balance["usage"] or 0
        monthly   = balance["usage_monthly"] or 0
        low       = remaining < LOW_BALANCE_THRESHOLD
        bal_color = "#c62828" if low else "#2e7d32"
        warning   = (
            f"<p style='background:#fff3e0;border-left:4px solid #e65100;"
            f"padding:10px 14px;margin:16px 0;font-size:14px'>"
            f"⚠️ <b>Low balance:</b> only <b>${remaining:.2f}</b> remaining — "
            f"please top up your OpenRouter account.</p>"
        ) if low else ""
        balance_html = f"""
        <h3>OpenRouter balance</h3>
        {warning}
        <table border='0' cellspacing='0' style='font-size:14px'>
          <tr><td style='padding:3px 12px;color:#666'>Remaining</td>
              <td style='padding:3px 12px'><b style='color:{bal_color}'>${remaining:.2f}</b>
              {'/ $' + str(int(limit)) if limit else ''}</td></tr>
          <tr><td style='padding:3px 12px;color:#666'>Used total</td>
              <td style='padding:3px 12px'>${used:.4f}</td></tr>
          <tr><td style='padding:3px 12px;color:#666'>Used this month</td>
              <td style='padding:3px 12px'>${monthly:.4f}</td></tr>
        </table>
        """
    else:
        balance_html = "<p style='color:#999;font-size:13px'>Balance unavailable (OPENROUTER_API_KEY not set).</p>"

    if not data["ran"]:
        return (
            f"<html><body style='font-family:Arial,sans-serif;color:#333;max-width:680px'>"
            f"<h2>No wave found for {data['date']}</h2>"
            f"<p>The pipeline did not run today, or the wave was not recorded in study.db.</p>"
            f"{balance_html}"
            f"</body></html>"
        )

    rows_html = ""
    for m in data["models"]:
        model  = m["model"] or "unknown"
        pct    = round(100 * m["ok"] / m["total"]) if m["total"] else 0
        color  = "#2e7d32" if m["errors"] == 0 else ("#e65100" if m["ok"] == 0 else "#f57c00")
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 12px'>{model}</td>"
            f"<td style='padding:6px 12px;text-align:center'>{m['total']}</td>"
            f"<td style='padding:6px 12px;text-align:center;color:{color}'><b>{m['ok']}</b></td>"
            f"<td style='padding:6px 12px;text-align:center;color:{'#c62828' if m['errors'] else '#999'}'>{m['errors']}</td>"
            f"<td style='padding:6px 12px;text-align:center'>{pct}%</td>"
            f"<td style='padding:6px 12px;text-align:right'>${m['cost']:.4f}</td>"
            f"</tr>"
        )

    error_html = ""
    if data["error_types"]:
        error_rows = "".join(
            f"<tr><td style='padding:4px 12px'>{e['error'][:80]}</td>"
            f"<td style='padding:4px 12px;text-align:center'>{e['cnt']}</td></tr>"
            for e in data["error_types"]
        )
        error_html = f"""
        <h3 style='color:#c62828'>Error breakdown</h3>
        <table border='0' cellspacing='0' cellpadding='0'
               style='border-collapse:collapse;font-family:monospace;font-size:13px'>
          <tr style='background:#fbe9e7'>
            <th style='padding:4px 12px;text-align:left'>Type</th>
            <th style='padding:4px 12px'>Count</th>
          </tr>
          {error_rows}
        </table>
        """

    overall_color = "#2e7d32" if data["errors"] == 0 else "#c62828"

    return f"""
    <html><body style='font-family:Arial,sans-serif;color:#333;max-width:680px'>
      <h2>LLM Study — Daily Run {data['date']}</h2>

      <p style='font-size:16px'>
        <b style='color:{overall_color}'>{data['ok']} OK</b> &nbsp;|&nbsp;
        <b style='color:{"#c62828" if data["errors"] else "#999"}'>{data['errors']} errors</b>
        &nbsp;|&nbsp; total cost: <b>${data['cost']:.4f}</b>
      </p>

      <table border='0' cellspacing='0' cellpadding='0'
             style='border-collapse:collapse;font-size:14px;width:100%'>
        <tr style='background:#1f4e79;color:white'>
          <th style='padding:8px 12px;text-align:left'>Model</th>
          <th style='padding:8px 12px'>Total</th>
          <th style='padding:8px 12px'>OK</th>
          <th style='padding:8px 12px'>Errors</th>
          <th style='padding:8px 12px'>Success %</th>
          <th style='padding:8px 12px'>Cost</th>
        </tr>
        {rows_html}
      </table>

      {error_html}

      {balance_html}

      <p style='margin-top:24px;font-size:12px;color:#888'>
        Wave ID: {data['wave_id']}<br>
        Sent by notify.py — LLM Dataset Longitudinal Study
      </p>
    </body></html>
    """


# ── Send email ────────────────────────────────────────────────────────────────

def send_email(subject: str, html: str) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASSWORD"]
    to_addr   = os.environ["NOTIFY_EMAIL"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = to_addr
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_addr, msg.as_string())

    print(f"Email sent → {to_addr}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date",   default=datetime.date.today().isoformat())
    p.add_argument("--db",     default="study.db")
    p.add_argument("--dry-run", action="store_true",
                   help="Print report to stdout instead of sending email")
    args = p.parse_args()

    data    = check_run(args.db, args.date)
    balance = fetch_balance()
    subject = build_subject(data, balance)
    body    = build_body(data, balance)

    if args.dry_run:
        print(subject)
        print("─" * 60)
        # Strip HTML tags for terminal output
        import re
        print(re.sub(r"<[^>]+>", "", body).strip())
        return

    send_email(subject, body)

    if not data["ran"] or data["ok"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
