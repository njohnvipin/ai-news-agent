"""
email_template.py — Builds the HTML version of the daily digest email.

Email clients (Gmail, Outlook, Apple Mail) support only a small, old subset of
HTML and CSS, so this template deliberately uses:
  - tables for layout (flexbox/grid are ignored by many clients)
  - inline styles on every element (some clients strip <style> blocks)
  - web-safe fonts and a fixed 600px width, which is the email standard
Every piece of text from the news or the LLM is HTML-escaped, and only
http(s) links are allowed, so headlines can't inject markup or scripts.
"""
import re
from datetime import date
from email.utils import parsedate_to_datetime
from html import escape
from typing import List, Optional

# Colours shared with the presentation deck
INK = "#1B1F24"
SLATE = "#3A4654"
MUTED = "#8A96A3"
CARD = "#F1F3F5"
ORANGE = "#E4572E"
TEAL = "#2A9D8F"
TINT = "#E3F4F1"
FONT = "Arial, Helvetica, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"


# ── small helpers ────────────────────────────────────────────────────────────

def safe_url(url: str) -> Optional[str]:
    """Only allow http(s) links (no javascript:, data:, etc.)."""
    url = (url or "").strip()
    return url if re.match(r"^https?://", url, re.IGNORECASE) else None


def split_headline(title: str) -> tuple:
    """Google News titles end with ' - Publisher'. Split them: ('Headline', 'Publisher')."""
    head, sep, publisher = (title or "").rpartition(" - ")
    if sep and head and 0 < len(publisher) <= 60:
        return head.strip(), publisher.strip()
    return (title or "").strip(), ""


def short_date(published: str) -> str:
    """'Sun, 25 Oct 2026 04:00:00 GMT' -> 'Oct 25'. Falls back to the original text."""
    try:
        when = parsedate_to_datetime(published)
        return f"{when:%b} {when.day}"
    except (TypeError, ValueError, IndexError):
        return published or ""


def split_sentences(text: str) -> List[str]:
    """Split the briefing into sentences so each can be shown as a numbered point."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


# ── sections ─────────────────────────────────────────────────────────────────

def _header(today: date) -> str:
    return f"""
<tr><td style="background:{INK};padding:32px 36px 28px 36px;">
  <div style="font-family:{FONT};font-size:12px;letter-spacing:2px;color:{ORANGE};font-weight:bold;text-transform:uppercase;">Valence Analytics</div>
  <div style="font-family:{SERIF};font-size:30px;line-height:36px;color:#FFFFFF;font-weight:bold;padding-top:6px;">Daily AI Insights</div>
  <div style="font-family:{FONT};font-size:14px;color:{MUTED};padding-top:6px;">{today:%A, %B} {today.day}, {today:%Y}</div>
</td></tr>
<tr><td style="background:{ORANGE};height:4px;line-height:4px;font-size:0;">&nbsp;</td></tr>"""


def _insights(insights: str) -> str:
    sentences = split_sentences(insights)
    rows = "".join(f"""
    <tr>
      <td valign="top" style="padding:0 14px 14px 0;width:28px;">
        <div style="width:26px;height:26px;line-height:26px;border-radius:13px;background:{TEAL};color:#FFFFFF;font-family:{FONT};font-size:13px;font-weight:bold;text-align:center;">{i}</div>
      </td>
      <td valign="top" style="padding:3px 0 14px 0;font-family:{FONT};font-size:15px;line-height:23px;color:{INK};">{escape(s)}</td>
    </tr>""" for i, s in enumerate(sentences, 1))
    return f"""
<tr><td style="padding:30px 36px 8px 36px;">
  <div style="font-family:{FONT};font-size:12px;letter-spacing:2px;color:{TEAL};font-weight:bold;text-transform:uppercase;padding-bottom:6px;">Today's briefing</div>
  <div style="font-family:{SERIF};font-size:22px;line-height:28px;color:{INK};font-weight:bold;padding-bottom:18px;">What matters in AI today</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;">{rows}</table>
</td></tr>"""


def _stories(articles: List[dict]) -> str:
    cards = []
    for i, article in enumerate(articles, 1):
        headline, publisher = split_headline(article.get("title", ""))
        url = safe_url(article.get("link", ""))
        meta = " · ".join(x for x in (publisher, short_date(article.get("published", ""))) if x)
        title_html = escape(headline)
        if url:
            title_html = (f'<a href="{escape(url, quote=True)}" style="color:{INK};text-decoration:none;" '
                          f'target="_blank">{title_html}</a>')
        read_more = (f'<a href="{escape(url, quote=True)}" style="font-family:{FONT};font-size:13px;color:{ORANGE};'
                     f'font-weight:bold;text-decoration:none;" target="_blank">Read article &rarr;</a>') if url else ""
        cards.append(f"""
    <tr><td style="padding:0 0 12px 0;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;background:{CARD};border-radius:8px;">
        <tr>
          <td valign="top" style="padding:16px 0 16px 18px;width:30px;font-family:{SERIF};font-size:20px;font-weight:bold;color:{ORANGE};">{i:02d}</td>
          <td style="padding:16px 18px 16px 10px;">
            <div style="font-family:{FONT};font-size:12px;color:{SLATE};text-transform:uppercase;letter-spacing:1px;">{escape(meta)}</div>
            <div style="font-family:{FONT};font-size:16px;line-height:22px;font-weight:bold;color:{INK};padding:4px 0 8px 0;">{title_html}</div>
            {read_more}
          </td>
        </tr>
      </table>
    </td></tr>""")
    return f"""
<tr><td style="padding:22px 36px 18px 36px;">
  <div style="font-family:{FONT};font-size:12px;letter-spacing:2px;color:{ORANGE};font-weight:bold;text-transform:uppercase;padding-bottom:6px;">Top stories</div>
  <div style="font-family:{SERIF};font-size:22px;line-height:28px;color:{INK};font-weight:bold;padding-bottom:16px;">{len(articles)} headlines behind this briefing</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;">{''.join(cards)}</table>
</td></tr>"""


def _footer(disclosure: str) -> str:
    return f"""
<tr><td style="padding:6px 36px 30px 36px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;background:{TINT};border-radius:8px;">
    <tr><td style="padding:14px 18px;font-family:{FONT};font-size:13px;line-height:19px;color:{SLATE};">
      <strong style="color:{INK};">AI-generated, human-reviewed.</strong> {escape(disclosure)}
    </td></tr>
  </table>
</td></tr>
<tr><td style="background:{INK};padding:20px 36px;font-family:{FONT};font-size:13px;line-height:19px;color:{MUTED};">
  Best regards,<br><span style="color:#FFFFFF;font-weight:bold;">Valence Analytics</span>
</td></tr>"""


# ── public API ───────────────────────────────────────────────────────────────

def build_email_html(articles: List[dict], insights: str, disclosure: str,
                     today: Optional[date] = None) -> str:
    """Return the complete HTML email."""
    today = today or date.today()
    sentences = split_sentences(insights)
    preheader = escape(sentences[0] if sentences else "Today's AI news briefing")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Daily AI Insights</title>
</head>
<body style="margin:0;padding:0;background:#E9ECEF;">
<!-- Preheader: the preview line shown next to the subject in the inbox -->
<div style="display:none;max-height:0;overflow:hidden;opacity:0;">{preheader}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#E9ECEF;">
  <tr><td align="center" style="padding:24px 12px;">
    <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0"
           style="width:100%;max-width:600px;background:#FFFFFF;border-radius:10px;overflow:hidden;">
      {_header(today)}
      {_insights(insights)}
      {_stories(articles)}
      {_footer(disclosure)}
    </table>
  </td></tr>
</table>
</body>
</html>"""