"""
email_service.py — Sends the digest email through Gmail SMTP.
"""
import logging
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Optional

from config import EMAIL, APP_PASSWORD, RECIPIENTS, missing_email_settings

logger = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_TIMEOUT = 30  # seconds
EMAIL_SUBJECT = "Valence Analytics - Daily AI Insights"


def build_message(content: str, html: Optional[str] = None):
    """
    Plain text only, or plain text + HTML as "multipart/alternative".

    With both parts, email clients show the HTML version and fall back to the
    plain text if they can't display HTML (or the reader prefers text).
    The HTML part must be attached LAST: clients pick the last part they support.
    """
    if not html:
        return MIMEText(content, "plain", "utf-8")
    msg = MIMEMultipart("alternative")
    msg.attach(MIMEText(content, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def send_email(content: str, html: Optional[str] = None) -> bool:
    """
    Send the email to all recipients. Returns True on success, False on failure.

    content: the plain-text version (always sent)
    html:    optional designed HTML version, shown by most email clients

    Recipients are sent as BCC: the visible "To" is the sender's own address,
    so recipients can't see each other's emails.
    """
    missing = missing_email_settings()
    if missing:
        logger.error("Cannot send email: missing in .env: %s", ", ".join(missing))
        return False

    msg = build_message(content, html)
    msg["Subject"] = f"Daily AI Insights · {date.today():%a, %b} {date.today().day}"
    msg["From"] = formataddr(("Valence Analytics", EMAIL))
    msg["To"] = EMAIL

    logger.info("Sending email to %d recipient(s)...", len(RECIPIENTS))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
            server.starttls()
            server.login(EMAIL, APP_PASSWORD)
            # to_addrs sets the real (envelope) recipients = BCC
            server.send_message(msg, to_addrs=RECIPIENTS)
    except smtplib.SMTPAuthenticationError:
        logger.error("Gmail authentication failed — check APP_PASSWORD in .env")
        return False
    except (smtplib.SMTPException, OSError) as e:
        logger.error("Sending email failed: %s: %s", type(e).__name__, e)
        return False

    logger.info("Email sent successfully")
    return True