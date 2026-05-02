import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL, APP_PASSWORD, RECIPIENTS

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SUBJECT = "Valence Analytics - Daily AI Insights"


def send_email(content: str) -> bool:
    """Send email to all recipients. Returns True on success, False on failure."""
    print(f"\n Sending email to {len(RECIPIENTS)} recipient(s)...")

    try:
        msg = MIMEMultipart()
        msg["Subject"] = EMAIL_SUBJECT
        msg["From"]    = EMAIL
        msg["To"]      = ", ".join(RECIPIENTS)
        msg.attach(MIMEText(content, "plain", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print(" Email sent successfully")
        return True

    except smtplib.SMTPAuthenticationError:
        print(" Authentication failed — check App Password in .env")
        return False

    except smtplib.SMTPException as e:
        print(f" SMTP error: {e}")
        return False

    except Exception as e:
        print(f" Unexpected error: {type(e).__name__}: {e}")
        return False