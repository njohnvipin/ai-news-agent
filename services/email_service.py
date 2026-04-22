import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL, APP_PASSWORD, RECIPIENTS


def send_email(content):
    print("\n🔧 send_email_tool was called!")   # ← add this
    print(f"Content length: {len(content)} chars")
    print(f"\nDEBUG — From     : {EMAIL}")
    print(f"DEBUG — To       : {RECIPIENTS}")

    try:
        msg = MIMEMultipart()
        msg["Subject"] = "Valence Analytics - Daily AI Insights"
        msg["From"] = EMAIL
        msg["To"] = ", ".join(RECIPIENTS)
        msg.attach(MIMEText(content, "plain", "utf-8"))

        print("DEBUG — Connecting to smtp.gmail.com:587...")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            print("DEBUG — TLS started, logging in...")
            server.login(EMAIL, APP_PASSWORD)
            print("DEBUG — Login successful, sending...")
            server.send_message(msg)

        print("Email sent successfully")

    except smtplib.SMTPAuthenticationError:
        print(" AUTH FAILED — Wrong App Password")

    except smtplib.SMTPException as e:
        print(f"SMTP ERROR: {e}")

    except Exception as e:
        print(f" ERROR: {type(e).__name__}: {e}")