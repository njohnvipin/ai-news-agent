import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL, APP_PASSWORD, RECIPIENTS

def send_email(content):
    try:
        msg = MIMEMultipart()
        msg["Subject"] = "Valence Analytics - Daily AI Insights"
        msg["From"] = EMAIL
        msg["To"] = ", ".join(RECIPIENTS)
        msg.attach(MIMEText(content, "plain", "utf-8"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print("Email sent successfully")

    except Exception as e:
        print("Email error:", e)