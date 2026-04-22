import re

def clean_text(text):
    if text:
        return re.sub('<.*?>', '', text)
    return ""

