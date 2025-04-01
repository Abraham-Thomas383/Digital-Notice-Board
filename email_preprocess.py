import re
import html
import base64
from bs4 import BeautifulSoup

def clean_email_body(body):
    try:
        body = base64.urlsafe_b64decode(body).decode("utf-8")
    except Exception:
        pass  

    body = html.unescape(body)
    body = BeautifulSoup(body, "html.parser").get_text(separator=" ")
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"[^a-zA-Z0-9@,.:/\-\s]", "", body)
    
    body = remove_email_signature(body)
    body = extract_latest_message(body)
    
    return body

def remove_email_signature(body):
    signature_patterns = [r"--\s.*", r"Sent from my .*", r"Best,\s*\n.*"]
    for pattern in signature_patterns:
        body = re.sub(pattern, "", body, flags=re.DOTALL)
    return body.strip()

def extract_latest_message(body):
    splitters = [r"On .* wrote:", r"From: .*", r"Sent: .*", r"To: .*"]
    for splitter in splitters:
        body = re.split(splitter, body, maxsplit=1)[0]
    return body.strip()