import torch
from final import extract_event_details, extract_date, extract_topic, extract_time, extract_venue, predict, _extract_registration_links, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
text = """
We are excited to announce a special screening of "Salaar" this evening at 6:00 PM, right in front of Sccoops!
Date: Today
Time: 6:00 PM
"""
    
model = load_model("AT_2005/best_model.pt")
event_details = extract_event_details(text)
    
print("Event Details:", event_details)
print("Topic:", extract_topic(text))
print("Time:", extract_time(text))
print("Date:", extract_date(text))
print("Venue:", extract_venue(text))
print("Speaker:", predict(text, model, event_details['event_name']))
print("Links:", _extract_registration_links(text))