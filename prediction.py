import torch
from final import extract_event_details, extract_date, extract_topic, extract_time, extract_venue, predict, _extract_registration_links, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
text = """Dear All,

As exams approach, we understand the pressure of lab exams, quizzes, vivas, and tests. To help you take a well-deserved break and recharge, we are excited to announce a Football Tournament for both men and women, set to take place at our football ground in the coming days.

This tournament is a fantastic opportunity to unwind, engage in friendly competition, and enjoy the game while taking a short break from your academic commitments.

If you are interested in participating, please register using the appropriate link below:

Men’s Registration: https://docs.google.com/forms/d/14tb-RWdol1CP6dtHgcN1av-zG_TvRuky11blc-gtDGM/edit

Women’s Registration: https://docs.google.com/forms/d/1EDrykPZK6vcsZM2zWPL_-imU1p-8iWPnwRI7dD-KfgQ/edit

Don't miss out on this exciting event! Register now, take a break, and make unforgettable memories on the field.

Play hard, have fun, and kick away the exam stress!
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