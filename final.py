import re
from dateparser import parse
from datetime import datetime, timedelta
import torch
from conll_2003 import EnhancedBiLSTM_CRF, config, CONLL2003Dataset
from typing import Dict, List, Optional
from transformers import pipeline

dataset = CONLL2003Dataset("data/conll2003/train.csv") 
word_vocab = dataset.word_vocab
char_vocab = dataset.char_vocab
label_vocab = dataset.label_vocab  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loc_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
event_extractor = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

def clean_event_name(raw_name: str) -> str:
    if re.match(r'^[A-Z]{1,3}\d{1,4}$', raw_name):
        return "Unnamed Event"
    
    location_indicators = {
        'room', 'hall', 'building', 'center', 'centre', 'lab',
        'theater', 'theatre', 'auditorium', 'campus', 'floor', 'mess'
    }
    lower_name = raw_name.lower()
    if any(indicator in lower_name for indicator in location_indicators):
        return "Unnamed Event"
    
    raw_words = raw_name.strip().split()
    has_capitalized = any(w[0].isupper() for w in raw_words if len(w) > 1)
    
    if len(raw_words) > 1 and has_capitalized:
        return raw_name.strip()
    
    clean = re.split(r'(?:\bon\b|\bat\b|\sfrom\s|,|;|-)', raw_name, maxsplit=1)[0].strip()
    clean = re.sub(r'\s*(?:to|will|the|a|for|by|of|on|at|p\.?s\.?)\s*$', '', clean, flags=re.I)
    
    cleaned_words = clean.split()
    if len(cleaned_words) > 1 and any(w[0].isupper() for w in cleaned_words):
        return clean
    
    return "Unnamed Event"


def extract_cultural_event(text: str) -> Optional[str]:
    patterns = [
        r'(?:celebrat|observ|invit|join).*?\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Puja|Pooja|Celebration|Festival|Utsav))',
        r'\b([A-Z][a-z]+\s+(?:Puja|Pooja|Utsav))\b',
        r'\b(?:Special\s+)?([A-Z][a-z]+\s+Celebration)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_event_name(match.group(1))
    return None

def transformer_extract_event(email_body: str) -> str:
    cultural_event = extract_cultural_event(email_body)
    if cultural_event:
        return cultural_event
        
    prompt = f"""Extract ONLY the full official event name from this email. 
    Return the COMPLETE name, even if it contains technical terms or is long.
    Preserve capitalization and exact wording
    Email: "{email_body[:1000]}"
    Event name: """
    
    try:
        result = event_extractor(
            prompt,
            max_length=50,
            num_beams=3,
            do_sample=False,
            early_stopping=True
        )[0]['generated_text']
        return clean_event_name(result)
    except:
        return "Unnamed Event"

def regex_extract_event(text: str) -> str:
    if cultural_event := extract_cultural_event(text):
        return cultural_event
    
    screening_match = re.search(
        r'screening\s+(?:of\s+)?([^.!?]+?(?:episodes?|movies?|films?)[^.!?]+)',
        text, 
        re.IGNORECASE
    )
    if screening_match:
        return clean_event_name(screening_match.group(1))
    
    venue_phrases = re.findall(r'(?:Where|Venue|Location)\s*[:\?]\s*([^\n]+)', text, re.I)
    venue_words = {word for phrase in venue_phrases for word in phrase.strip().split()}
    
    location_indicators = {
        'room', 'hall', 'building', 'center', 'centre', 'lab', 
        'theater', 'theatre', 'auditorium', 'campus', 'floor'
    }
    venue_words.update(location_indicators)
    
    pattern = r'''
        (?:^|\s)  
        (
            (?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)  
            |
            (?:[A-Z][A-Za-z0-9]+\s+[A-Z][A-Za-z0-9]+) 
        )
        (?=\s|$|[.,;:!?)])  
    '''
    
    candidates = [
        match.group(1).strip() 
        for match in re.finditer(pattern, text, re.VERBOSE)
        if not any(
            word.lower() in match.group(1).lower() 
            for word in venue_words
        )
    ]
    
    if candidates:
        longest_candidate = max(candidates, key=len)
        return re.sub(r'\s*[.,;:!?)]*$', '', longest_candidate).strip()
    
    return "Unnamed Event"


def extract_event_details(email_body: str) -> Dict[str, str]:
    clean_text = ' '.join(email_body.split()[:500])
    
    cultural_event = extract_cultural_event(clean_text)
    if cultural_event:
        return {
            'event_name': cultural_event
        }
    
    venue_names = set()
    venue = extract_venue(clean_text)
    if venue:
        venue_names.update(venue.split())
    
    event_name = transformer_extract_event(clean_text)
    
    if (event_name == "Unnamed Event" or 
        any(venue_word.lower() in event_name.lower() for venue_word in venue_names)):
        event_name = regex_extract_event(clean_text)
    
    if any(venue_word.lower() in event_name.lower() for venue_word in venue_names):
        event_name = "Unnamed Event"
    
    return {
        'event_name': event_name
    }


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EnhancedBiLSTM_CRF(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        label_vocab=label_vocab
    ).to(device)
    
    model_state = checkpoint['model_state_dict']
    current_state = model.state_dict()
    
    for name, param in model_state.items():
        if name in current_state and param.size() == current_state[name].size():
            current_state[name].copy_(param)
    
    model.eval()
    return model

def preprocess_text(text):
    tokens = []
    temp = []
    for word in text.split():
        if word.isupper() and len(word) > 3:
            temp.append(word)
        else:
            if temp:
                tokens.append(" ".join(temp))
                temp = []
            tokens.append(word)
    if temp:
        tokens.append(" ".join(temp))
    
    word_ids = [word_vocab.get(token.lower(), 1) for token in tokens]
    
    char_ids = []
    for token in tokens:
        chars = [char_vocab.get(c.lower(), 1) for c in token[:config['max_char_len']]]
        chars += [0] * (config['max_char_len'] - len(chars))
        char_ids.append(chars)
    
    word_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
    char_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
    mask = torch.ones(1, len(tokens), dtype=torch.bool).to(device)
    
    return word_tensor, char_tensor, mask, tokens

def remove_event_name(text: str, event_name: str) -> str:
    if not event_name or event_name == "Unnamed Event":
        return text
    
    pattern = re.compile(re.escape(event_name), re.IGNORECASE)
    return pattern.sub("", text)

def predict(text: str, model, current_event_name: str = None) -> List[str]:
    text = re.sub(r'^\s*(Dear|Hello|Hi)\s+[^,\n]+,?\s*', '', text, flags=re.IGNORECASE)
    
    if current_event_name:
        text = remove_event_name(text, current_event_name)
    
    speaker_patterns = [
        r'(?:Speaker|Presented by|By|Talk by|Keynote by)\s*[:\-]\s*((?:Prof\.|Dr\.)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'(?:Speaker|Presented by|By|Talk by|Keynote by)\s*[:\-]\s*([A-Z]+(?:\s+[A-Z]+)*)',
        r'\b(?:by|from)\s+((?:Prof\.|Dr\.)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    ]
    
    for pattern in speaker_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            speaker = match.group(1).strip()
            if is_proper_name(speaker):
                return [speaker]
    
    signature_match = re.search(
        r'(?:Regards|Thanks|From)[,\s]+([A-Z][a-zA-Z\s]+Team\b|[A-Z][a-zA-Z\s]+Club\b)',
        text,
        re.IGNORECASE
    )
    if signature_match:
        return []

    word_tensor, char_tensor, mask, tokens = preprocess_text(text)
    if not tokens:
        return []
    
    with torch.no_grad():
        tag_ids = model(word_tensor, char_tensor, mask=mask)[0]
    
    reverse_label_vocab = {v: k for k, v in label_vocab.items()}
    predictions = [(token, reverse_label_vocab.get(tag, "O")) 
                 for token, tag in zip(tokens, tag_ids)]

    person_names = []
    current_name = []
    
    for token, tag in predictions:
        if tag == "B-PER":
            if current_name and is_proper_name(' '.join(current_name)):
                person_names.append(' '.join(current_name))
            current_name = [token]
        elif tag == "I-PER" and current_name:
            current_name.append(token)
        else:
            if current_name and is_proper_name(' '.join(current_name)):
                person_names.append(' '.join(current_name))
            current_name = []
    
    if current_name and is_proper_name(' '.join(current_name)):
        person_names.append(' '.join(current_name))
    
    return person_names

def is_proper_name(text: str) -> bool:
    if len(text) < 2:
        return False
    
    excluded_terms = {
        'uncover', 'join', 'attend', 'learn', 'discover', 'explore',
        'register', 'participate', 'submit', 'pumped', 'hey', 'there',
        'welcome', 'hello', 'hi', 'dear', 'thanks', 'regards'
    }
    if any(term in text.lower() for term in excluded_terms):
        return False
    
    words = text.split()
    if len(words) > 1 and not any(w[0].isupper() for w in words[1:] if w):
        return False
    
    generic_terms = {
        'exam', 'exams', 'test', 'tests', 'meeting', 'event',
        'announcement', 'notification', 'deadline', 'assignment',
        'team', 'club', 'committee', 'department'
    }
    if any(word.lower() in generic_terms for word in words):
        return False
    
    titles = {'Dr.', 'Prof.', 'Mr.', 'Mrs.', 'Ms.', 'PhD', 'M.Tech', 'B.Tech'}
    if any(word in titles for word in words):
        return len(words) > 1
    
    if text.isupper():
        return len(text) <= 3 and '.' not in text
    
    return text[0].isupper()

def extract_time(text):
    time_match = re.search(r'(?:Time|Timing)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if time_match:
        time_str = time_match.group(1).strip()
        first_time = re.search(r'\b(?:[01]?\d|2[0-3])(?:[.:][0-5]\d)?(?:\s?[APap][Mm])?\b', time_str)
        if first_time:
            return first_time.group(0).strip()
    
    time_pattern = r'''
        \b(?:[01]?\d|2[0-3])       
        [.:]                        
        [0-5]\d                   
        (?:\s?[APap][Mm])?         
        \b
        |
        \b(?:[1-9]|1[0-2])\s?[APap][Mm]\b  
    '''

    try:
        matches = re.findall(time_pattern, text, re.VERBOSE)
        if not matches:
            return None
        
        first_time = matches[0]
        return first_time.replace('.', ':').replace(' ', '').upper()
    
    except Exception as e:
        return None

def get_day_suffix(day: int) -> str:
    if 10 <= day <= 20:
        return 'th'  
    else:
        suffix_map = {1: 'st', 2: 'nd', 3: 'rd'}
        return suffix_map.get(day % 10, 'th')

def get_day_suffix(day: int) -> str:
    if 11 <= day <= 13:
        return 'th'
    else:
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

def extract_date(text: str) -> str | None:
    month_match = re.search(
        r'(\d{1,2}(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|'
        r'August|September|October|November|December)\s+\d{4})',
        text, 
        re.IGNORECASE
    )
    if month_match:
        full_date = month_match.group(1)
        day = re.sub(r'\D', '', full_date.split()[0])
        month = month_match.group(2)
        year = re.search(r'\d{4}', full_date).group()
        suffix = get_day_suffix(int(day))
        return f"{day}{suffix} {month.capitalize()} {year}"

    when_match = re.search(r'(?:When|Date)\s*[:\?]\s*([^\n]+)', text, re.IGNORECASE)
    if when_match:
        when_text = when_match.group(1).strip()
        parsed_date = parse(when_text, settings={'PREFER_DATES_FROM': 'future'})
        if parsed_date:
            day = parsed_date.day
            suffix = get_day_suffix(day)
            return parsed_date.strftime(f"%d{suffix} %B %Y")

    time_phrase_match = re.search(
        r'\b(this evening|tonight|this morning)\b', 
        text, 
        re.IGNORECASE
    )
    if time_phrase_match:
        today = datetime.today()
        day = today.day
        suffix = get_day_suffix(day)
        return today.strftime(f"%d{suffix} %B %Y")

    if re.search(r'\btoday\b', text, re.IGNORECASE):
        today = datetime.today()
        day = today.day
        suffix = get_day_suffix(day)
        return today.strftime(f"%d{suffix} %B %Y")

    if re.search(r'\btomorrow\b', text, re.IGNORECASE):
        tomorrow = datetime.today() + timedelta(days=1)
        day = tomorrow.day
        suffix = get_day_suffix(day)
        return tomorrow.strftime(f"%d{suffix} %B %Y")

    date_pattern = r"""
        \b(?P<dmy>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b|
        \b(?P<ymd>\d{4}-\d{1,2}-\d{1,2})\b|
        \b(?P<word>\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})\b|
        \b(?P<word_comma>[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4})\b
    """
    match = re.search(date_pattern, text, re.VERBOSE | re.IGNORECASE)
    if match:
        matched_date = next(g for g in match.groups() if g)
        matched_date = re.sub(r'(\d)(st|nd|rd|th)\b', r'\1', matched_date)
        parsed_date = parse(matched_date, settings={'PREFER_DATES_FROM': 'future'})
        if parsed_date:
            day = parsed_date.day
            suffix = get_day_suffix(day)
            return parsed_date.strftime(f"%d{suffix} %B %Y")

    return None

def extract_topic(text):
    topic_match = re.search(r'(?:Topic|Title|Subject)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if topic_match:
        return topic_match.group(1).strip()
    return None

def extract_venue(text):
    room_match = re.search(
        r'\b([A-Z]{1,5}\d{1,5}[A-Z]?)\b',  
        text
    )
    if room_match:
        return room_match.group(1)
    
    area_match = re.search(
        r'\b(?:in|at)\s+(?:the\s+)?([A-Z][a-zA-Z0-9\s]+?(?:area|room|hall|theater|lab))\b',
        text,
        re.IGNORECASE
    )
    if area_match:
        raw_venue = area_match.group(1).strip()
        clean_venue = re.sub(r'[^\w\s]', '', raw_venue).strip()
        
        if re.search(r'\bSc\s*ops\b', clean_venue, re.IGNORECASE):
            clean_venue = 'Sccoops area'
        return clean_venue
    
    venue_match = re.search(
        r'(?:Venue|Location|Place)\s*:\s*([^\n]+)',
        text,
        re.IGNORECASE
    )
    if venue_match:
        raw_venue = venue_match.group(1).strip()
        clean_venue = re.sub(r'[^\w\s]', '', raw_venue).strip()
        
        if re.search(r'\bSc\s*ops\b', clean_venue, re.IGNORECASE):
            clean_venue = 'Sccoops area'
        return clean_venue
    
    entities = loc_model(text)
    location = []
    
    for entity in entities:
        if entity['entity'] in ['B-LOC', 'I-LOC']:
            clean_word = re.sub(r'[^\w\s]', '', entity['word']).strip()
            if clean_word:
                if re.search(r'\bSc\s*ops\b', clean_word, re.IGNORECASE):
                    clean_word = 'Sccoops'
                location.append(clean_word)
    
    if location:
        venue = ' '.join(location)
        if re.search(r'\bSc\s*ops\b', venue, re.IGNORECASE):
            venue = 'Sccoops area'
        return venue
    
    return None

def extract_links(text: str) -> Dict[str, List[str]]:
    closing_phrases = r"(regards|cheers|thank you|thanks|sincerely|best wishes)[\s\S]*$"
    truncated_text = re.sub(closing_phrases, "", text, flags=re.IGNORECASE)

    url_pattern = r'(?<!@)(?:https?://|www\.)[^\s<>"\']+'

    context_patterns = {
        "Registration": r'(?:register|sign\s*up|registration|rsvp|join|participate|last\s*day\s*to\s*register)\b[\s\S]*?(?:here|link|at)?\s*[:=]?\s*({})'.format(url_pattern),
        "Photo Album": r'(?:album|photos?|pictures?|gallery)\b[\s\S]*?(?:here|link)?\s*[:=]?\s*({})'.format(url_pattern),
        "Social Media": r'(?:\binstagram\b|\bfb\b|\bfacebook\b|\btwitter\b|\bx\b|\blinkedin\b|\bsocial\s*media\b)\b[\s\S]*?(?:here|profile|page)?\s*[:=]?\s*({})'.format(url_pattern),
        "Contact": r'(?:\bcontact\b|\breach\s*out\b|\bwhatsapp\b|\bcall\b|\bphone\b|\bmobile\b|\bnumber\b)\b[\s\S]*?(?:here|us|at)?\s*[:=]?\s*({})'.format(url_pattern),
    }

    extracted_links = {key: [] for key in context_patterns.keys()}
    matched_urls = set()

    for link_type, pattern in context_patterns.items():
        matches = re.findall(pattern, truncated_text, re.IGNORECASE)
        for url in matches:
            if url not in matched_urls:
                extracted_links[link_type].append(url)
                matched_urls.add(url)

    all_urls = re.findall(url_pattern, truncated_text, re.IGNORECASE)
    other_urls = [url for url in all_urls if url not in matched_urls]

    if other_urls:
        extracted_links["Other Links"] = other_urls

    return {k: v for k, v in extracted_links.items() if v}

def similar_text(text1: str, text2: str) -> bool:
    return text1.lower() in text2.lower() or text2.lower() in text1.lower()