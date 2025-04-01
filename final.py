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
    
    if len(raw_name.split()) > 2 and any(w.isupper() for w in raw_name.split()):
        return raw_name.strip()
    
    clean = re.split(r'(?:\son\s|\sat\s|\sfrom\s|,|;|-)', raw_name)[0]
    clean = re.sub(r'\s*(?:to|will|the|a|for|by|of|on|at|p\.?s\.?)\s*$', '', clean, flags=re.I)
    return clean.strip() if len(clean.split()) > 1 else "Unnamed Event"

def classify_event_type(event_name: str) -> str:
    if not event_name or event_name == "Unnamed Event":
        return "unknown"
    
    lower_name = event_name.lower()
    
    if any(word in lower_name for word in ['puja', 'pooja', 'utsav', 'festival', 'celebration']):
        return "cultural"
    if any(word in lower_name for word in ['tournament', 'cup', 'match', 'game']):
        return "sports"
    if any(word in lower_name for word in ['conference', 'seminar', 'workshop']):
        return "academic"
    if any(word in lower_name for word in ['meeting', 'briefing', 'convention']):
        return "business"
    return "other"

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
        if not any(word in match.group(1) for word in venue_words)
    ]
    
    if candidates:
        longest_candidate = max(candidates, key=len)
        return re.sub(r'\s*[.,;:!?)]*$', '', longest_candidate).strip()
    
    return "Unnamed Event"


def extract_event_details(email_body: str) -> Dict[str, str]:
    clean_text = ' '.join(email_body.split()[:200])
    
    cultural_event = extract_cultural_event(clean_text)
    if cultural_event:
        return {
            'event_name': cultural_event,
            'event_type': 'cultural'
        }
    
    event_name = transformer_extract_event(clean_text)
    if event_name == "Unnamed Event":
        event_name = regex_extract_event(clean_text)
    
    return {
        'event_name': event_name,
        'event_type': classify_event_type(event_name)
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
    
    speaker_match = re.search(r'(?:Speaker|Presented by|By)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
        if is_proper_name(speaker):
            return [speaker]
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
    
    words = text.split()
    if not all(word[0].isupper() for word in words if word):
        return False
    
    generic_terms = {
        'exam', 'exams', 'test', 'tests', 'meeting', 'event',
        'announcement', 'notification', 'deadline', 'assignment'
    }
    if any(word.lower() in generic_terms for word in words):
        return False
    
    titles = {'Dr.', 'Prof.', 'Mr.', 'Mrs.', 'Ms.', 'PhD'}
    if any(word in titles for word in words):
        return len(words) > 1  
    
    if text.isupper():
        return len(text) <= 3 and '.' not in text
    
    return True

def extract_time_range(text):
    time_match = re.search(r'(?:Time|Timing)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if time_match:
        return time_match.group(1).strip()
    
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
        
        normalized_matches = list({m.replace('.', ':').replace(' ', '').upper() for m in matches})
        normalized_matches.sort()

        explicit_time_match = re.search(r'(?:time|timing)\s*:\s*([0-9]{1,2}[.:][0-9]{2}\s*[APap][Mm]?)', text, re.IGNORECASE)
        
        if explicit_time_match:
            explicit_time = explicit_time_match.group(1).replace('.', ':').replace(' ', '').upper()
            return f"{explicit_time} - {normalized_matches[0]}" if len(normalized_matches) > 1 else explicit_time

        return " - ".join(normalized_matches[:2]) if len(normalized_matches) >= 2 else normalized_matches[0]
    
    except Exception as e:
        return f"Error extracting time: {str(e)}"


def get_day_suffix(day: int) -> str:
    if 10 <= day <= 20:
        return 'th'  
    else:
        suffix_map = {1: 'st', 2: 'nd', 3: 'rd'}
        return suffix_map.get(day % 10, 'th')

def extract_date(text: str):
    date_match = re.search(r'(?:Date)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if date_match:
        return date_match.group(1).strip()
    date_pattern = r"""
        \b(?P<dmy>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b |          
        \b(?P<ymd>\d{4}-\d{1,2}-\d{1,2})\b |                
        \b(?P<word>\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})\b | 
        \b(?P<word_comma>[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4})\b  
    """

    match = re.search(date_pattern, text, re.VERBOSE | re.IGNORECASE)
    if match:
        matched_date = match.group(0)
        
        matched_date = re.sub(r'(\d)(st|nd|rd|th)\b', r'\1', matched_date)
        
        try:
            parsed_date = parse(matched_date, settings={'PREFER_DATES_FROM': 'future'})
            if parsed_date:
                day = parsed_date.day
                suffix = get_day_suffix(day)
                return parsed_date.strftime(f"%d{suffix} %B %Y")
        except:
            return matched_date  

    if re.search(r'\btoday\b', text, re.IGNORECASE):
        return datetime.today().strftime("%d/%m/%Y")
    
    if re.search(r'\btomorrow\b', text, re.IGNORECASE):
        return (datetime.today() + timedelta(days=1)).strftime("%d/%m/%Y")

    return None

def extract_topic(text):
    topic_match = re.search(r'(?:Topic|Title|Subject)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if topic_match:
        return topic_match.group(1).strip()
    return None

def extract_venue(text):
    venue_match = re.search(
        r'(?:in|at|venue)\s+(?:the\s+)?([A-Z][a-zA-Z0-9\s]+?(?:area|room|hall|theater|lab))', 
        text, 
        re.IGNORECASE
    )
    if venue_match:
        return venue_match.group(1).strip()
    
    venue_match = re.search(r'(?:Venue|Location|Place)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if venue_match:
        return venue_match.group(1).strip()
    
    code_match = re.search(r'\b[A-Za-z]{2}\d{3}\b', text)
    if code_match:
        return code_match.group(0)
    
    entities = loc_model(text)
    location = []
    
    for entity in entities:
        if entity['entity'] in ['B-LOC', 'I-LOC']:
            location.append(entity['word'])
    
    if location:
        return ' '.join(location)
    
    return None

def _extract_registration_links(text: str) -> Dict[str, str]:
    matches = re.findall(
        r'([^\n:]+Registration)\s*[:\-]\s*((?:https?://|www\.)[^\s<>"\']+)', 
        text, 
        re.IGNORECASE
    )

    if not matches:
        matches = re.findall(
            r'([^\n:]+Registration)\s*[:\-]\s*\n\s*((?:https?://|www\.)[^\s<>"\']+)', 
            text, 
            re.IGNORECASE
        )

    clean_matches = []
    for label, url in matches:
        url = re.sub(r'[.,;:!?\)\]\}\>]+$', '', url)
        clean_matches.append((label.strip(), url))
    
    return dict(clean_matches)

def similar_text(text1: str, text2: str) -> bool:
    return text1.lower() in text2.lower() or text2.lower() in text1.lower()
