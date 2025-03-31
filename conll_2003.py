import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from seqeval.metrics import f1_score, classification_report
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import numpy as np
import ast
from torch.amp import autocast
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

config = {
    'seed': 42,
    'batch_size': 16,
    'max_seq_len': 128,
    'max_char_len': 25,
    'word_embed_dim': 300,
    'char_embed_dim': 100,
    'hidden_dim': 384,
    'char_hidden_dim': 100,
    'dropout': 0.3,
    'variational_dropout': False,
    'lstm_layers': 3,
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 50,
    'patience': 5,
    'min_improvement': 0.001,
    'grad_clip': 5.0,
    'model_save_dir': "AT_2005",
    'use_amp': True,
    'glove_path': 'glove.6B.300d.txt',
    'attention_heads': 4,
    'attention_dim': 384,
    'transformer_layers': 1,
    'second_order_crf': False
}
config['model_save_path'] = os.path.join(config['model_save_dir'], "best_model.pt")

os.makedirs(config['model_save_dir'], exist_ok=True)

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['seed'])

class CONLL2003Dataset(Dataset):
    def __init__(self, file_path, word_vocab=None, char_vocab=None, label_vocab=None):
        self.df = pd.read_csv(file_path)
        
        self.sentences = []
        self.labels = []
        for _, row in self.df.iterrows():
            try:
                sent = ast.literal_eval(row['Sentence'])
                lbls = ast.literal_eval(row['Labels'])
                if len(sent) == len(lbls):
                    self.sentences.append(sent)
                    self.labels.append(lbls)
            except:
                continue
        
        self.word_vocab = word_vocab or self._build_word_vocab()
        self.char_vocab = char_vocab or self._build_char_vocab()
        self.label_vocab = label_vocab or {
            "O": 0, "B-PER": 1, "I-PER": 2,
            "B-ORG": 3, "I-ORG": 4,
            "B-LOC": 5, "I-LOC": 6,
            "B-MISC": 7, "I-MISC": 8,
        }
        self.reverse_label_vocab = {v: k for k, v in self.label_vocab.items()}
        
    def _build_word_vocab(self):
        word_counts = Counter()
        for sentence in self.sentences:
            word_counts.update([word.lower() for word in sentence])
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        vocab.update({word: i+2 for i, (word, _) in enumerate(word_counts.most_common())})
        return vocab
    
    def _build_char_vocab(self):
        chars = set()
        for sentence in self.sentences:
            for word in sentence:
                chars.update(word.lower())
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        vocab.update({char: i+2 for i, char in enumerate(sorted(chars))})
        return vocab
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.labels[idx]
        
        word_ids = [self.word_vocab.get(word.lower(), 1) for word in words]
        
        char_ids = []
        for word in words:
            chars = [self.char_vocab.get(c.lower(), 1) for c in word[:config['max_char_len']]]
            chars += [0] * (config['max_char_len'] - len(chars))
            char_ids.append(chars)
        
        if isinstance(labels[0], str):
            label_ids = [self.label_vocab.get(tag, 0) for tag in labels]
        else:
            label_ids = [int(tag) for tag in labels]
        
        return {
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'char_ids': torch.tensor(char_ids, dtype=torch.long),
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'length': len(word_ids)
        }

def collate_fn(batch):
    batch = [b for b in batch if len(b['word_ids']) > 0]
    if not batch:
        return {
            'word_ids': torch.zeros(1, 1, dtype=torch.long),
            'char_ids': torch.zeros(1, 1, config['max_char_len'], dtype=torch.long),
            'label_ids': torch.zeros(1, 1, dtype=torch.long),
            'mask': torch.zeros(1, 1, dtype=torch.bool)
        }
    
    word_ids = [item['word_ids'] for item in batch]
    char_ids = [item['char_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    sort_order = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
    word_ids = [word_ids[i] for i in sort_order]
    char_ids = [char_ids[i] for i in sort_order]
    label_ids = [label_ids[i] for i in sort_order]
    lengths = [lengths[i] for i in sort_order]
    
    max_len = min(config['max_seq_len'], max(lengths))
    padded_words = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_chars = torch.zeros(len(batch), max_len, config['max_char_len'], dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, (w, c, l, ln) in enumerate(zip(word_ids, char_ids, label_ids, lengths)):
        actual_len = min(ln, max_len)
        padded_words[i, :actual_len] = w[:actual_len]
        padded_chars[i, :actual_len] = c[:actual_len]
        padded_labels[i, :actual_len] = l[:actual_len]
        mask[i, :actual_len] = 1
    
    return {
        'word_ids': padded_words,
        'char_ids': padded_chars,
        'label_ids': padded_labels,
        'mask': mask
    }

class Highway(nn.Module):
    def __init__(self, size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        for layer in range(self.num_layers):
            gate = self.sigmoid(self.gate[layer](x))
            nonlinear = torch.relu(self.nonlinear[layer](x))
            x = gate * nonlinear + (1 - gate) * x
        return x

class CharCNNEncoder(nn.Module):
    def __init__(self, char_vocab_size):
        super().__init__()
        self.char_embed_dim = config['char_embed_dim']
        self.embedding = nn.Embedding(char_vocab_size, self.char_embed_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(self.char_embed_dim, config['char_hidden_dim'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.char_embed_dim, config['char_hidden_dim'], kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(self.char_embed_dim, config['char_hidden_dim'], kernel_size=7, padding=3)
        
        self.dropout = nn.Dropout(config['dropout'])
        self.highway = Highway(config['char_hidden_dim'] * 3)
        self.proj = nn.Linear(config['char_hidden_dim'] * 3, config['char_hidden_dim'])
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, char_ids):
        batch_size, seq_len, max_char_len = char_ids.size()
        char_ids = char_ids.view(-1, max_char_len)
        char_embs = self.embedding(char_ids)
        char_embs = char_embs.permute(0, 2, 1)
        
        conv1_out = torch.relu(self.conv1(char_embs))
        conv2_out = torch.relu(self.conv2(char_embs))
        conv3_out = torch.relu(self.conv3(char_embs))
        
        conv1_out, _ = torch.max(conv1_out, dim=2)
        conv2_out, _ = torch.max(conv2_out, dim=2)
        conv3_out, _ = torch.max(conv3_out, dim=2)
        
        char_features = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        char_features = self.dropout(char_features)
        char_features = self.highway(char_features)
        char_features = self.proj(char_features)
        
        char_features = char_features.view(batch_size, seq_len, -1)
        return char_features

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x, src_key_padding_mask=None):
        return self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

class FirstOrderCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.transitions)
        with torch.no_grad():
            self.transitions.data.clamp_(-1.0, 1.0)
    
    def forward(self, emissions, tags, mask):
        batch_size, seq_len, num_tags = emissions.size()
        
        score = (emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=1)
        
        if seq_len > 1:
            prev_tags = tags[:, :-1]
            next_tags = tags[:, 1:]
            trans_scores = self.transitions[prev_tags, next_tags] * mask[:, 1:]
            score += trans_scores.sum(dim=1)
        
        logZ = self._compute_log_partition(emissions, mask)
        return (logZ - score).mean()
    
    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        
        log_alpha = emissions[:, 0]  
        
        for t in range(1, seq_len):
            emit_scores = emissions[:, t].unsqueeze(1)  
            trans_scores = self.transitions.unsqueeze(0)  
            
            log_alpha_new = torch.logsumexp(
                log_alpha.unsqueeze(-1) + trans_scores + emit_scores,
                dim=1
            )
            
            mask_t = mask[:, t].unsqueeze(-1)
            log_alpha = torch.where(mask_t.bool(), log_alpha_new, log_alpha)
        
        return torch.logsumexp(log_alpha, dim=1)
    
    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        
        viterbi = torch.zeros(batch_size, seq_len, num_tags, device=emissions.device)
        backpointers = torch.zeros(batch_size, seq_len, num_tags, dtype=torch.long, device=emissions.device)
        
        viterbi[:, 0] = emissions[:, 0]
        
        for t in range(1, seq_len):
            emit_scores = emissions[:, t].unsqueeze(1)  
            trans_scores = self.transitions.unsqueeze(0)  
            
            scores = viterbi[:, t-1].unsqueeze(-1) + trans_scores + emit_scores
            viterbi[:, t], backpointers[:, t] = torch.max(scores, dim=1)
            
            mask_t = mask[:, t].unsqueeze(-1)
            viterbi[:, t] = torch.where(mask_t.bool(), viterbi[:, t], viterbi[:, t-1])
        
        best_paths = []
        _, last_tags = torch.max(viterbi[:, -1], dim=-1)
        
        for i in range(batch_size):
            path = [last_tags[i].item()]
            for t in range(seq_len-1, 0, -1):
                path.append(backpointers[i, t, path[-1]].item())
            best_paths.append(path[::-1])
        
        return best_paths

class EnhancedBiLSTM_CRF(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, label_vocab):
        super().__init__()
        self.word_embeds = nn.Embedding(word_vocab_size, config['word_embed_dim'], padding_idx=0)
        self.char_encoder = CharCNNEncoder(char_vocab_size)
        
        self.combine_proj = nn.Sequential(
            nn.Linear(config['word_embed_dim'] + config['char_hidden_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        
        self.bilstm = nn.LSTM(
            config['hidden_dim'],
            config['hidden_dim'] // 2,
            num_layers=config['lstm_layers'],
            bidirectional=True,
            batch_first=True,
            dropout=config['dropout'] if config['lstm_layers'] > 1 else 0
        )
        
        self.lstm_proj = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        
        self.transformer = TransformerEncoderWrapper(
            hidden_dim=config['hidden_dim'],
            num_heads=config['attention_heads'],
            num_layers=config['transformer_layers']
        )
        
        self.linear = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], len(label_vocab))
        )
        
        self.crf = FirstOrderCRF(len(label_vocab))
        self.dropout = nn.Dropout(config['dropout'])
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'embeds' not in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def load_glove_embeddings(self, word_vocab, glove_path):
        print("Loading GloVe embeddings...")
        glove_embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
                glove_embeddings[word] = vector
        
        embedding_matrix = torch.zeros(len(word_vocab), config['word_embed_dim'])
        for word, idx in word_vocab.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word]
            elif word.lower() in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word.lower()]
            else:
                embedding_matrix[idx] = torch.randn(config['word_embed_dim']) * 0.1
        
        self.word_embeds.weight.data.copy_(embedding_matrix)
        self.word_embeds.weight.requires_grad = True
    
    def forward(self, word_ids, char_ids, labels=None, mask=None):
        word_embs = self.word_embeds(word_ids)
        char_features = self.char_encoder(char_ids)
        
        combined = torch.cat([word_embs, char_features], dim=-1)
        combined = self.combine_proj(combined)
        
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            combined, lengths, batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=mask.size(1)
        )
        
        lstm_out = self.lstm_proj(lstm_out) + lstm_out
        
        attn_out = self.transformer(
            lstm_out,
            src_key_padding_mask=~mask.bool()
        )
        
        emissions = self.linear(attn_out)
        
        if labels is not None:
            labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
            loss = self.crf(emissions, labels, mask=mask.bool())
            return loss
        
        return self.crf.decode(emissions, mask=mask.bool())

class EarlyStopper:
    def __init__(self, patience=5, min_improvement=0.001):
        self.patience = patience
        self.min_improvement = min_improvement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score * (1 + self.min_improvement):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def evaluate(model, dataloader, device, dataset):
    model.eval()
    true_tags, pred_tags = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            word_ids = batch['word_ids'].to(device)
            char_ids = batch['char_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            mask = batch['mask'].to(device)
            
            loss = model(word_ids, char_ids, label_ids, mask)
            total_loss += loss.item() * word_ids.size(0)
            
            preds = model(word_ids, char_ids, mask=mask)
            
            for i in range(word_ids.size(0)):
                valid_len = mask[i].sum().item()
                true = label_ids[i][:valid_len].cpu().numpy()
                pred = preds[i][:valid_len]
                
                true = [t for t in true if t != -100]
                pred = pred[:len(true)]
                
                if true:
                    true_str = [dataset.reverse_label_vocab.get(t, "O") for t in true]
                    pred_str = [dataset.reverse_label_vocab.get(p, "O") for p in pred]
                    true_tags.append(true_str)
                    pred_tags.append(pred_str)
    
    if not true_tags:
        return {
            'loss': 0.0,
            'f1': 0.0,
            'report': "No valid predictions"
        }
    
    avg_loss = total_loss / len(dataloader.dataset)
    f1 = f1_score(true_tags, pred_tags)
    
    print("\nSample predictions:")
    for i in range(min(3, len(true_tags))):
        print("True:", true_tags[i])
        print("Pred:", pred_tags[i])
        print()
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'report': classification_report(true_tags, pred_tags, digits=4)
    }

def train():
    train_dataset = CONLL2003Dataset("data/conll2003/train.csv")
    valid_dataset = CONLL2003Dataset("data/conll2003/valid.csv", 
                                   word_vocab=train_dataset.word_vocab,
                                   char_vocab=train_dataset.char_vocab,
                                   label_vocab=train_dataset.label_vocab)
    test_dataset = CONLL2003Dataset("data/conll2003/test.csv",
                                   word_vocab=train_dataset.word_vocab,
                                   char_vocab=train_dataset.char_vocab,
                                   label_vocab=train_dataset.label_vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedBiLSTM_CRF(
        len(train_dataset.word_vocab),
        len(train_dataset.char_vocab),
        train_dataset.label_vocab
    ).to(device)
    
    model.load_glove_embeddings(train_dataset.word_vocab, config['glove_path'])
    
    optimizer = AdamW(model.parameters(), 
                     lr=config['lr'], 
                     weight_decay=config['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
    early_stopper = EarlyStopper(
        patience=config['patience'],
        min_improvement=config['min_improvement']
    )
    best_f1 = 0.0
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            word_ids = batch['word_ids'].to(device)
            char_ids = batch['char_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16, enabled=config['use_amp']):
                loss = model(word_ids, char_ids, label_ids, mask)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        valid_results = evaluate(model, valid_loader, device, valid_dataset)
        print(f"\nValidation - Epoch {epoch+1}:")
        print(f"Loss: {valid_results['loss']:.4f} | F1: {valid_results['f1']:.4f}")
        print(valid_results['report'])
        
        scheduler.step(valid_results['f1'])
        
        if valid_results['f1'] > best_f1:
            best_f1 = valid_results['f1']
            torch.save({
                'model_state_dict': model.state_dict(),
                'word_vocab': train_dataset.word_vocab,
                'char_vocab': train_dataset.char_vocab,
                'label_vocab': train_dataset.label_vocab,
                'config': config,
                'best_f1': best_f1
            }, config['model_save_path'])
            print(f"Saved best model with F1: {best_f1:.4f}")
        
        early_stopper(valid_results['f1'])
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("\nLoading best model for testing...")
    checkpoint = torch.load(config['model_save_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, device, test_dataset)
    print("\nTest Results:")
    print(f"F1: {test_results['f1']:.4f}")
    print(test_results['report'])

if __name__ == "__main__":
    train()