import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu

# Cấu hình dữ liệu
DATA_DIR = './Week6/dataset'
EN_FILE = os.path.join(DATA_DIR, 'en_sents')
VI_FILE = os.path.join(DATA_DIR, 'vi_sents')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiền xử lý
def load_data(en_path, vi_path):
    with open(en_path, 'r', encoding='utf-8') as f:
        en_sentences = [line.strip().lower() for line in f.readlines()]
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_sentences = [line.strip().lower() for line in f.readlines()]
    return list(zip(en_sentences, vi_sentences))

def tokenize(sentence):
    return sentence.split()

class Vocab:
    def __init__(self, sentences, min_freq=1):
        self.freq = {}
        for sent in sentences:
            for word in tokenize(sent):
                self.freq[word] = self.freq.get(word, 0) + 1
        self.itos = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        for word, f in self.freq.items():
            if f >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
    
    def encode(self, sentence):
        return [self.stoi.get(word, self.stoi['<unk>']) for word in tokenize(sentence)]
    
    def decode(self, indices):
        return ' '.join([self.itos[i] for i in indices if i not in (self.stoi['<pad>'], self.stoi['<sos>'], self.stoi['<eos>'])])
    
    def __len__(self):
        return len(self.itos)

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = [self.src_vocab.stoi['<sos>']] + self.src_vocab.encode(src) + [self.src_vocab.stoi['<eos>']]
        tgt_ids = [self.tgt_vocab.stoi['<sos>']] + self.tgt_vocab.encode(tgt) + [self.tgt_vocab.stoi['<eos>']]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    return src_batch, tgt_batch

# Mô hình
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers)
    
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers)
        self.fc = nn.Linear(hid_dim, output_dim)
    
    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        hidden = self.encoder(src)
        input = tgt[0,:]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if random.random() < teacher_forcing_ratio else top1
        return outputs

    def translate(self, src, max_len=50):
        self.eval()
        with torch.no_grad():
            src = src.to(self.device)
            hidden = self.encoder(src)
            input = torch.tensor([1] * src.shape[1]).to(self.device)  # <sos>
            outputs = []
            for _ in range(max_len):
                output, hidden = self.decoder(input, hidden)
                top1 = output.argmax(1)
                outputs.append(top1)
                input = top1
            return torch.stack(outputs, dim=0)

# Huấn luyện
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, tgt in iterator:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt = tgt[1:].view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Đánh giá Loss
def evaluate_loss(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in iterator:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Đánh giá BLEU
def evaluate_bleu(model, dataloader, src_vocab, tgt_vocab):
    refs = []
    hyps = []
    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        output = model.translate(src)
        for i in range(tgt.shape[1]):
            ref = tgt[:, i].tolist()
            hyp = output[:, i].tolist()
            refs.append([tgt_vocab.decode(ref).split()])
            hyps.append(tgt_vocab.decode(hyp).split())
    return corpus_bleu(refs, hyps)

# Cấu hình thử nghiệm
CONFIGS = [
    {"emb_dim": 128, "hid_dim": 256, "n_layers": 1, "batch_size": 64, "lr": 0.001},
    {"emb_dim": 256, "hid_dim": 512, "n_layers": 2, "batch_size": 64, "lr": 0.0005},
    {"emb_dim": 128, "hid_dim": 256, "n_layers": 2, "batch_size": 32, "lr": 0.001},
    {"emb_dim": 256, "hid_dim": 512, "n_layers": 1, "batch_size": 32, "lr": 0.0007},
    {"emb_dim": 200, "hid_dim": 400, "n_layers": 2, "batch_size": 64, "lr": 0.0003},
]

def main():
    data = load_data(EN_FILE, VI_FILE)
    src_sentences, tgt_sentences = zip(*data)
    src_vocab = Vocab(src_sentences)
    tgt_vocab = Vocab(tgt_sentences)

    dataset = TranslationDataset(data, src_vocab, tgt_vocab)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    for i, config in enumerate(CONFIGS):
        config_name = f"cfg{i+1}_emb{config['emb_dim']}_hid{config['hid_dim']}"
        wandb.init(project="rnn-seq2seq", config=config, name=config_name)
        
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

        encoder = Encoder(len(src_vocab), config["emb_dim"], config["hid_dim"], config["n_layers"])
        decoder = Decoder(len(tgt_vocab), config["emb_dim"], config["hid_dim"], config["n_layers"])
        model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(20):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss = evaluate_loss(model, val_loader, criterion)
            bleu = evaluate_bleu(model, val_loader, src_vocab, tgt_vocab)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu": bleu
            })
        
        wandb.finish()

if __name__ == "__main__":
    main()
