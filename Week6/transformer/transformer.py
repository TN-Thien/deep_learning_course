import os
import pickle
import re
import string
import math
import warnings
import wandb
from timeit import default_timer as timer
from typing import Iterable, List

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torchtext
torchtext.disable_torchtext_deprecation_warning()
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

warnings.filterwarnings('ignore')

# ĐƯỜNG DẪN DỮ LIỆU
DATA_DIR = './Week6/dataset'

EN_FILE = os.path.join(DATA_DIR, 'en_sents')
VI_FILE = os.path.join(DATA_DIR, 'vi_sents')

# ĐỌC DỮ LIỆU
def load_data(en_file, vi_file):
    with open(en_file, 'r', encoding='utf-8') as f:
        en_sents = f.read().splitlines()
    with open(vi_file, 'r', encoding='utf-8') as f:
        vi_sents = f.read().splitlines()
    return en_sents, vi_sents

# TIỀN XỬ LÝ DỮ LIỆU
def preprocessing(df):
    # Loại bỏ dấu câu
    # df["en"] = df["en"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation)))
    # df["vi"] = df["vi"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation)))
    # Chuyển về chữ thường
    df["en"] = df["en"].apply(lambda ele: ele.lower())
    df["vi"] = df["vi"].apply(lambda ele: ele.lower())
    # Xóa khoảng trắng đầu cuối
    df["en"] = df["en"].apply(lambda ele: ele.strip())
    df["vi"] = df["vi"].apply(lambda ele: ele.strip())
    # Chuẩn hóa khoảng trắng
    df["en"] = df["en"].apply(lambda ele: re.sub("\s+", " ", ele))
    df["vi"] = df["vi"].apply(lambda ele: re.sub("\s+", " ", ele))
    return df

# TOKENIZER
def vi_tokenizer(sentence):
    return word_tokenize(sentence)

# TẠO VOCAB
def yield_tokens(data_iter: Iterable, language: str, token_transform):
    for _, data_sample in data_iter:
        yield token_transform[language](data_sample[language])

# ĐỊNH NGHĨA CHỈ SỐ ĐẶC BIỆT
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# TẠO MASK
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, PAD_IDX, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# LỚP ĐỊNH VỊ
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# LỚP EMBEDDING
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# MÔ HÌNH TRANSFORMER
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory, tgt_mask)

# CHUYỂN TEXT THÀNH TENSOR
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

def collate_fn(batch, text_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# HÀM DỊCH (GREEDY)
def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# HÀM DỊCH TOÀN BỘ
def translate(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer,
              max_len=50, device=torch.device('cpu')):
    model.eval()
    src_tokens = tensor_transform([src_vocab[token] for token in src_tokenizer(src_sentence)])
    num_tokens = src_tokens.size(0)
    src = src_tokens.unsqueeze(1).to(device)
    src_mask = torch.zeros((num_tokens, num_tokens), device=device).type(torch.bool)

    tgt_tokens = greedy_decode(model, src, src_mask, max_len, BOS_IDX, device).flatten()
    tgt_tokens = tgt_tokens.cpu().numpy()

    tokens = []
    for tok in tgt_tokens:
        token = tgt_vocab.lookup_token(tok)
        if token == '<eos>':
            break
        if token not in ['<bos>', '<pad>']:
            tokens.append(token)
    return " ".join(tokens)

# ĐIỂM BLEU
def calculate_bleu(references, hypotheses):
    smoothie = SmoothingFunction().method4
    return corpus_bleu([[ref] for ref in references], hypotheses, smoothing_function=smoothie)

# MAIN
def main():
    # Thiết lập thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Đọc dữ liệu
    en_sents, vi_sents = load_data(EN_FILE, VI_FILE)
    df = pd.DataFrame({'en': en_sents, 'vi': vi_sents})

    # Tiền xử lý
    df = preprocessing(df)

    # Chia train-test
    train_size = int(0.9 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Định nghĩa tokenizer
    token_transform = {}
    token_transform['en'] = get_tokenizer('basic_english')
    token_transform['vi'] = vi_tokenizer

    # Xây dựng vocab
    train_iter = [(row['en'], row['vi']) for _, row in train_df.iterrows()]
    def yield_tokens(data_iter, language):
        for src_sample, tgt_sample in data_iter:
            yield token_transform[language](src_sample if language == 'en' else tgt_sample)

    src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, 'en'),
                                          min_freq=2,
                                          specials=special_symbols,
                                          special_first=True)
    tgt_vocab = build_vocab_from_iterator(yield_tokens(train_iter, 'vi'),
                                          min_freq=2,
                                          specials=special_symbols,
                                          special_first=True)

    src_vocab.set_default_index(UNK_IDX)
    tgt_vocab.set_default_index(UNK_IDX)

    with open("./Week6/transformer/src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)

    with open("./Week6/transformer/tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    # Chuẩn bị text transform
    text_transform = {}

    def vocab_transform(vocab):
        return lambda x: torch.tensor([vocab[token] for token in x], dtype=torch.long)

    text_transform['en'] = sequential_transforms(token_transform['en'], vocab_transform(src_vocab), tensor_transform)
    text_transform['vi'] = sequential_transforms(token_transform['vi'], vocab_transform(tgt_vocab), tensor_transform)

    # Tạo DataLoader
    train_data = [(row['en'], row['vi']) for _, row in train_df.iterrows()]
    test_data = [(row['en'], row['vi']) for _, row in test_df.iterrows()]

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, text_transform, 'en', 'vi', PAD_IDX))
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, text_transform, 'en', 'vi', PAD_IDX))

    # Khởi tạo model
    # EMB_SIZE = 512
    # NHEAD = 8
    # FFN_HID_DIM = 512
    # NUM_ENCODER_LAYERS = 3
    # NUM_DECODER_LAYERS = 3

    # model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
    #                            len(src_vocab), len(tgt_vocab), FFN_HID_DIM)

    # model = model.to(device)

    # # Loss và optimizer
    # loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Hàm train
    def train_epoch(model, optimizer):
        model.train()
        losses = 0
        for src, tgt in train_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, device)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            loss.backward()
            optimizer.step()

            losses += loss.item()

        return losses / len(train_dataloader)

    # Hàm evaluate
    def evaluate(model):
        model.eval()
        losses = 0
        with torch.no_grad():
            for src, tgt in test_dataloader:
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_input = tgt[:-1, :]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, device)

                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

                tgt_out = tgt[1:, :]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()

        return losses / len(test_dataloader)

    # Huấn luyện
    configurations = [
        {"emb_size": 256, "nhead": 4, "ffn_dim": 512, "enc_layers": 2, "dec_layers": 2},
        {"emb_size": 512, "nhead": 8, "ffn_dim": 512, "enc_layers": 3, "dec_layers": 3},
        {"emb_size": 512, "nhead": 8, "ffn_dim": 1024, "enc_layers": 4, "dec_layers": 4},
        {"emb_size": 256, "nhead": 4, "ffn_dim": 1024, "enc_layers": 2, "dec_layers": 3},
        {"emb_size": 384, "nhead": 6, "ffn_dim": 768, "enc_layers": 3, "dec_layers": 3},
    ]

    # Duyệt từng cấu hình
    for i, cfg in enumerate(configurations):
        wandb.init(
            project="nmt-transformer-en-vi",
            name=f"run_{i+1}_emb{cfg['emb_size']}_head{cfg['nhead']}",
            config=cfg
        )

        print(f"\nĐang huấn luyện cấu hình {i+1}: {cfg}")

        # Tạo mô hình mới
        model = Seq2SeqTransformer(
            num_encoder_layers=cfg['enc_layers'],
            num_decoder_layers=cfg['dec_layers'],
            emb_size=cfg['emb_size'],
            nhead=cfg['nhead'],
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            dim_feedforward=cfg['ffn_dim']
        ).to(device)

        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        NUM_EPOCHS = 20
        best_val_loss = float('inf')

        for epoch in range(1, NUM_EPOCHS + 1):
            start_time = timer()
            train_loss = train_epoch(model, optimizer)
            val_loss = evaluate(model)
            end_time = timer()

            # BLEU
            sample_refs = [token_transform['vi'](sent) for sent in test_df['vi'].tolist()[:50]]
            sample_hyps = [
                translate(model, src_sent, src_vocab, tgt_vocab, token_transform['en'], device=device).split()
                for src_sent in test_df['en'].tolist()[:50]
            ]
            bleu = calculate_bleu(sample_refs, sample_hyps)

            # Ghi log
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu_score": bleu,
                "time": end_time - start_time
            })

            print(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f} | BLEU {bleu:.4f} | Time {end_time - start_time:.2f}s")

            # Lưu mô hình tốt nhất
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_config{i+1}.pth')
                print(f"[CONFIG {i+1}] Đã lưu mô hình tốt nhất tại epoch {epoch}")

        wandb.finish()

    # Dịch thử
    example_sentence = "this is a test sentence ."
    translation = translate(model, example_sentence, src_vocab, tgt_vocab, token_transform['en'], device=device)
    print(f"Source: {example_sentence}")
    print(f"Translation: {translation}")

    # Đánh giá BLEU
    references = [token_transform['vi'](sent) for sent in test_df['vi'].tolist()]
    hypotheses = []

    for src_sent in test_df['en'].tolist():
        tgt_trans = translate(model, src_sent, src_vocab, tgt_vocab, token_transform['en'], device=device)
        hypotheses.append(tgt_trans.split())

    bleu_score = calculate_bleu(references, hypotheses)
    print(f"BLEU score: {bleu_score:.4f}")

if __name__ == "__main__":
    main()
