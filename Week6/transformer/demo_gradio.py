import torch
import gradio as gr
import pickle
import torchtext

from transformer import Seq2SeqTransformer, generate_square_subsequent_mask, tensor_transform, vi_tokenizer, UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocab đã lưu bằng pickle
with open("./Week6/transformer/src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)

with open("./Week6/transformer/tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

# Tokenizer
token_transform = {
    'en': torchtext.data.utils.get_tokenizer('basic_english'),
    'vi': vi_tokenizer
}

# Model config
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 1024
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

# Load model
model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
                           len(src_vocab), len(tgt_vocab), FFN_HID_DIM)
model.load_state_dict(torch.load("./Week6/transformer/best_model_config3.pth", map_location=device))
model = model.to(device)
model.eval()

# Hàm dịch
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for _ in range(max_len - 1):
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

def translate_sentence(sentence):
    tokens = token_transform['en'](sentence)
    src_tensor = tensor_transform([src_vocab[token] for token in tokens]).unsqueeze(1).to(device)
    num_tokens = src_tensor.size(0)
    src_mask = torch.zeros((num_tokens, num_tokens), device=device).type(torch.bool)

    tgt_tokens = greedy_decode(model, src_tensor, src_mask, max_len=50, start_symbol=BOS_IDX).flatten()
    decoded_tokens = [tgt_vocab.lookup_token(tok.item()) for tok in tgt_tokens]

    # Loại bỏ token đặc biệt
    translated = []
    for tok in decoded_tokens:
        if tok == '<eos>':
            break
        if tok not in ['<bos>', '<pad>']:
            translated.append(tok)
    return ' '.join(translated)

# Giao diện Gradio
gr.Interface(
    fn=translate_sentence,
    inputs=gr.Textbox(label="Nhập câu tiếng Anh"),
    outputs=gr.Textbox(label="Dịch tiếng Việt"),
    title="English to Vietnamese Translation (Transformer)",
    description="Nhập câu tiếng Anh để dịch sang tiếng Việt bằng mô hình Transformer."
).launch()
