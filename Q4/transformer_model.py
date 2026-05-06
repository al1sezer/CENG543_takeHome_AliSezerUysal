import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x.transpose(0, 1) # [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1) # [batch, seq_len, d_model]

class MiniTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_enc_layers, num_dec_layers, d_ff, dropout, pad_idx):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        # src: [batch, src_len]
        src_mask = (src == self.pad_idx)
        return src_mask # [batch, src_len]

    def make_tgt_mask(self, tgt):
        # tgt: [batch, tgt_len]
        tgt_len = tgt.shape[1]
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Transformer in PyTorch expected pad masks to be True for padding positions
        # Actually src_key_padding_mask is [batch, src_len]
        
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        tgt_padding_mask = (tgt == self.pad_idx)
        
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_mask
        )
        
        return self.fc_out(output)

def get_transformer_attention_weights(model, src, tgt_input):
    # Hook to capture cross-attention weights
    attn_weights = []

    def hook_fn(module, input, output):
        # output[1] is the attention weights for MultiheadAttention
        attn_weights.append(output[1])

    # Register hook on the last decoder layer's multihead_attn
    handle = model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(src, tgt_input)
    
    handle.remove()
    
    # attn_weights is a list containing the weights from the hook call
    # multihead_attn returns weights as [batch, tgt_len, src_len] when batch_first=True
    return attn_weights[0] if attn_weights else None
