import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))
        # outputs: [batch, src_len, hid_dim * 2]
        # hidden: [2, batch, hid_dim]
        outputs, hidden = self.rnn(embedded)
        
        # Concat the two directions of the last hidden state
        # hidden[-2, :, :] is forward, hidden[-1, :, :] is backward
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # energy: [batch, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # attention: [batch, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch]
        # hidden: [batch, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim * 2]
        
        input = input.unsqueeze(1) # [batch, 1]
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs) # [batch, src_len]
        a = a.unsqueeze(1) # [batch, 1, src_len]
        
        # weighted: [batch, 1, hid_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        # output: [batch, 1, hid_dim]
        # hidden: [1, batch, hid_dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        assert (output == hidden.transpose(0, 1)).all()
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(self.dropout(torch.cat((output, weighted, embedded), dim=1)))
        
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = []
        
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the <sos> tokens
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            prediction, hidden, attention = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = prediction
            attentions.append(attention)
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs, torch.stack(attentions, dim=1) if attentions else None
