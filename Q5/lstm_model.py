import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=4, dropout=0.4):
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM Layer: expects input=[seq_len, batch, embed_dim]
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout)
        
        # Output projection
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        
        # 🚀 WEIGHT TYING RULE
        # Bind embedding weights to output (decoder) weights.
        # This drastically reduces the number of parameters and provides regularization.
        if embed_dim == hidden_dim:
            self.decoder.weight = self.encoder.weight
            
        self.init_weights()

    def init_weights(self):
        # Uniform initialization (Zaremba 2014 standard)
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        # Due to weight tying, self.decoder.weight is already linked to the encoder.

    def forward(self, input, hidden):
        # input: [seq_len, batch_size]
        emb = self.drop(self.encoder(input))
        
        # output: [seq_len, batch_size, hidden_dim]
        # hidden: (h_n, c_n)
        output, hidden = self.rnn(emb, hidden)
        
        output = self.drop(output)
        
        # Flatten for logit computation
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        
        # Reshape to original: [seq_len, batch_size, vocab_size]
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state (used at the beginning of epochs)
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
