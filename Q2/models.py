import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # Embedding Layer (Weights loaded via GloVe in training script)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Map LSTM output to tag space (Emissions)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF Layer
        self.crf = CRF(self.tagset_size, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)

    def _get_lstm_features(self, sentence):
        """
        Input: (batch_size, seq_len)
        Output: (batch_size, seq_len, tagset_size) - Emission scores
        """
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood(self, sentence, tags, mask):
        """
        Calculate CRF loss during training.
        """
        feats = self._get_lstm_features(sentence)
        # Negative log likelihood for minimization
        return -self.crf(feats, tags, mask=mask, reduction='token_mean')

    def forward(self, sentence, mask):
        """
        Predict most likely tag sequence (Viterbi decoding) during inference.
        """
        feats = self._get_lstm_features(sentence)
        # Viterbi decoding
        return self.crf.decode(feats, mask=mask)
