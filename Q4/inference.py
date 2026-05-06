import torch
from config import SOS_IDX, EOS_IDX, PAD_IDX, DEVICE, MAX_LEN
from transformer_model import get_transformer_attention_weights

def greedy_decode_seq2seq(model, src_tensor, tgt_vocab, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        input = torch.tensor([SOS_IDX]).to(DEVICE)
        tokens = []
        attentions = []
        
        for _ in range(max_len):
            prediction, hidden, attention = model.decoder(input, hidden, encoder_outputs)
            top1 = prediction.argmax(1)
            
            if top1.item() == EOS_IDX:
                break
            
            tokens.append(tgt_vocab.itos[top1.item()])
            attentions.append(attention.squeeze(0).cpu().numpy())
            input = top1
            
    return tokens, attentions

def greedy_decode_transformer(model, src_tensor, tgt_vocab, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        src_emb = model.pos_encoding(model.src_embedding(src_tensor) * (model.d_model ** 0.5))
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        
        tgt_indices = [SOS_IDX]
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(DEVICE)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            tgt_emb = model.pos_encoding(model.tgt_embedding(tgt_tensor) * (model.d_model ** 0.5))
            
            output = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
            prediction = model.fc_out(output)
            
            next_token = prediction[0, -1, :].argmax().item()
            tgt_indices.append(next_token)
            
            if next_token == EOS_IDX:
                break
        
        # Translate indices to tokens (excluding SOS and EOS)
        tokens = [tgt_vocab.itos[i] for i in tgt_indices if i not in [SOS_IDX, EOS_IDX]]
        
        # Get attention weights for the final result
        tgt_tensor = torch.LongTensor(tgt_indices[:-1]).unsqueeze(0).to(DEVICE) # Input for cross-attention
        attentions = get_transformer_attention_weights(model, src_tensor, tgt_tensor)
        # attentions: [1, tgt_len, src_len]
        if attentions is not None:
            attentions = attentions.squeeze(0).cpu().numpy()

    return tokens, attentions

def translate_sentence(sentence, model, src_vocab, tgt_vocab, tokenize_fn, is_transformer=False):
    tokens = tokenize_fn(sentence)
    src_indices = [SOS_IDX] + src_vocab.numericalize(tokens) + [EOS_IDX]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)
    
    if is_transformer:
        return greedy_decode_transformer(model, src_tensor, tgt_vocab)
    else:
        return greedy_decode_seq2seq(model, src_tensor, tgt_vocab)

def translate_batch(sentences, model, src_vocab, tgt_vocab, tokenize_fn, is_transformer=False):
    """Batch translation for metric computation (with tqdm)."""
    from tqdm import tqdm
    results = []
    for sent in tqdm(sentences, desc="Translating Batch", leave=False):
        tokens, _ = translate_sentence(sent, model, src_vocab, tgt_vocab, tokenize_fn, is_transformer)
        results.append(" ".join(tokens))
    return results
