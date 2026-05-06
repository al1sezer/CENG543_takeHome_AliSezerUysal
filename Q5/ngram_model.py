import math
import random
import torch
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class NgramLanguageModel:
    def __init__(self, n=5, discount=0.75):
        self.n = n
        self.discount = discount
        
        # N-gram counts
        self.counts = defaultdict(Counter) # counts[context][word]
        self.context_counts = Counter()    # sum of counts for context
        
        self.vocab = set()
        self.vocab_list = []
        self.vocab_size = 0
        
        # For Kneser-Ney unigram backoff calculations
        self.num_context_types = defaultdict(int) # Number of unique contexts preceding 'word'
        self.total_bigram_types = 0

    def train(self, tokens):
        logger.info(f"Training {self.n}-gram model ({len(tokens)} tokens)...")
        self.vocab = set(tokens)
        self.vocab_list = list(self.vocab)
        self.vocab_size = len(self.vocab)
        
        # Counting N-gram frequencies
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i : i + self.n - 1])
            word = tokens[i + self.n - 1]
            
            self.counts[context][word] += 1
            self.context_counts[context] += 1

        # Pre-calculation for Kneser-Ney continuation probabilities
        seen_bigrams = set()
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            if (w1, w2) not in seen_bigrams:
                seen_bigrams.add((w1, w2))
                self.num_context_types[w2] += 1
                self.total_bigram_types += 1
                
        logger.info(f"{self.n}-gram model successfully trained.")

    def score(self, word, context):
        """ Calculates P(word | context) using Absolute Discounting + KN Backoff """
        context = tuple(context[-(self.n - 1):])
        
        if len(context) == self.n - 1:
            count_cw = self.counts[context][word]
            count_c = self.context_counts[context]
            
            if count_c > 0:
                # 1. Discounted probability
                prob = max(count_cw - self.discount, 0) / count_c
                
                # 2. Interpolation weight (Lambda)
                unique_following = len(self.counts[context])
                lam = (self.discount / count_c) * unique_following
                
                # 3. Kneser-Ney Unigram Backoff
                backoff_prob = self.num_context_types[word] / max(self.total_bigram_types, 1)
                if backoff_prob == 0:
                    backoff_prob = 1.0 / self.vocab_size # Fallback
                    
                return prob + lam * backoff_prob
                
        # If context never seen, return backoff directly
        backoff_prob = self.num_context_types[word] / max(self.total_bigram_types, 1)
        if backoff_prob == 0:
            backoff_prob = 1.0 / self.vocab_size
        return backoff_prob

    def perplexity(self, tokens):
        """ Calculates Perplexity (PPL) over test tokens. """
        log_prob_sum = 0
        N = len(tokens) - self.n + 1
        
        if N <= 0:
            return float('inf')
            
        for i in range(N):
            context = tuple(tokens[i : i + self.n - 1])
            word = tokens[i + self.n - 1]
            
            p = self.score(word, context)
            log_prob_sum -= math.log(max(p, 1e-10))
            
        # PPL = e^CE
        return math.exp(log_prob_sum / N)

    def generate(self, prompt_tokens, max_len=50, temperature=1.0, top_k=0, top_p=1.0):
        """
        Generates N-gram based text from the prompt.
        Manually applies strategies (using probabilities) identical to LSTM.
        """
        current_tokens = list(prompt_tokens)
        
        for _ in range(max_len):
            context = tuple(current_tokens[-(self.n - 1):])
            
            # Tüm kelimeler için olasılıkları hesapla
            probs = torch.tensor([self.score(w, context) for w in self.vocab_list])
            
            if temperature == 0:
                # Greedy Decoding
                best_idx = torch.argmax(probs).item()
                next_word = self.vocab_list[best_idx]
            else:
                # Temperature Scaling
                if temperature != 1.0:
                    logits = torch.log(probs + 1e-10)
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    
                # Top-k Sampling
                if top_k > 0:
                    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask[top_indices] = True
                    probs[~mask] = 0.0
                    probs = probs / probs.sum()
                    
                # Top-p (Nucleus) Sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                        
                # Sample
                if probs.sum() <= 0 or torch.isnan(probs).any():
                    next_word = random.choice(self.vocab_list) # Safety net
                else:
                    next_word_idx = torch.multinomial(probs, 1).item()
                    next_word = self.vocab_list[next_word_idx]
                    
            current_tokens.append(next_word)
            if next_word == '<eos>':
                break
                
        return current_tokens
