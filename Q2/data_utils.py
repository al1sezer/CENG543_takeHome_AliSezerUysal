import logging
from datasets import load_dataset
from transformers import AutoTokenizer

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_conll_dataset():
    """
    Loads CoNLL-2003 dataset from Hugging Face.
    """
    try:
        # Use Parquet version to avoid Hub script restrictions.
        dataset = load_dataset("lhoestq/conll2003")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def align_labels_with_tokens(labels, word_ids):
    """
    Strategy A: Label first subword with original tag, mask others with -100.
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # New word started
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special tokens (CLS, SEP, etc.)
            new_labels.append(-100)
        else:
            # Subwords of the same word
            new_labels.append(-100)
    return new_labels

def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    """
    Tokenize and align labels for Transformer models.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True, 
        max_length=max_length,
        padding="max_length"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels.append(align_labels_with_tokens(label, word_ids))

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

class CRFFeatureExtractor:
    """
    Extract features for CRF model.
    """
    @staticmethod
    def word2features(sent, i):
        word = sent[i]
        
        # Base features: word, case sensitivity, suffixes
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        
        # Context window: Previous word
        if i > 0:
            word1 = sent[i-1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True # Beginning of Sentence

        # Context window: Next word
        if i < len(sent)-1:
            word1 = sent[i+1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True # End of Sentence

        return features

    @classmethod
    def sent2features(cls, sent):
        return [cls.word2features(sent, i) for i in range(len(sent))]
