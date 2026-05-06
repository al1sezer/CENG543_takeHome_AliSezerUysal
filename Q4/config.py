import torch

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

# ── Device Configuration ──────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset & Vocabulary ──────────────────────────────────────
DATASET_NAME = "bentrevett/multi30k"
SPACY_DE = "de_core_news_sm"
SPACY_EN = "en_core_web_sm"

MAX_LEN = 50
MIN_FREQ = 2

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = {
    "<pad>": PAD_IDX,
    "<sos>": SOS_IDX,
    "<eos>": EOS_IDX,
    "<unk>": UNK_IDX,
}

# ── General Training ──────────────────────────────────────────
BATCH_SIZE = 128
CLIP_GRAD = 1.0

# ── Seq2Seq + Bahdanau Attention ──────────────────────────────
SEQ2SEQ_EMB_DIM = 256
SEQ2SEQ_HIDDEN_DIM = 256
SEQ2SEQ_ENC_LAYERS = 1
SEQ2SEQ_DEC_LAYERS = 1
SEQ2SEQ_ENC_DROPOUT = 0.5
SEQ2SEQ_DEC_DROPOUT = 0.5
SEQ2SEQ_LR = 1e-3
NUM_EPOCHS_SEQ2SEQ = 50
EARLY_STOP_PATIENCE = 5
TEACHER_FORCING_RATIO = 0.5

# ── Mini-Transformer ──────────────────────────────────────────
TF_D_MODEL = 256
TF_NHEAD = 8
TF_NUM_ENC_LAYERS = 4
TF_NUM_DEC_LAYERS = 4
TF_D_FF = 512
TF_DROPOUT = 0.1
TF_LR = 5e-4
TF_WARMUP_STEPS = 4000
TF_LABEL_SMOOTHING = 0.1
NUM_EPOCHS_TRANSFORMER = 50

# ── BERTScore ─────────────────────────────────────────────────
BERTSCORE_MODEL = "roberta-large"
