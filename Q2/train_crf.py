import logging
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from seqeval.metrics import classification_report
from data_utils import get_conll_dataset, CRFFeatureExtractor

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data_for_crf(split_data):
    """
    Convert HuggingFace split to CRF features and labels.
    """
    X = []
    y = []
    
    # Map integers to BIO tags
    # 0:O, 1:B-PER, 2:I-PER, 3:B-ORG, 4:I-ORG, 5:B-LOC, 6:I-LOC, 7:B-MISC, 8:I-MISC
    label_map = {
        0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 
        5:'B-LOC', 6:'I-LOC', 7:'B-MISC', 8:'I-MISC'
    }

    for example in split_data:
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        
        X.append(CRFFeatureExtractor.sent2features(tokens))
        y.append([label_map[t] for t in ner_tags])
        
    return X, y

def train_and_eval_crf():
    logger.info("Loading CoNLL-2003 dataset...")
    dataset = get_conll_dataset()
    if not dataset:
        return

    logger.info("Converting data to CRF format...")
    X_train, y_train = prepare_data_for_crf(dataset['train'])
    X_test, y_test = prepare_data_for_crf(dataset['test'])

    logger.info("Training CRF model (L-BFGS)...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, # L1 regularization
        c2=0.1, # L2 regularization
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    
    try:
        crf.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

    logger.info("Predicting on test set...")
    y_pred = crf.predict(X_test)

    # seqeval formatinda raporlama
    logger.info("CRF Performance Report (Entity-Level):")
    report = classification_report(y_test, y_pred)
    
    # Save results to file. 
    with open("crf_results.txt", "w") as f:
        f.write("CRF Baseline Results\n")
        f.write("====================\n")
        f.write(report)
    
    logger.info("Results saved to crf_results.txt")

if __name__ == "__main__":
    train_and_eval_crf()
