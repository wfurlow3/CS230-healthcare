import sys

try:
    from transformers import BertConfig, BertForMaskedLM
except ImportError:
    print("error: transformers is required but not installed. Try `pip install transformers`.")
    sys.exit(1)

from .config import (
    ATTN_DROPOUT_PROB,
    HIDDEN_DROPOUT_PROB,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MAX_POSITION_EMBEDDINGS,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
)


def make_mlm_model(vocab_size: int) -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=ATTN_DROPOUT_PROB,
    )
    return BertForMaskedLM(config)
