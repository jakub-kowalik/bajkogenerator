from transformers import XLMTokenizer


def create_tokenizer() -> XLMTokenizer:
    return XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
