from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def train_tokenizer(filename, vocab_size, lang):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<sos>", "<eos>", "<mask>"]
        )
    
    tokenizer.train([filename], trainer)
    tokenizer.save(f"{lang}_tokenizer.json")
    
    return tokenizer

en_tokenizer = train_tokenizer("data/Multi30k/train.en", vocab_size=10000, lang="en")
de_tokenizer = train_tokenizer("data/Multi30k/train.de", vocab_size=10000, lang="de")