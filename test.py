from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer        # the class that can create the vocab given the list of sentences
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))    # e.g: config[tokenizer_file] = ".../tokenizers/tokenizer_{lang}.json"
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))           # If the tokenizer sees a word that is not in the vocabulary it was trained on, it will replace the word with [UNK]
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)        # min_freq: for a word to appear in the vocab, it needs to appear at least 2 times
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 1,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

config = get_config()
ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])

sos_id = tokenizer_src.token_to_id('[SOS]')
print(sos_id)
