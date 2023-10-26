import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer        # the class that can create the vocab given the list of sentences
from tokenizers.pre_tokenizers import Whitespace 
from pathlib import Path