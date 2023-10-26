import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # Convert special tokens into token_id
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
    
    def __len__(self):
        return len(self.ds) 
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Convert each word into token ids
        src_token_ids = self.tokenizer_src.encode(src_text).ids         # First tokenize the sentence into words, then map each word to its corresponding ID. Return as an array
        tgt_token_ids = self.tokenizer_tgt.encode(tgt_text).ids

        # Pad the sentence to reach the seq_len
        src_num_padding = self.seq_len - len(src_token_ids) - 2         # -2 for the [START] and [END] tokens
        tgt_num_padding = self.seq_len - len(src_token_ids) - 1         # -1 because in decoder, we only add the [START] token, it has to generate token [END] by itself

        if src_num_padding < 0 or tgt_num_padding < 0:
            raise ValueError('Sentence is too long')
        
        # Add SOS and EOS token to the encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_token_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * src_num_padding, dtype=torch.int64)
            ]
        )

        # Add SOS token to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_token_ids, dtype=torch.int64),
                torch.tensor([self.pad_token] * tgt_num_padding, dtype=torch.int64)
            ]
        )

        # Add EOS token to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(tgt_token_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * tgt_num_padding, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, 
            "decoder_input": decoder_input, 
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),      # (1, 1, seq_len). Mask the PAD tokens from interacting with other tokens via self-attention.
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),      # (1, 1, seq_len) & (1, seq_len, seq_len). causal_mask = (1, seq_len, seq_len): each word can only look at previous words
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0