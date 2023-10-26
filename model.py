import math
import resource
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        # Params:
        # - d_model: the size of the 1D embedding vector. It's 512 in the original paper
        # - vocab_size: how many words are there in the vocabulary
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        # Params:
        # - d_model: the size of the 1D embedding vector. It's 512 in the original paper
        # - seq_len: the maximum length of the input sentence
        # - dropout: for regularisation
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Initialise Positional Embedding matrix
        pe = torch.zeros(seq_len, d_model)

        # Calculate the PE for every position up to the maximum sequence length
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pe[:, 0::2] = math.sin(pos / (10000 ** ((2*i)/self.d_model)))
                pe[:, 1::2] = math.cos(pos / (10000 ** ((2*(i+1))/self.d_model)))

        # Add batch dimension to PE
        pe = pe.unsqueeze(0)        # pe = (1, seq_len, d_model)

        # Register this PE tensor in the buffer of this module => PE will be saved in the file along with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Params:
        # - x: token embeddings (batch_size, num_tokens, d_model)
        x = x + (self.pe[:,  :x.shape[1], :]).requires_grad(False)          # pe = (1, seq_len, d_model), where seq_len is the maximum sequence length. Because the current sequence x may not up to maximum sequence length => we just need it up to x.shape[1]
        return self.dropout(x)
    
class AddAndNorm(nn.Module):
    def __init__(self, epsilon: float = 10**-6):
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
         
    def forward(self, prev_layer, res_input):
        # Add
        x = prev_layer + res_input

        # Norm
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/math.sqrt(std + self.epsilon) + self.bias
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):            # There are only 1 hidden layer in this FC network
        # Params:
        # - d_model: input & output dimension of the FC network
        # - d_ff: dimension of the hidder layer
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)         # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)         # W2 and b2
        
    def forward(self, x):
        # Params:
        # - x: input (batch, seq_len, d_model)
        hidden_layer = torch.relu(self.w_1(x))
        output = self.w_2(self.dropout(hidden_layer))
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        # Params:
        # - d_model: the size of the 1D embedding vector. It's 512 in the original paper
        # - h: number of heads for multi-head attention
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h         # size of 1 head
        self.w_q = nn.Linear(d_model, d_model)      # Query weights
        self.w_k = nn.Linear(d_model, d_model)      # Key weights
        self.w_v = nn.Linear(d_model, d_model)      # Value weights

        self.w_o = nn.Linear(d_model, d_model)      # Multihead attention weights

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):                      # Make it static so we can call this function without instantiate an instance of this class
        d_k = q.shape[-1]
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)       # q, k = (batch_size, h, seq_len, d_k); attention_scores = (batch_size, h, seq_len, seq_len)
        
        # Apply mask
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)                   # "mask" = (batch_size, h, seq_len, seq_len). For any position that mask == 0, it will be replaced by -inf

        # Calculate the final attention scores
        attention_scores = attention_scores.softmax(dim = -1)               # attention_scores = (batch_size, h, seq_len, seq_len). Apply softmax over the last dimension.
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Return the final representation weighted by the attention scores, and the attention scores
        output = attention_scores @ v                                       # output = (batch_size, h, seq_len, d_k)    
        return output, attention_scores                                     

    
    def forward(self, x1, x2, x3, mask):
        # Params:
        # - x: input (batch_size, seq_len, d_model)
        q = self.w_q(x1)         # q: (batch_size, seq_len, d_model)
        k = self.w_k(x2)         # k: (batch_size, seq_len, d_model)
        v = self.w_v(x3)         # v: (batch_size, seq_len, d_model)
        
        # Divide q, k, v into different parts for different heads
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k)        # Keep dimension 0 (batch_size) and 1 (seq_len). Divide the dimenion 2 (d_model) into "h x d_k"
        q = q.transpose(1, 2)                                       # Transpose the dimension 1 and 2: (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k): represents better data for each head
        
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k)
        k = k.transpose(1, 2)

        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k)
        v = v.transpose(1, 2)

        multihead_attention_output, self.attention_scores = MultiHeadAttention.attention(q, k, v, mask, self.dropout)      

        # Concatenate the output of each head
        multihead_attention_output = multihead_attention_output.transpose(1, 2)         # mhead_a_output = (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k)
        shape = multihead_attention_output.shape
        multihead_attention_output.view(shape[0], shape[1], shape[2] * shape[3])        # mhead_a_output = (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, h * d_k) = (batch_size, seq_len, d_model)

        return self.w_o(multihead_attention_output)                                     # mhead_a_output (..., seq_len, d_model) * w_o (d_model, d_model) = (..., seq_len, d_model)
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.add_and_norm = nn.ModuleList([AddAndNorm() for _ in range(2)])

    def forward(self, x, src_mask):
        # Params:
        # - x: input tokens
        # - src_mask: mask to hide the interaction between padding tokens vs other tokens
        mha_output = self.self_attention_block(x, x, x, src_mask)
        output_1 = self.add_and_norm[0](mha_output, x)
        output_2 = self.feed_forward_block(output_1)
        output_2 = self.add_and_norm[1](output_2, output_1)
        return output_2

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.add_and_norm = nn.ModuleList([AddAndNorm() for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Params:
        # - x: input of the decoder
        # - encoder_output: output of the encoder
        # - src_mask: mask applied to the encoder
        # - tgt_mask: mask applied to the decoder

        mha_output = self.self_attention_block(x, x, x, tgt_mask)
        output_1 = self.add_and_norm[0](mha_output, x)
        output_2 = self.cross_attention_block(output_1, encoder_output, encoder_output, src_mask)         # (q, k, v, mask)
        output_2 = self.add_and_norm[1](output_2, output_1)
        output_3 = self.feed_forward_block(output_2)
        output_3 = self.add_and_norm[2](output_3, output_2)
        return output_3

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
    
class LastLinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        output = self.linear(x)         # x = (batch_size, seq_len, d_model) --> output = (batch_size, seq_len, vocab_size)
        output = torch.log_softmax(output, dim = -1)
        return output
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, 
                 src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, linear_layer:LastLinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear_layer = linear_layer
    
    def encode(self, src, src_mask):
        # Params:
        # - src: input sentence
        # - src_mask: mask for encoder, to mask padding tokens from interacting with other tokens
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_embedding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def linear(self, x):
        return self.linear_layer(x)         # (batch_size, seq_len, vocab_size)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, 
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Params:
    # - src_vocab_size: the vocab size of the source language
    # - tgt_vocab_size: the vocab size of the target language
    # - srq_seq_len: maximum sequence length of source sequence
    # - tgt_seq_len: maximum sequence length of target sequence
    # - d_model: size of the embedding vector
    # - N: number of encoder & decoder blocks
    # - h: number of heads
    # - d_ff: dimension of the hidden layer of the Feed Forward block

    # Create the embeddings layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional embedding layers
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout) 
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block)
        encoder_blocks.append(encoder_block)
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block)
        decoder_blocks.append(decoder_block)
    
    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create last linear layer
    last_linear_layer = LastLinearLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, last_linear_layer)

    # Initialise the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return transformer