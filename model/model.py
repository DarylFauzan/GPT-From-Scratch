import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

# Create Model Configuration
class ModelConfig:
    vocab_size: int = 50257
    n_encoder_block: int = 8
    block_size: int = 512
    num_heads: int = 16
    embd_dim: int = 512
    drop_rate: int = 0

# Create multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embd_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.c_attn = nn.Linear(embd_dim, 3 * embd_dim) # (B, T, C)
        self.c_proj = nn.Linear(embd_dim, embd_dim) # (B, T, C)

    def forward(self, x: torch.Tensor, decoder: bool = False):
        B, T, C = x.shape

        # compute the query, key, value
        x = self.c_attn(x) # (B, T, C)
        q, k, v = x.split(self.embd_dim, dim = -1)

        # reshape the tensor to perform the multiheadattention
        q = q.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2) # (B, nh, T, c/nh)
        k = k.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2) # (B, nh, T, c/nh)
        v = v.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2) # (B, nh, T, c/nh)

        # multiply q and k.T
        qk = torch.matmul(q, k.transpose(-1, -2)) / (self.embd_dim ** -0.5) # (B, nh, T, T)

        # encoder or decoder, if decoder, perform future masking
        if decoder:
            tril = torch.tril(torch.ones(T, T), diagonal = 0).to(x.device)
            qk = qk.masked_fill_(tril == 0, float("-inf"))

        qk = F.softmax(qk, dim= -1)
        output = torch.matmul(qk, v) # (B, nh, T, c/nh)
        output = output.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        output = self.c_proj(output)
        return output
    
# Create a decoder block
class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, embd_dim: int, drop_rate: int = 0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embd_dim, bias = False)
        self.attn = MultiHeadAttention(num_heads, embd_dim)
        self.mlp = nn.Sequential(*[
            nn.Linear(embd_dim, 2 * embd_dim, bias = False),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(2 * embd_dim, embd_dim, bias = False)
        ])
        self.ln_2 = nn.LayerNorm(embd_dim, bias = False)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), decoder = True)
        x = x + self.ln_2(self.mlp(x))
        return x
    
# Create Encoder Transformer Model
class JeroGPT(nn.Module):
    def __init__(self, config):
        """This is a GPT Model I created myself. I name it based on my cat's name :D. 
        """
        super().__init__()
        # fetch the parameters from the config
        self.vocab_size = config.vocab_size
        self.n_encoder_block = config.n_encoder_block
        self.block_size = config.block_size
        self.num_heads = config.num_heads
        self.embd_dim = config.embd_dim
        self.drop_rate = config.drop_rate

        # define decoder for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained("tokenizer/local_gpt2_tokenizer")

        # initiate model
        self.wte = nn.Embedding(self.vocab_size, self.embd_dim)
        self.wpe = nn.Embedding(self.block_size, self.embd_dim)
        self.h = nn.ModuleList(
            DecoderBlock(self.num_heads, self.embd_dim, self.drop_rate) for _ in range(self.n_encoder_block)
        )
        self.ln = nn.Linear(self.embd_dim, self.vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """The input size will be (B, T). B is the number of batch and T is the block size."""

        #get the x shape
        B, T = x.shape

        # create the embeddings and add with positional encoding
        token_embd = self.wte(x) # (B, T, embd_dim)
        pos_embd = self.wpe(torch.arange(start = 0, end = T, device = x.device)) # (T, embd_dim)
        x = token_embd + pos_embd # (B, T, embd_dim)
        for block in self.h:
            x = block(x) # (B, T, C)
        logits = self.ln(x) # (B, T, C)

        loss = None
        if y is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(-1, C), y.view(-1)) # compute the loss function
        return logits, loss

    def generate(self, text: str, max_gen_len = 30, temperature = 1, top_k: int = None, top_p: int = None, device = "cpu"):
        """
        This is a function to perform text generation. The function provide parameters such as temperature, top_k, and top_p.
        The input to this model is a string."""
        # encode the text
        assert (top_k is None) or (top_p is None), ValueError("can only pick either top_k or top_p")
        tokens = self.tokenizer.encode(text)
        x = torch.Tensor(tokens).type(torch.long).view(1, -1).to(device)
        output = []

        # perform text generation
        while (x[:, -1] != 50256) and (len(output) < max_gen_len):
            logits = self(x[:, -self.block_size:])
            # fetch only the last logits
            logits = logits[:, -1, :]
            # apply softmax
            prob = F.softmax(logits/(temperature + 1e-6), dim = -1)
            
            # sort the probability descending
            sort_prob, indices = torch.sort(prob, dim = -1, descending = True)
            # apply top_k sampling
            if top_k is not None:
                # get the only top_k prob
                sort_prob, indices = sort_prob[:, :top_k], indices[:, :top_k]
                # perform softmax in the sorted prob
                sort_prob = F.softmax(sort_prob/ (temperature + 1e-6), dim = -1)
            if top_p is not None:
                cdf = 0
                for p, j in enumerate(sort_prob[0]):
                    cdf += j.item()
                    if cdf >= top_p:
                        break
                sort_prob, indices = sort_prob[:, :p], indices[:, :p]
                # perform softmax in the sorted prob
                sort_prob = F.softmax(sort_prob/ (temperature + 1e-6), dim = -1)
                
            # sample the tokens
            ind_next = torch.multinomial(sort_prob, num_samples = 1)
            # append x and the output
            tkn = indices[:, ind_next]
            x = torch.cat((x, tkn.view(1, -1)), dim = -1)
            output.append(tkn.item())

        # return decode the output
        return self.tokenizer.decode(output)