# Bits per Row metric

i need to create a bits per row metric to evaluate model performance on tokenized sequences across different tokenization strategies. this will help us understand the trade-offs between concept tokens, BPE tokens, and numeric tokens in terms of information density and model efficiency.

you need to look at tokenizer.py and i've copied the code of two model files below

# models.
assume that models come in two types discrete and multivariate. discrete delivers B,T token id prob. continuous can be made to deliver B,T token id probs and B,T log prob of centered scale values.


## special tokens
should i mask any special tokens?
sos, eos -> keep eos and mask sos?
class derived special tokens -> keep we need to know how well the model is doing on these tokens. they are important for the model to learn.
Q tokens -> numeric handling as below, but how to capture that the model has learned "a number comes next"
<NUM> tokens -> in case of continuous factored, this is "a num comes next"


# Numeric tokens
this is by far the hardest. we need to convert the Q to prob density from mass.
contious provides a log prob (neg?) of the value, but this is the centered and scaled value 
level tokens throw a wrench as well. how to turn this into a density? how can we compare something that handles a numeric class as levels vs. quantiles?

## categorical tokens
i think these are the easiest. add up loss across categorical tokens in a row, normalize by rows in the batch comnputation


## rgular model

```python

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, fields
import inspect
import math
from .registry import register_model


@dataclass
class Config:
    total_batch_size: int = 1024 # total training step batch size in tokens per batch
    batch_size: int = 8          # minibatch size in sequences per minibatch
    block_size: int = 16         # context window
    vocab_size: int = 18
    n_embd: int = 32
    n_heads: int = 2
    n_layer: int = 2
    max_iters: int = 1000
    eval_iters: int = 100
    eval_interval: int = 200
    learning_rate: float = 1e-3
    warmup_iters: int = 1000
    lr_decay_iters: int = 4000
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad: bool = True
    weight_decay: float = 0.05
    eps: float = 1e-8
    device: str = 'cpu'
    dropout: int = 0
    bias: int = 0

class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd
        bias = config.bias
        self.c_fc = nn.Linear(n_embd,4*n_embd,bias = bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*n_embd,n_embd,bias = bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd
        n_heads = config.n_heads
        block_size = config.block_size
        bias = config.bias
        assert n_embd % n_heads == 0
        # do key, query, value projections for all heads in a batch instead of splitting
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = config.dropout
        # flash attention only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x

@register_model("gpt2_v1")
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        if isinstance(config, dict):
            cfg_keys = {f.name for f in fields(Config)}
            filtered_config = {k: v for k, v in config.items() if k in cfg_keys}
            config = Config(**filtered_config)
        
        self.config = config
        # Base GPT model (from NanoGPT)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # Next-token prediction head

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence too long"

        # Token and positional embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(t, device=device))
        x = tok_emb + pos_emb

        for layer in self.transformer.h:
            x = layer(x)

        x = self.transformer.ln_f(x)

        # Next-token prediction
        logits = self.lm_head(x)

        # Loss calculation
        loss = None
        if targets is not None:
            # Standard next-token prediction loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_mask(self, idx, targets=None, mask_token_ids=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence too long"
        
        # Token and positional embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(t, device=device))
        x = tok_emb + pos_emb
        for layer in self.transformer.h:
            x = layer(x)
        x = self.transformer.ln_f(x)
        
        # Next-token prediction
        logits = self.lm_head(x)
        
        # Loss calculation
        loss = None
        if targets is not None and mask_token_ids is not None:
            # Flatten
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            # Create mask: True for tokens we WANT to include
            mask = torch.ones_like(targets_flat, dtype=torch.bool)
            for mtid in mask_token_ids:
                mask = mask & (targets_flat != mtid)
            
            # Compute loss only on non-masked tokens
            loss = F.cross_entropy(logits_flat[mask], targets_flat[mask], reduction='mean')
        
        return loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """
        Note that due to weight tying the token embeddings (will be) included as non-embedding
        parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        embedding = self.transformer.wpe.weight.numel()
        non_embedding = n_params - embedding

        return non_embedding,embedding
    
    def get_model_size(self):
        # return model size in MB
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def configure_optimizer(self):
    
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in self.config.device
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(self.config.beta1,self.config.beta2), **extra_args)

        return optimizer
    
    @torch.no_grad()
    def estimate_loss(self,dl_train,dl_val):
        # compute train and val loss averaged over multiple iterations
        # more stable than single batch loss
        # note this has been modified to repeat same minibatch for train. moving to lighter version
        out = {}
        self.eval() # no need to track gradients
        for split in ['train','val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                # don't waste a training data batch just to get the loss
                # but advance through the val batches or else we'd just repeat
                if split =='train':
                    xib,yib = dl_train.batch_no_step()
                else:
                    xib,yib = dl_val.next_batch()
                _,loss = self(xib,yib)
                losses[k] = loss.item()
            out[split] = {}
            out[split]['total'] = losses.mean()
        self.train()
        return out

    @torch.no_grad()
    def estimate_loss_mask(self, dl_train, dl_val, mask_token_ids=None):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                if split == 'train':
                    xib, yib = dl_train.get_batch_no_step()
                else:
                    xib, yib = dl_val.get_batch()
                loss = self.forward_mask(xib, yib, mask_token_ids=mask_token_ids)
                losses[k] = loss.item()
            out[split] = {}
            out[split]['total'] = losses.mean()
        self.train()
        return out
    
    @torch.no_grad()
    def estimate_loss_light(self,dl_train,dl_val):
        # compute train loss over one minibatch and val loss over eval_iters
        out = {}
        self.eval() # no need to track gradients
        
        losses = torch.zeros(self.config.eval_iters)
        for k in range(self.config.eval_iters):

            xib,yib = dl_val.next_batch()
            _,loss = self(xib,yib)
            losses[k] = loss.item()
            out['val'] = {}
            out['val']['total'] = losses.mean()

        xib,yib = dl_train.batch_no_step()
        _,loss = self(xib,yib)
        out['train'] = {}
        out['train']['total'] = loss

        self.train()
        return out
    
    def _generate(self, idx, max_new_tokens):
        # under the hood function. returns bpe indices

        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.config.block_size:] # crop to last block size number of tokens
            # sample the model with the current input
            logits,_ = self(idx_cond) # (B,T,C)
            logits = logits[:,-1,:] # becomes (B,C)
            probs = F.softmax(logits,dim=-1) # (B, C)
            #TODO: top-k, temperature
            idx_next = torch.multinomial(probs,num_samples=1) # (B, 1)
            # append next token, new age, new enc
            idx = torch.cat((idx,idx_next),dim=1) # (B, T+1)


        return idx
    
    def generate(self, idx, max_new_tokens):
        # wrapper for uniform API with other models
        return self._generate(idx, max_new_tokens)

    def sample_all_positions(self, idx):
        # returns sampled next idx across a full batch
        idx_cond = idx[:,-self.config.block_size:] # crop to last block size number of tokens
        # sample the model with the current input
        logits,_ = self(idx_cond) # (B,T,C)
        probs = F.softmax(logits,dim=-1) # (B, T, C)
        idx_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)  # (B*T, 1)
        idx_next = idx_next.view(probs.size(0), probs.size(1), 1)  # Reshape to (B, T, 1)
        return idx_next.squeeze(-1)  # Return (B, T)
    
    def sample(self,tok,seed_idx=0,max=50):
        seed_idx = torch.tensor([seed_idx],dtype=torch.int64).to(self.config.device)
        tokens = self._generate(seed_idx,max)
        tokens = tokens[0].tolist()
        text = tok.decode(tokens)
        return tokens,text
    
    @classmethod
    def load_from_checkpoint(cls,ckpt_path,device_override=None):
        # load from checkpoint

        checkpoint = torch.load(ckpt_path, map_location='cpu') # always load to cpu first. load to gpu != same memory as run on GPU
        config = checkpoint['model_config']
        config = Config(**config)

        if device_override is not None:
            config.device = device_override

        model = cls(config)
        model = model.to(config.device)

        state_dict = checkpoint['model']
        # if running with torch.compile, need to remove a prefix
        compile_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(compile_prefix):
                state_dict[k[len(compile_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


```

## continous model

```python

# same as v1 but does not materialize all mean/variances because they are not needed

import torch
import math
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from dataclasses import dataclass, fields
import inspect
import sys
from .registry import register_model



@dataclass
class GPTConfig:
    total_batch_size: int = 1024 # total training step batch size in tokens per batch
    batch_size: int = 8          # minibatch size in sequences per minibatch
    block_size: int = 16         # context window
    n_embd: int = 32             # embedding dimension
    n_heads: int = 2             # number of attention heads
    n_layer: int = 2             # number of transformer layers
    max_iters: int = 1000        # total number of training iterations
    eval_iters: int = 100        # number of evaluation iterations
    eval_interval: int = 200     # evaluation interval
    learning_rate: float = 1e-3  # learning rate
    warmup_iters: int = 1000     # number of warmup iterations
    lr_decay_iters: int = 4000   # number of iterations for learning rate decay
    beta1: float = 0.9           # adam optimizer beta1
    beta2: float = 0.95          # adam optimizer beta2
    clip_grad: bool = True       # clip gradients
    weight_decay: float = 0.05   # adam optimizer weight decay
    eps: float = 1e-8            # adam optimizer epsilon
    device: str = 'cpu'          # device to train on (e.g. 'cuda' or 'cpu')
    dropout: int = 0             # dropout rate
    bias: int = 0                
    vocab_size: int = 512        # size of the vocabulary
    n_head_blocks: int = 0       # extra transformer blocks between backbone and heads (0 = none, backward-compatible)


# compute the log probability loss based on gaussian
# model predicts both loc and scale
# [B,T,C],[B,T,C],[B,T],[B,T] -> [B,T]
def gaussian_loss(x_v_l,x_v_s,t_c,t_v,fit_scale=False):
    mask = ~torch.isnan(t_v)
    loc = torch.gather(x_v_l,-1,t_c.unsqueeze(-1))
    scale = torch.gather(x_v_s,-1,t_c.unsqueeze(-1))
    if fit_scale:
        normal_dist = torch.distributions.normal.Normal(loc.squeeze(-1),scale.squeeze(-1))
    else:
        normal_dist = torch.distributions.normal.Normal(loc.squeeze(-1),torch.tensor(2).to(t_v.device))
    temp = torch.where(mask,t_v,torch.zeros_like(t_v))
    # if it's a categorical just give the peak of a gaussian with sigma=1
    scalar_val = torch.log(torch.tensor(1 / (math.sqrt(2 * math.pi)), device=t_v.device, dtype=torch.float32))
    return torch.where(mask,normal_dist.log_prob(temp),scalar_val)


# efficient variant: loc and scale are already gathered [B,T] — no full [B,T,C] materialization
# [B,T],[B,T],[B,T] -> [B,T]
def gaussian_loss_efficient(loc,scale,t_v,fit_scale=False):
    mask = ~torch.isnan(t_v)
    if fit_scale:
        normal_dist = torch.distributions.normal.Normal(loc,scale)
    else:
        normal_dist = torch.distributions.normal.Normal(loc,torch.tensor(2).to(t_v.device))
    temp = torch.where(mask,t_v,torch.zeros_like(t_v))
    scalar_val = torch.log(torch.tensor(1 / (math.sqrt(2 * math.pi)), device=t_v.device, dtype=torch.float32))
    return torch.where(mask,normal_dist.log_prob(temp),scalar_val)


# mutlivariate embedding: B,T input
# maps class,value to vector of embedding dim B,T -> B,T,E
# embedding = class_emb + val_emb; class_emb is standard lookup, val_emb is single linear layer if val is not nan
class MVEmbedding(nn.Module):

    def __init__(self,vocab_size,n_embd):
        super().__init__()
        self.ce = nn.Embedding(vocab_size,n_embd)
        self.vw = nn.Embedding(vocab_size,n_embd) # each row are the weights of a linear layer mapping dim_val to dim_embed
        self.vb = nn.Embedding(vocab_size,n_embd) # each row is a bias of dim_embed
        self.gelu = nn.GELU()

    def forward(self,c,v):
        # B,T -> B,T,E
        x = self.ce(c) # B,T,E
        # we can get away with element wise mult here because (1,1) @ (1,E) matmul is the same as element wise
        pv = self.gelu(v.unsqueeze(-1)*self.vw(c) + self.vb(c)) # B,T,E 
        mask = torch.isnan(v) # B,T
        mask = mask.unsqueeze(-1).expand_as(pv) #B,T,E
        x = torch.where(mask,x, x+pv)
        return x

class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd
        bias = config.bias
        self.c_fc = nn.Linear(n_embd,4*n_embd,bias = bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*n_embd,n_embd,bias = bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd
        n_heads = config.n_heads
        block_size = config.block_size
        bias = config.bias
        assert n_embd % n_heads == 0
        # do key, query, value projections for all heads in a batch instead of splitting
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = config.dropout
        # flash attention only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x


@register_model('multivariategpt_v2')
class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()

        # most of the time we're making from yaml so allow for easy dict -> config
        if isinstance(config, dict):
            cfg_keys = {f.name for f in fields(GPTConfig)}
            filtered_config = {k: v for k, v in config.items() if k in cfg_keys}
            config = GPTConfig(**filtered_config)

        self.config = config

        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.fit_scale = False

        self.transformer = nn.ModuleDict(dict(
            mve = MVEmbedding(self.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.head_blocks = nn.ModuleList([Block(config) for _ in range(config.n_head_blocks)])
        if config.n_head_blocks >0:
            self.ln_nh = nn.LayerNorm(config.n_embd)
        self.c_head = nn.Linear(config.n_embd,self.vocab_size,bias=False)
        self.v_head_l = nn.Linear(config.n_embd,self.vocab_size,bias=False)
        self.v_head_s = nn.Linear(config.n_embd,self.vocab_size,bias=False)

        # could do weight tying from mve.ce.weights with c_head. should try this out.

        self.apply(self._init_weights)
        # gpt-2 applies a different scaling to the residual projections. find these by name
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,mean=0.0, std=0.02/math.sqrt(2*config.n_layer))


    @classmethod
    def load_from_checkpoint(cls,ckpt_path,device_override=None):
        # load from checkpoint

        # checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=True) # always load to cpu first. load to gpu != same memory as run on GPU
        config = checkpoint['model_config']
        config = GPTConfig(**config)

        if device_override is not None:
            config.device = device_override

        model = cls(config)
        model = model.to(config.device)

        state_dict = checkpoint['model']
        # if running with torch.compile, need to remove a prefix
        compile_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(compile_prefix):
                state_dict[k[len(compile_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,c,v,t_c=None,t_v=None):
        # input: B,T
        # training:  x_v_l, x_v_s are [B,T] loc/scale for the true next token's class
        # inference: x_v_l, x_v_s are [B,T,C] loc/scale for all classes
        device = c.device
        B,T = c.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)
        cv_emb = self.transformer.mve(c,v) # B,T,E
        pos_emb = self.transformer.wpe(pos) # B,T,E
        x = cv_emb+pos_emb # B,T,E
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # B,T,E
        x_c = self.c_head(x) # B,T,C

        # run the latents through additional blocks before the value heads
        if self.head_blocks:
            for block in self.head_blocks:
                x = block(x)
            x = self.ln_nh(x)

        if t_v is not None and t_c is not None:
            # efficient path: gather weight rows for target classes, then dot with x
            # avoids materializing full [B,T,C] tensors for v_head_l and v_head_s
            w_l = self.v_head_l.weight[t_c]              # B,T,E
            w_s = self.v_head_s.weight[t_c]              # B,T,E
            loc = (x * w_l).sum(-1)                      # B,T
            scale = torch.exp((x * w_s).sum(-1))         # B,T
            v_loss = gaussian_loss_efficient(loc,scale,t_v,self.fit_scale) # log probability
            c_loss = F.cross_entropy(x_c.view(-1,x_c.size(-1)),t_c.view(-1),reduction='none',ignore_index=-1) # negative log likelihood
            loss = -v_loss.view(-1)+c_loss
            loss = torch.mean(loss)
            c_loss = torch.mean(c_loss)
            v_loss = -torch.mean(v_loss)
            x_v_l = loc    # B,T — loc for true next token class
            x_v_s = scale  # B,T — scale for true next token class
        else:
            # inference path: materialize full distributions for sampling
            x_v_l = self.v_head_l(x) # B,T,C
            x_v_s = torch.exp(self.v_head_s(x)) # B,T,C
            loss = None
            c_loss = None
            v_loss = None

        return x_c,x_v_l,x_v_s,c_loss,v_loss,loss
    
    def forward_mask(self,c,v,t_c=None,t_v=None,mask_token_ids=None):
        # compute forward pass and return loss masking out loss from certain token ids in input
        # input: B,T
        # output: B,T,C x 2 -> loss function -> scalar loss
        device = c.device
        B,T = c.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)
        cv_emb = self.transformer.mve(c,v) # B,T,E
        pos_emb = self.transformer.wpe(pos) # B,T,E
        x = cv_emb+pos_emb # B,T,E
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # B,T,E

        x_c = self.c_head(x) # B,T,C

        if self.head_blocks:
            for block in self.head_blocks:
                x = block(x)
            x = self.ln_nh(x)

        if t_v is not None and t_c is not None:
            # only compute class loss on non-masked tokens
            if mask_token_ids is not None:
                xc = x_c.view(-1,x_c.size(-1))
                tc = t_c.view(-1)
                mask = torch.ones_like(tc,dtype=torch.bool)
                for mtid in mask_token_ids:
                    mask = mask & (tc != mtid)
                c_loss = F.cross_entropy(xc[mask],tc[mask],reduction='mean',ignore_index=-1)        
        else:
            c_loss = None

        return c_loss


    def sample_target(self,c,v,t_c):
        # return the probability of the target class and a value sampled from the target class

        # sample value from the target class
        c_cond = c[:, -self.block_size:]
        v_cond = v[:, -self.block_size:]
        
        x_c, x_v_l, x_v_s, _, _, _ = self(c_cond, v_cond)  # (B, T, C)
        
        # get the next class
        probs = F.softmax(x_c, dim=-1)  # (B, T, C)
            
        # get the loc and scale from the target class
        loc = torch.gather(x_v_l,-1,t_c.unsqueeze(-1))
        scale = torch.gather(x_v_s,-1,t_c.unsqueeze(-1))
        p_target = torch.gather(probs,-1,t_c.unsqueeze(-1)) # val loss essentially

        # sample v
        if self.fit_scale:
            v_target = torch.normal(mean=loc.squeeze(-1), std=scale.squeeze(-1))  # (B, T)
        else:
            v_target = loc.squeeze(-1)  # (B, T)
        
        return p_target.squeeze(-1), v_target

    
    def generate(self,c,v,max_new_tokens,mask_value_ids=None):

        for _ in range(max_new_tokens):

            # make sure fits pos embedding
            c_cond = c[:,-self.block_size:]
            v_cond = v[:,-self.block_size:]
            x_c,x_v_l,x_v_s,_,_,_ = self(c_cond,v_cond) # (B,T,C)

            # sample the model with the current input
            x_c = x_c[:,-1,:] # becomes (B,C) with last time prediction
            x_v_l = x_v_l[:,-1,:] # becomes (B,C) with last time prediction
            x_v_s = x_v_s[:,-1,:] # becomes (B,C) with last time prediction
            probs = F.softmax(x_c,dim=-1) # (B, C)
            #TODO: top-k, temperature
            c_next = torch.multinomial(probs,num_samples=1) # (B, 1)
            loc_next = torch.gather(x_v_l,-1,c_next)
            scale_next = torch.gather(x_v_s,-1,c_next)
            if self.fit_scale:
                v_next = torch.normal(mean=loc_next, std=scale_next)
            else:
                v_next = loc_next
            # set v_next to nan if c_next is in mask_value_ids
            if mask_value_ids is not None:
                mask = torch.isin(c_next,torch.tensor(mask_value_ids).to(c_next.device))
                v_next = torch.where(mask,torch.tensor(float('nan')).to(v_next.device),v_next)
                

            # append next token, new age, new enc
            c = torch.cat((c,c_next),dim=1) # (B, T+1)
            v = torch.cat((v,v_next),dim=1)

        return c,v
    
    def sample_all_positions(self, c, v):
        # sample next token from a B,T input

        # make sure fits in positional embedding
        c_cond = c[:, -self.block_size:]
        v_cond = v[:, -self.block_size:]
        
        x_c, x_v_l, x_v_s, _, _, _ = self(c_cond, v_cond)  # (B, T, C)
        
        # get the next class
        probs = F.softmax(x_c, dim=-1)  # (B, T, C)
        c_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)  # (B*T, 1)
        c_next = c_next.view(probs.size(0), probs.size(1), 1)  # Reshape to (B, T, 1)
        
        # get the loc and scale from the selected class
        loc_next = torch.gather(x_v_l, -1, c_next)  # (B, T, 1)
        scale_next = torch.gather(x_v_s, -1, c_next)  # (B, T, 1)
        
        # sample v
        if self.fit_scale:
            v_next = torch.normal(mean=loc_next.squeeze(-1), std=scale_next.squeeze(-1))  # (B, T, 1)
        else:
            v_next = loc_next.squeeze(-1)  # (B, T, 1)
        
        return c_next.squeeze(-1), v_next.squeeze(-1)
    
    def sample(self,seed_c='[sos]',seed_v='',max_new_tokens=5):
        seed_tki,seed_tkv = self.tokenizer.encode([seed_c],[seed_v])
        seed_tki = torch.tensor([seed_tki],dtype=torch.int64).to(self.config.device)
        seed_tkv = torch.tensor([seed_tkv],dtype=torch.float32).to(self.config.device)
        tki,tkv = self.generate(seed_tki,seed_tkv,max_new_tokens=max_new_tokens)
        tki = tki.detach().numpy().squeeze()
        tkv = tkv.detach().numpy().squeeze()
        c,v = self.tokenizer.decode(tki,tkv)
        return c,v
    
    def get_num_params(self):
        """
        Note that due to weight tying the token embeddings (will be) included as non-embedding
        parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        embedding = self.transformer.wpe.weight.numel() + self.transformer.mve.ce.weight.numel() + self.transformer.mve.vw.weight.numel() + self.transformer.mve.vb.weight.numel()
        non_embedding = n_params - embedding

        return non_embedding,embedding
    
    def get_model_size(self):
        # return model size in MB
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def configure_optimizer(self):
    
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in self.config.device
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(self.config.beta1,self.config.beta2), **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad()
    def estimate_loss(self,dl_train,dl_val):
        # compute train and val loss averaged over multiple iterations
        # more stable than single batch loss
        out = {}
        self.eval() # no need to track gradients
        for split in ['train','val']:
            losses = torch.zeros(self.config.eval_iters)
            c_losses = torch.zeros(self.config.eval_iters)
            v_losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                # don't waste a training data batch just to get the loss
                # but advance through the val batches or else we'd just repeat
                if split =='train':
                    Xi,Xv,Yi,Yv = dl_train.batch_no_step()
                else:
                    Xi,Xv,Yi,Yv = dl_val.next_batch()
                _,_,_,c_loss,v_loss,loss = self(Xi,Xv,Yi,Yv)
                losses[k] = loss.item()
                c_losses[k] = c_loss.item()
                v_losses[k] = v_loss.item()
            out[split] = {}
            out[split]['t'] = losses.mean()
            out[split]['c'] = c_losses.mean()
            out[split]['v'] = v_losses.mean()
        self.train()
        return out
    
    @torch.no_grad()
    def estimate_loss_mask(self,dl_train,dl_val,mask_token_ids=None):
        # compute train and val loss averaged over multiple iterations
        # more stable than single batch loss
        out = {}
        self.eval() # no need to track gradients
        for split in ['train','val']:
            c_losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                # don't waste a training data batch just to get the loss
                # but advance through the val batches or else we'd just repeat
                if split =='train':
                    Xi,Xv,Yi,Yv = dl_train.batch_no_step()
                else:
                    Xi,Xv,Yi,Yv = dl_val.next_batch()
                c_loss = self.forward_mask(Xi,Xv,Yi,Yv,mask_token_ids=mask_token_ids)
                c_losses[k] = c_loss.item()
            out[split] = {}
            out[split]['c'] = c_losses.mean()
        self.train()
        return out

    def fix_unused(self):
        mask = torch.isnan(self.transformer.mve.vw.weight.grad)
        self.transformer.mve.vw.weight.grad = torch.where(mask,0,self.transformer.mve.vw.weight.grad)
        self.transformer.mve.vw.weight.grad
        mask = torch.isnan(self.transformer.mve.vb.weight.grad)
        self.transformer.mve.vb.weight.grad = torch.where(mask,0,self.transformer.mve.vb.weight.grad)
        self.transformer.mve.vb.weight.grad
        return None
  
```