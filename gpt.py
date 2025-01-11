"""
Implementation of a GPT-style transformer model with optimizations for training speed and efficiency.
This implementation follows the architecture described in the GPT-2 paper while incorporating
modern training techniques like gradient accumulation, mixed precision training, and cosine learning rate scheduling.
"""

# GPT-3 Paper 
# add cosing delay  
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


"""
The CausalSelfAttention module implements the core self-attention mechanism used in GPT models.
Key features:
- Uses Flash Attention for optimized performance on modern GPUs
- Implements causal (masked) attention to prevent looking at future tokens
- Projects input into Query, Key and Value matrices in a batched operation
- Scales attention scores by 1/sqrt(head_size) to maintain stable gradients
- Supports multi-head attention for parallel attention computations

Architecture details:
- Input is split into Q/K/V projections using a single linear layer for efficiency
- Attention scores are computed as scaled dot products between Q and K
- V is then weighted by the attention scores to produce the output
- Final projection layer combines multi-head outputs
"""
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal = True) # Flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


"""
The MLP module implements the feed-forward network component of each transformer block.
Design choices:
- Uses 4x expansion in hidden layer (standard in transformer architectures)
- GELU activation with tanh approximation for better performance
- Special initialization scaling for deeper networks
- Projection back to original dimension with careful initialization

The 4x expansion allows the network to learn more complex token interactions
while the GELU activation provides non-linearity with better gradients than ReLU.
"""
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

"""
The Block module combines self-attention and MLP layers into a transformer block.
Key features:
- Layer normalization before each sub-component (pre-norm formulation)
- Residual connections around both attention and MLP
- Careful initialization for stable training of deep networks

This implementation uses the pre-norm formulation which has been shown to enable
better training of deep transformers compared to the post-norm variant.
"""
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


"""
GPTConfig defines the architectural hyperparameters for the model.
The defaults match GPT-2 small (124M parameters):
- 12 layers with 12 attention heads
- 768 embedding dimension
- 1024 sequence length
- ~50k vocab size (GPT-2 BPE tokenizer)

These parameters can be scaled up to match larger GPT-2 variants
like medium (350M), large (774M), or XL (1.5B).
"""
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


"""
The main GPT model implementation combining all components into a full language model.
Key features:
- Token + positional embeddings shared with output layer
- Stack of transformer blocks
- Final layer norm and projection to vocab size
- Weight initialization scaled by network depth
- Optimized training with AdamW and weight decay

The model supports:
- Pretrained weight loading from Hugging Face
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpoint saving/loading

Training optimizations:
- Uses fused AdamW when available
- Separates weight decay and non-weight decay params
- Implements cosine learning rate schedule with warmup
- Gradient clipping for stability
"""
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)



    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def save_checkpoint(self, optimizer, loss, step, filename):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'step': step
        }
        torch.save(checkpoint, filename)

# model = GPT.from_pretrained('gpt2')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30



import tiktoken

"""
DataLoaderLite provides efficient batched data loading for training.
Design choices:
- Loads full dataset into memory for maximum throughput
- Uses tiktoken for fast tokenization
- Provides circular iteration through the dataset
- Returns batched inputs and shifted targets for language modeling

The loader maintains its state between batches and automatically
resets when reaching the end of the dataset.
"""
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# CHANGES IN CURRENT CODE
torch.set_float32_matmul_precision('high')
model = GPT(GPTConfig())
model.to(device)
# model = torch.compile(model)

# CODE UPDATE HERE
max_lr = 1e-3  # Increased from 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100  # Increased warmup
max_steps = 500
gradient_accumulation_steps = 4  # New parameter

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

train_loader = DataLoaderLite(B=32, T=1024)  # Increased batch size from 16 to 32

# NEW CODE
import time
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

# Initialize scaler only for CUDA
scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
best_loss = float('inf')

def print_model_summary(model):
    """Print model summary including parameter count and layer information"""
    print("\nModel Summary:")
    print("=" * 50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    print("\nLayer Details:")
    print("-" * 50)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {module.__class__.__name__} ({params:,} parameters)")
        if name == 'transformer':
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                print(f"  └─{sub_name}: {sub_module.__class__.__name__} ({sub_params:,} parameters)")
                if sub_name == 'h':
                    print("    └─Transformer Blocks:")
                    for i, block in enumerate(sub_module):
                        block_params = sum(p.numel() for p in block.parameters())
                        print(f"      └─Block {i}: ({block_params:,} parameters)")
    
    print("\nModel Architecture:")
    print("-" * 50)
    print(model)
    print("=" * 50)

print_model_summary(model)
print("\nStarting training...\n")

"""
Training loop implementation with modern optimization techniques:
- Gradient accumulation for effective larger batch sizes
- Mixed precision training using torch.cuda.amp
- Cosine learning rate schedule with warmup
- Model checkpointing based on loss
- Detailed progress monitoring and logging
- Automatic device placement (CPU/CUDA/MPS)

Performance optimizations:
- Uses high precision matrix multiplications
- Implements gradient clipping for stability
- Synchronizes CUDA operations for accurate timing
- Tracks tokens/second for throughput monitoring

The training loop provides comprehensive logging of:
- Loss values
- Learning rates
- Gradient norms
- Training speed (tokens/second)
- Time per batch
"""
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    
    # Gradient accumulation loop
    accumulated_loss = 0
    for accum_step in range(gradient_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        if device == 'cuda':
            # Use autocast only for CUDA
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / gradient_accumulation_steps
                accumulated_loss += loss.item()
                scaler.scale(loss).backward()
        else:
            # For CPU and MPS, just do regular forward pass
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()
    
    # Clip gradients
    if device == 'cuda':
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Optimizer step
    if device == 'cuda':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    # Synchronize if needed
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * gradient_accumulation_steps) / (t1 - t0)
    
    # Save best model
    if accumulated_loss < best_loss:
        best_loss = accumulated_loss
        model.save_checkpoint(optimizer, accumulated_loss, step, 'best_model.pt')
    
    if step % 10 == 0:  # Print every 10 steps
        print(f'step {step} | loss: {accumulated_loss:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f} | norm: {norm:.2f} | lr: {lr:.6f}')


print(loss)
import sys; sys.exit(0)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)