# train_custom_llm.py
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. HYPERPARAMETERS (Adjust based on your local GPU)
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 1500  # Increase for better accuracy
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4

print(f"🚀 Training custom layer using device: {DEVICE}")

# 2. TOKENIZER SETUP (Character-level for simplicity from scratch)
CHARS = " \nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[],.:?!"
VOCAB_SIZE = len(CHARS)
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}
encode = lambda s: [STOI[c] for c in s if c in STOI]
decode = lambda l: ''.join([ITOS[i] for i in l])


# 3. TRANSFORMER BLOCKS DEFINITION
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(N_EMBD, N_EMBD)
        self.query = nn.Linear(N_EMBD, N_EMBD)
        self.value = nn.Linear(N_EMBD, N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        q = self.query(x).view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = self.value(x).view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.shape[-1] ** 0.5))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyNurseLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# 4. TRAINING SIMULATION LOOP
if __name__ == "__main__":
    # Mock conversation strings matching your hybrid format for testing
    training_dialogue = (
                            "[SYS] ID:0 [USER] i feel sick [NURSE] Take plenty of rest and drink hot tea. [END]\n"
                            "[SYS] ID:2 [USER] head is spinning [NURSE] Please sit down immediately, keep steady, and avoid fast head moves. [END]\n"
                        ) * 100  # Multiplied to mimic a dataset stream

    data = torch.tensor(encode(training_dialogue), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]


    def get_batch():
        ix = torch.randint(len(train_data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([train_data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([train_data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)


    model = TinyNurseLanguageModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("🏋️ Training your conversational transformer model layers...")
    for iter in range(MAX_ITERS):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 300 == 0:
            print(f"Step {iter}: Loss calculation -> {loss.item():.4f}")

    # Create target folder structures dynamically
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/my_custom_nurse_weights.pth")
    print("✅ Model weights compiled successfully to 'models/my_custom_nurse_weights.pth'!")