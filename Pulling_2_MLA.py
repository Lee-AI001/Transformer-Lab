import os
import torch
import torch.nn as nn
import sentencepiece as spm
import torch.nn.functional as F


# Project settings (match your training script)
project_name = "testing_2_MLA"
base_dir = r"C:\Users\dell\Desktop\AI\Lantern.ai"
project_dir = os.path.join(base_dir, project_name)
SAVE_PATH = project_dir

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer
tokenizer_model_path = os.path.join(SAVE_PATH, "movie_tokenizer.model")
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model_path)
print("✅ Tokenizer loaded successfully with vocab size:", sp.get_piece_size())





#======================================================== Hypers ========================================================
# Define model hyperparameters (match the training script)
num_layers = 2               # Number of transformer layers
d_model = 256                # Hidden dimension of the model
nhead = 8                    # Number of attention heads
dropout = 0.1                # Dropout rate
dim_feedforward = 1024       # Feedforward network dimension
max_len = 256                # Maximum sequence length
d_c = 32                     # KV compression dimension
d_qc = 32                    # Query compression dimension
d_rh = 16                    # RoPE dimension
vocab_size = 10000           # Vocabulary size
#======================================================== Hypers ========================================================



# RoPE
def get_rotary_matrix(seq_len, d_model, device):
    theta = 10000.0 ** (-2.0 * (torch.arange(0, d_model, 2, device=device).float()) / d_model)
    positions = torch.arange(seq_len, device=device).float().unsqueeze(1)
    angles = positions * theta
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    rotary_matrix = torch.stack([cosines, -sines, sines, cosines], dim=-1).view(seq_len, d_model // 2, 2, 2)
    return rotary_matrix

def apply_rotary_embeddings(x, rotary_matrix):
    batch_size, seq_len, d_model = x.shape
    x_ = x.view(batch_size, seq_len, d_model // 2, 2)  # Reshape input tensor
    rotary_matrix = rotary_matrix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # Broadcast rotary_matrix

    # Perform einsum operation
    x_rotated = torch.einsum('bsdh,bsdij->bsdh', x_, rotary_matrix)

    # Reshape back to original dimensions
    x_rotated = x_rotated.reshape(batch_size, seq_len, d_model)
    return x_rotated

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, nhead, d_c, d_qc, d_rh, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_h = d_model // nhead
        self.d_c = d_c
        self.d_qc = d_qc
        self.d_rh = d_rh
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)
        self.W_UK = nn.Linear(d_c, nhead * self.d_h, bias=False)
        self.W_UV = nn.Linear(d_c, nhead * self.d_h, bias=False)
        self.W_KR = nn.Linear(d_model, d_rh, bias=False)
        self.W_DQ = nn.Linear(d_model, d_qc, bias=False)
        self.W_UQ = nn.Linear(d_qc, nhead * self.d_h, bias=False)
        self.W_QR = nn.Linear(d_qc, nhead * d_rh, bias=False)
        self.W_O = nn.Linear(nhead * self.d_h, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W_DKV.weight)
        nn.init.xavier_uniform_(self.W_UK.weight)
        nn.init.xavier_uniform_(self.W_UV.weight)
        nn.init.xavier_uniform_(self.W_KR.weight)
        nn.init.xavier_uniform_(self.W_DQ.weight)
        nn.init.xavier_uniform_(self.W_UQ.weight)
        nn.init.xavier_uniform_(self.W_QR.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(self, query, key, value, tgt_mask=None, tgt_key_padding_mask=None):
        batch_size, seq_len, _ = query.shape
        device = query.device
        c_KV = self.W_DKV(key)
        k_C = self.W_UK(c_KV).view(batch_size, seq_len, self.nhead, self.d_h).transpose(1, 2)
        k_R = self.W_KR(key)
        rotary_matrix = get_rotary_matrix(seq_len, self.d_rh, device)
        k_R = apply_rotary_embeddings(k_R, rotary_matrix).unsqueeze(1).expand(-1, self.nhead, -1, -1)
        keys = torch.cat([k_C, k_R], dim=-1)
        v_C = self.W_UV(c_KV)
        values = v_C.view(batch_size, seq_len, self.nhead, self.d_h).transpose(1, 2)
        c_Q = self.W_DQ(query)
        q_C = self.W_UQ(c_Q).view(batch_size, seq_len, self.nhead, self.d_h).transpose(1, 2)
        q_R = self.W_QR(c_Q).view(batch_size, seq_len, self.nhead, self.d_rh)
        q_R = apply_rotary_embeddings(q_R.view(batch_size * self.nhead, seq_len, self.d_rh), rotary_matrix)
        q_R = q_R.view(batch_size, self.nhead, seq_len, self.d_rh)
        queries = torch.cat([q_C, q_R], dim=-1)
        scaling = (self.d_h + self.d_rh) ** -0.5
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        if tgt_mask is not None:
            attn_scores = attn_scores + tgt_mask
        if tgt_key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(tgt_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.nhead * self.d_h)
        output = self.W_O(attn_output)
        return output

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_c, d_qc, d_rh, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(d_model, nhead, d_c, d_qc, d_rh, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask, tgt_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.feedforward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_layers=num_layers, d_model=d_model, nhead=nhead, d_c=d_c, d_qc=d_qc, d_rh=d_rh, dropout=dropout, max_len=max_len, dim_feedforward=dim_feedforward):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert d_rh % 2 == 0, "d_rh must be even for RoPE"
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([
            CustomDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                d_c=d_c,
                d_qc=d_qc,
                d_rh=d_rh,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.fc_out.weight)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_padding_mask=None):
        batch_size, seq_len = tgt.shape
        device = tgt.device
    
        # Embedding for target sequence
        tgt_emb = self.embedding(tgt)
    
        # RoPE: Compute rotary matrix for current sequence length
        rotary_matrix = get_rotary_matrix(seq_len, d_model, device)
        tgt_emb = apply_rotary_embeddings(tgt_emb, rotary_matrix)
    
        # Memory handling: Use tgt as memory if none provided
        if memory is None:
            memory_emb = tgt_emb
        else:
            # Check if memory is already embedded
            if memory.dtype == torch.long or memory.dtype == torch.int:
                memory_emb = self.embedding(memory)
            else:
                memory_emb = memory  # Assume memory is already embedded
    
            memory_seq_len = memory.shape[1]
            memory_rotary_matrix = get_rotary_matrix(memory_seq_len, d_model, device)
            memory_emb = apply_rotary_embeddings(memory_emb, memory_rotary_matrix)
    
        # Pass through custom decoder layers
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory_emb, tgt_mask, tgt_padding_mask)
    
        out = self.layer_norm(tgt_emb)
        return self.fc_out(out)

model = TinyTransformer(
    vocab_size=vocab_size,  # Use the defined vocab size
    num_layers=num_layers,  # Use the defined number of layers
    d_model=d_model,        # Use the defined hidden dimension
    nhead=nhead,            # Use the defined number of attention heads
    d_c=d_c,                # Use the defined KV compression dimension
    d_qc=d_qc,              # Use the defined Query compression dimension
    d_rh=d_rh,              # Use the defined RoPE dimension
    dropout=dropout,        # Use the defined dropout rate
    dim_feedforward=dim_feedforward  # Use the defined feedforward dimension
).to(device)

# Load the weights
final_model_path = os.path.join(project_dir, "transformer_final_model.pth")
if os.path.exists(final_model_path):
    model.load_state_dict(torch.load(final_model_path))  # Load the state dictionary
    print(f"✅ Model weights loaded successfully from: {final_model_path}")
else:
    raise FileNotFoundError(f"Model file not found at: {final_model_path}")


# Generation function (unchanged from your original logic)
def generate_story(model, tokenizer, prompt, max_length=250, temperature=0.6, pad_idx=0, eos_idx=None):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids.clone()

    if eos_idx is None:
        eos_idx = tokenizer.eos_id()

    with torch.no_grad():
        for _ in range(max_length):
            output = model(generated)
            next_token_logits = output[:, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            next_token = next_token.detach().view(-1, 1).to(device)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == pad_idx or (eos_idx is not None and next_token.item() == eos_idx):
                break

    generated_text = tokenizer.decode(generated.squeeze(0).cpu().tolist())
    return generated_text

# Input query from user
query = input("Enter your story prompt: ")

# Generate output
output = generate_story(
    model,
    sp,
    query,
    max_length=250,
    temperature=0.9,
    pad_idx=sp.pad_id()
)

# Print the result
print("\nGenerated Story:")
print(output)

