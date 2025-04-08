import os
import torch
import torch.nn as nn
import sentencepiece as spm
import torch.nn.functional as F


# Project settings (match your training script)
project_name = "testing_2_MHA"
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


#======================================================== Hypers ========================================================
# Define model hyperparameters (match the training script)
num_layers = 2               # Number of transformer layers
d_model = 256                # Hidden dimension of the model
nhead = 8                    # Number of attention heads
dropout = 0.1                # Dropout rate
dim_feedforward = 1024       # Feedforward network dimension
max_len = 256                # Maximum sequence length


#======================================================== Hypers ========================================================



# MAIN 
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_layers=num_layers, d_model=d_model, nhead=nhead, dropout=dropout, max_len=max_len, dim_feedforward=dim_feedforward):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"


        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
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
    
            memory_seq_len = memory_emb.shape[1]
            memory_rotary_matrix = get_rotary_matrix(memory_seq_len, d_model, device)
            memory_emb = apply_rotary_embeddings(memory_emb, memory_rotary_matrix)
    
        # Decoder forward pass
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        out = self.layer_norm(out)
        return self.fc_out(out)








model = TinyTransformer(
    vocab_size=sp.get_piece_size(),  # Use the tokenizer's vocab size
    num_layers=num_layers,          # Use the predefined number of layers
    d_model=d_model,                # Use the predefined hidden dimension
    nhead=nhead,                    # Use the predefined number of attention heads
    dropout=dropout,                # Use the predefined dropout rate
    dim_feedforward=dim_feedforward # Use the predefined feedforward dimension
).to(device)

# Load the weights
final_model_path = os.path.join(project_dir, "transformer_final_model.pth")
if os.path.exists(final_model_path):
    model.load_state_dict(torch.load(final_model_path))  # Load the state dictionary
    print(f"✅ Model weights loaded successfully from: {final_model_path}")
else:
    raise FileNotFoundError(f"Model file not found at: {final_model_path}")

# Example usage
print("✅ Model is ready for inference!")


# Generation function (unchanged from your original logic)
def generate_story(model, tokenizer, prompt, max_length=250, temperature=0.6, pad_idx=0, eos_idx=None):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids.clone()

    # Use the input prompt as memory
    memory = model.embedding(input_ids)

    if eos_idx is None:
        eos_idx = tokenizer.eos_id()

    with torch.no_grad():
        for _ in range(max_length):
            output = model(generated, memory=memory)  # Pass memory to the model
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

