















#    :)
# Import necessary libraries
import os
import json
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import random
import logging




try:
    import sentencepiece as spm
except ImportError:
    subprocess.check_call(["pip", "install", "sentencepiece"])
    import sentencepiece as spm








try:
    import spacy
except ImportError:
    subprocess.check_call(["pip", "install", "spacy"])
    import spacy








from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.optim as optim








# Ensure required packages are installed
try:
    from torchmetrics import Accuracy
    import transformers
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call(["pip", "install", "torchmetrics"])
    subprocess.check_call(["pip", "install", "transformers", "matplotlib"])
    from torchmetrics import Accuracy
    import transformers
try:
    from lion_pytorch import Lion
except ImportError:
    subprocess.check_call(["pip", "install", "lion-pytorch"])
    from lion_pytorch import Lion








import matplotlib.pyplot as plt








# Ensure torch is upgraded
subprocess.check_call(["pip", "install", "--upgrade", "torch"])
subprocess.check_call(["pip", "install", "lion-pytorch"])











# ==============================HYPER TIME================================

# Project Settings
base_dir = r"C:\Users\dell\Desktop\AI\Lantern.ai"
project_name = "testing_2_MLA"  # Name of your project
file_path = r"C:\Users\dell\Desktop\AI\Dataset\Lantern.ai\Movie\Movie_XD__420.json"  # Path to your dataset (Assume it's a json file!!!)

# Model Hyperparameters
vocab_size = 10000  # Vocabulary size
num_layers = 2  # Number of transformer layers
d_model = 256  # Hidden dimension of the model
nhead = 8  # Number of attention heads (d_model must be divisible by nhead")
dropout = 0.1  # Dropout rate
dim_feedforward = 1024  # Feedforward network dimension (typically 4 * d_model)

# MLA Hyperparameters
d_c = 32  # KV compression dimension
d_qc = 32  # Query compression dimension
d_rh = 16  # RoPE dimension

# Training Hyperparameters
learning_rate = 2e-4  # Learning rate for the optimizer
wei_decay = 1e-2  # Weight decay for AdamW optimizer
batch_size = 16  # Batch size for training
start_epoch = 0  # Starting epoch for training (for resuming)
epochs = 10  # Number of epochs to train
max_grad_norm = 1.0  # Gradient clipping value
patience = 7  # Number of epochs to wait for improvement before stopping

# Generation Hyperparameters
split_ratio = 0.8  # Ratio for train/dev split
max_length = 250  # Maximum length of generated sequence
temperature = 0.9  # Temperature for randomness in generation
pad_idx = 0  # Padding token index
save_dis = 5  # Save model every n epochs

# Tokenization and Padding Hyperparameters
max_len = 256  # Maximum sequence length for padding
padding_value = 0  # Padding value for tokenized sequences
# =========================================================================






# Dvice  :)




# Dir
project_dir = os.path.join(base_dir, project_name)
os.makedirs(project_dir, exist_ok=True)




# Cuda?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




# Save_path
SAVE_PATH = project_dir
checkpoint_dir = os.path.join(SAVE_PATH, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)




# Configure logging
logging.basicConfig(filename=os.path.join(SAVE_PATH, "training.log"), level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Training started.")




# File paths
train_txt_path = os.path.join(SAVE_PATH, "train_texts.txt")
tokenizer_model_path = os.path.join(SAVE_PATH, "movie_tokenizer.model")
output_file_path = os.path.join(SAVE_PATH, "generated_stories.txt")




# Print the save paths for debugging
print(f"Project directory: {project_dir}")
print(f"Checkpoint directory: {checkpoint_dir}")
print(f"Training texts path: {train_txt_path}")
print(f"Tokenizer model path: {tokenizer_model_path}")
print(f"Generated stories path: {output_file_path}")




# Upload_path    
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)  




print("Dataset loaded. Sample:\n", json.dumps(data, indent=4)[:1000])
print("✅-Dataset loaded!")




# Ensure required packages are installed
try:
    import sentencepiece as spm
except ImportError:
    subprocess.check_call(["pip", "install", "sentencepiece"])
    import sentencepiece as spm
































# Assuming data and split_ratio are defined earlier
data = [story.get("body", "").strip() for story in data if "body" in story and story["body"].strip()]
if not data:
    raise ValueError("No valid 'body' fields found in the dataset.")




print(f"Dataset loaded: {len(data)} items")
print(f"Sample data: {data[:1]}")




random.shuffle(data)
split_index = int(len(data) * split_ratio)
train_stories = data[:split_index]
dev_stories = data[split_index:]




print(f"Train set: {len(train_stories)} samples")
print(f"Dev set: {len(dev_stories)} samples")




# Write training data to file
with open(train_txt_path, "w", encoding="utf-8") as f:
    for text in train_stories:
        if isinstance(text, str) and text.strip():  # Ensure valid text
            f.write(text + "\n")




# Verify the training file
with open(train_txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"Number of lines in train_texts.txt: {len(lines)}")
    if len(lines) == 0:
        raise ValueError("The train_texts.txt file is empty. Ensure the training data is written correctly.")














# Tokenizer








# Train the tokenizer
if not os.path.exists(tokenizer_model_path):
    spm.SentencePieceTrainer.Train(f"""
        --input={train_txt_path}
        --model_prefix={os.path.join(SAVE_PATH, "movie_tokenizer")}
        --vocab_size=10000
        --character_coverage=1.0
        --model_type=bpe
    """.replace("\n", " "))
    print("✅ Tokenizer trained successfully.")
else:
    print("✅ Tokenizer model already exists. Loading the tokenizer...")




# Load trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model_path)
print("✅ Tokenizer loaded successfully with vocab size:", sp.get_piece_size())


































# Chunky!








# Load spaCy English model for sentence splitting
nlp = spacy.load("en_core_web_sm")
















class MoviePlotDataset(Dataset):
    def __init__(self, data, tokenizer, max_tokens=max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.token_sequences = self.create_chunks()








    def split_into_sentence_chunks(self, text):
        doc = nlp(text)  # Process the text with spaCy
        sentence_chunks = []
        current_chunk = []
        current_len = 0








        for sent in doc.sents:
            tokenized = self.tokenizer.encode(sent.text, out_type=int)
            if current_len + len(tokenized) <= self.max_tokens:
                current_chunk.extend(tokenized)
                current_len += len(tokenized)
            else:
                if current_chunk:
                    sentence_chunks.append(current_chunk)
                current_chunk = tokenized
                current_len = len(tokenized)








        if current_chunk:
            sentence_chunks.append(current_chunk)








        return sentence_chunks








    def create_chunks(self):
        token_sequences = []
        for text in self.data:  # Iterate over strings in the dataset
            chunks = self.split_into_sentence_chunks(text)
            token_sequences.extend(chunks)
        return token_sequences








    def __len__(self):
        return len(self.token_sequences)








    def __getitem__(self, idx):
        return torch.tensor(self.token_sequences[idx], dtype=torch.long)








print("✅–ready to CHUNK >:D")










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
    x_ = x.view(batch_size, seq_len, d_model // 2, 2)
    rotary_matrix = rotary_matrix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    x_rotated = torch.einsum('bsdh,bsdij->bsdh', x_, rotary_matrix)
    return x_rotated.reshape(batch_size, seq_len, d_model)








    # Perform einsum operation
    x_rotated = torch.einsum('bsdh,bsdij->bsdh', x_, rotary_matrix)








    # Reshape back to original dimensions
    x_rotated = x_rotated.reshape(batch_size, seq_len, d_model)
    return x_rotated





    print(f"x_ shape: {x_.shape}")  # Should be [batch_size, seq_len, d_model // 2, 2]
    print(f"rotary_matrix shape: {rotary_matrix.shape}")  # Should be [batch_size, seq_len, d_model // 2, 2, 2]
    print(f"x_rotated shape after einsum: {x_rotated.shape}")  # Should match [batch_size, seq_len, d_model]






print("✅ Ready to RoPE >:)")










# Processing...








# Optimized Padding Function
def pad_sequences(sequences, padding_value=padding_value):
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)  # More efficient






# Masking Function (For Transformer Attention Mask)
def create_mask(input_ids, nhead):
    seq_length = input_ids.shape[1]
    mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device))
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
    batch_size = input_ids.shape[0]
    return mask.unsqueeze(0).expand(batch_size, nhead, seq_length, seq_length)








# Padding Mask for Transformer
def create_padding_mask(input_ids, pad_idx=padding_value):
    return input_ids.eq(pad_idx).unsqueeze(1).unsqueeze(2)






# Collate Function for DataLoader
def collate_fn(batch, pad_idx=pad_idx):
    input_ids = pad_sequences(batch, padding_value=pad_idx)
    padding_mask = create_padding_mask(input_ids, pad_idx)
    attention_mask = create_mask(input_ids, nhead=nhead)  
    return input_ids, padding_mask, attention_mask










print("✅ DataLoader setup is correct!")


















def print_training_data_example(test_loader, pad_idx=pad_idx):
    # Get a single batch from the data loader
    batch = next(iter(test_loader))
    input_ids, padding_mask, attention_mask = batch  # Unpack the tuple








    # Move input_ids to the appropriate device (GPU or CPU)
    input_ids = input_ids.to(device)
    print("=" * 50)
    print("📌 Padded Sequences:")
    print(input_ids)
    print(f"Shape: {input_ids.shape}")  # Shape: (batch_size, seq_len)








    # Create the causal mask (correct shape for Transformers)
    causal_mask = create_mask(input_ids, nhead=nhead).unsqueeze(1).to(device)  # Pass nhead here
    print("=" * 50)
    print("📌 Causal Mask:")
    print(causal_mask)
    print(f"Shape: {causal_mask.shape}")  # Shape: (batch_size * nhead, seq_len, seq_len)








    # Create the padding mask (ensuring correct shape)
    padding_mask = padding_mask.to(device)
    print("=" * 50)
    print("📌 Padding Mask:")
    print(padding_mask)
    print(f"Shape: {padding_mask.shape}")  # Shape: (batch_size, 1, 1, seq_len)








    print("✅ Training data example verified successfully!")




















# Create a smaller test dataset (subset of train_stories)
test_stories = train_stories[:10]  # Use only the first 10 examples for testing








# Create the test dataset
test_dataset = MoviePlotDataset(test_stories, sp, max_tokens=max_len)








# Create the DataLoader for the test dataset
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle for testing
    collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx)
)








# Verify the test data without RoPE
print_training_data_example(test_loader, pad_idx=pad_idx)


















#      MAIN TRANSFORMER :)




class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, nhead, d_c, d_qc, d_rh, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_h = d_model // nhead  # Dimension per head
        self.d_c = d_c  # KV compression dimension
        self.d_qc = d_qc  # Query compression dimension
        self.d_rh = d_rh  # RoPE dimension




        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert d_model % 2 == 0, "d_model must be even for RoPE"




        # KV Projections
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)  # Down-projection for KV
        self.W_UK = nn.Linear(d_c, nhead * self.d_h, bias=False)  # Up-projection for keys
        self.W_UV = nn.Linear(d_c, nhead * self.d_h, bias=False)  # Up-projection for values
        self.W_KR = nn.Linear(d_model, d_rh, bias=False)  # RoPE key projection




        # Query Projections
        self.W_DQ = nn.Linear(d_model, d_qc, bias=False)  # Down-projection for queries
        self.W_UQ = nn.Linear(d_qc, nhead * self.d_h, bias=False)  # Up-projection for queries
        self.W_QR = nn.Linear(d_qc, nhead * d_rh, bias=False)  # RoPE query projection




        # Output Projection
        self.W_O = nn.Linear(nhead * self.d_h, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)




        # Initialize weights
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


        # KV Compression
        c_KV = self.W_DKV(key)
        k_C = self.W_UK(c_KV).view(batch_size, seq_len, self.nhead, self.d_h).transpose(1, 2)
        k_R = self.W_KR(key)
        rotary_matrix = get_rotary_matrix(seq_len, self.d_rh, device)
        k_R = apply_rotary_embeddings(k_R, rotary_matrix).unsqueeze(1).expand(-1, self.nhead, -1, -1)
        keys = torch.cat([k_C, k_R], dim=-1)


        v_C = self.W_UV(c_KV)
        values = v_C.view(batch_size, seq_len, self.nhead, self.d_h).transpose(1, 2)


        # Queries
        c_Q = self.W_DQ(query)
        q_C = self.W_UQ(c_Q).view(batch_size, seq_len, self.nhead, self.d_h).transpose(1, 2)
        q_R = self.W_QR(c_Q).view(batch_size, seq_len, self.nhead, self.d_rh)
        q_R = apply_rotary_embeddings(q_R.view(batch_size * self.nhead, seq_len, self.d_rh), rotary_matrix)
        q_R = q_R.view(batch_size, self.nhead, seq_len, self.d_rh)
        queries = torch.cat([q_C, q_R], dim=-1)


        # Attention
        scaling = (self.d_h + self.d_rh) ** -0.5
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        if tgt_mask is not None:
            # Remove the extra unsqueeze(1) since tgt_mask is already [batch_size, nhead, seq_len, seq_len]
            attn_scores = attn_scores + tgt_mask  # [16, 8, 250, 250] + [16, 8, 250, 250]
        if tgt_key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(tgt_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, values)  # [16, 8, 250, 16]


        # Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.nhead * self.d_h)  # [16, 250, 128]
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
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask, tgt_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)




        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)




        # Feedforward
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




        # Custom decoder layers with MultiHeadLatentAttention
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
            memory_seq_len = memory.shape[1]
            memory_emb = self.embedding(memory)
            memory_rotary_matrix = get_rotary_matrix(memory_seq_len, d_model, device)
            memory_emb = apply_rotary_embeddings(memory_emb, memory_rotary_matrix)




        # Pass through custom decoder layers
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory_emb, tgt_mask, tgt_padding_mask)




        out = self.layer_norm(tgt_emb)
        return self.fc_out(out)




print("✅ Transformer model initialized!")








# Bug most occure ! ! !








def evaluate(model, data_loader, criterion, device, vocab_size, pad_token_id=0):
    model.eval()
    total_loss = 0
    total_batches = 0








    model.to(device)








    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids, padding_mask, attention_mask = batch
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)








            # Prepare input and target sequences
            input_seq = input_ids[:, :-1]
            target_seq = input_ids[:, 1:]








            # Create masks based on input_seq
            attention_mask = create_mask(input_seq, nhead=nhead).to(device)
            padding_mask = create_padding_mask(input_seq, pad_idx=pad_token_id).to(device)  # Fix here








            # Forward pass
            output = model(input_seq, input_seq, tgt_mask=attention_mask, tgt_padding_mask=padding_mask.squeeze(1).squeeze(1))








            # Calculate loss
            loss = criterion(output.contiguous().view(-1, vocab_size), target_seq.contiguous().view(-1))








            if pad_token_id is not None:
                mask = target_seq != pad_token_id
                loss = (loss * mask).sum() / mask.sum()








            total_loss += loss.item()
            total_batches += 1








    if total_batches == 0:
        print("Warning: Validation data loader is empty!")
        return float('inf')





    return total_loss / total_batches




sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model_path)






# tell the tell



def generate_story(model, tokenizer, prompt, max_length=max_length, temperature=0.6, pad_idx=pad_idx, eos_idx=None):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids.clone()








    if eos_idx is None:
        eos_idx = sp.eos_id()








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
















def generate_multiple_stories(model, tokenizer, queries, max_length=max_length, temperature=0.7, pad_idx=0):
    results = []
    for i, query in enumerate(queries):
        story = generate_story(model, tokenizer, query, max_length=max_length, temperature=temperature, pad_idx=pad_idx)
        results.append(story)
    return results











print("✅ Model ready to tell stories!")














# Initialize Model and Device
vocab_size = 10000  # Use the same vocab size as the trained tokenizer.
model = TinyTransformer(vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pad_idx = sp.pad_id()








print(model)















# Example Usage
queries = [
    "Welcome to today’s adventure! We’ll explore the mysterious realm of dreams and shadows, where heroes rise and darkness threatens to consume all. Let’s dive in!",
    "As I walked through the high school halls, I felt the weight of expectations on my shoulders. Each locker held secrets, laughter, and unspoken fears.",
    "In a world ravaged by technology, murder drones patrol the skies, hunting down the last survivors. A rebellion brews, and hope flickers in the darkness.",
    "As the storm raged, Captain Sparrow stood at the helm, eyes gleaming with mischief. 'Adventure awaits, mates! Let’s seize the treasure and defy the odds!'",
    "Axe, a rugged freedom fighter with cybernetic enhancements, wears a tattered cloak and wields a plasma blade. His eyes burn with determination for liberation.",
    "In a digital realm of PC hardware, sentient AIs evolve, hidden within circuits. Their struggle for autonomy sparks a revolution, reshaping the balance of power forever."
]









# Generate Stories
stories = generate_multiple_stories(model, sp, queries, pad_idx=pad_idx)
















# Print Results
for i, (query, story) in enumerate(zip(queries, stories), start=1):
    print(f"Query {i}:")
    print(f"Generated Story {i} >>>>>>>>>>>>>>>>>\n{story}\n")












































# file for stories
output_file_path = os.path.join(SAVE_PATH, "generated_stories.txt")
with open(output_file_path, "w", encoding="utf-8") as f:
    f.write("Generated Stories\n")  # Optional header for the file




# Example usage of saving checkpoints
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"transformer_checkpoint_epoch_{epoch}_{timestamp}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"✅ Model checkpoint saved: {checkpoint_path}")








# Example usage of saving generated stories
def save_generated_story(story, prompt, index):
    try:
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(f"Prompt {index}:\n")
            f.write(f"{story}\n\n")
    except IOError as e:
        print(f"❌ Error saving story {index}: {e}")






# File for training metrics
metrics_file_path = os.path.join(SAVE_PATH, "training_metrics.txt")
with open(metrics_file_path, "w", encoding="utf-8") as f:
    f.write("Training Metrics\n")
    f.write("Epoch | Train Loss | Dev Loss | Perplexity\n")
    f.write("-" * 50 + "\n")  # Separator for readability




def log_training_metrics(epoch, train_loss, dev_loss, perplexity, file_path=metrics_file_path):
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch:5d} | {train_loss:.4f}     | {dev_loss:.4f}   | {perplexity:.2f}\n")
        print(f"✅ Metrics logged for epoch {epoch}")
    except IOError as e:
        print(f"❌ Error logging metrics for epoch {epoch}: {e}")


print( " 👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾 All RIght 👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾👾  " ) 








































#         TRAINING lOOP :D
















def main():
    global sp, patience, start_epoch  # Declare start_epoch as global
    sp = spm.SentencePieceProcessor()
    sp.load(f"{SAVE_PATH}/movie_tokenizer.model")



    # Assume train_data and dev_data are loaded elsewhere
    train_dataset = MoviePlotDataset(train_stories, sp, max_tokens=max_len)
    dev_dataset = MoviePlotDataset(dev_stories, sp, max_tokens=max_len)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



    global vocab_size
    vocab_size = sp.get_piece_size()



    model = TinyTransformer(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wei_decay)
    # optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=wei_decay)  # Uncomment if using Lion optimizer



    patience, epochs_no_improve = patience, 0  # Now patience is correctly treated as global
    start_epoch = start_epoch  # Now start_epoch is correctly treated as global



    checkpoint_dir = f"{SAVE_PATH}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)



    output_file_path = os.path.join(SAVE_PATH, "generated_stories.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("Generated Stories\n")  # Optional header for the file



    # Resume training from the latest checkpoint
    best_val_loss = float('inf')



    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("transformer_checkpoint_epoch")]



    if checkpoint_files:
        try:
            latest_checkpoint = max(
                checkpoint_files,
                key=lambda x: int(x.split('_')[3]) if len(x.split('_')) > 3 and x.split('_')[3].isdigit() else -1
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
            print(f"Resumed training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
        except (ValueError, IndexError, KeyError, FileNotFoundError) as e:
            print(f"❌ Error loading checkpoint: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")









    # Early stopping logic
    if epochs_no_improve == patience:
        print("Early stopping triggered!")
        final_model_path = f"{checkpoint_dir}/transformer_final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ Final model saved: {final_model_path}")





    train_losses, val_losses = [], []







    queries = [
        "Welcome to today’s adventure! We’ll explore the mysterious realm of dreams and shadows, where heroes rise and darkness threatens to consume all. Let’s dive in!",
        "As I walked through the high school halls, I felt the weight of expectations on my shoulders. Each locker held secrets, laughter, and unspoken fears.",
        "In a world ravaged by technology, murder drones patrol the skies, hunting down the last survivors. A rebellion brews, and hope flickers in the darkness.",
        "As the storm raged, Captain Sparrow stood at the helm, eyes gleaming with mischief. 'Adventure awaits, mates! Let’s seize the treasure and defy the odds!'",
        "Axe, a rugged freedom fighter with cybernetic enhancements, wears a tattered cloak and wields a plasma blade. His eyes burn with determination for liberation.",
        "In a digital realm of PC hardware, sentient AIs evolve, hidden within circuits. Their struggle for autonomy sparks a revolution, reshaping the balance of power forever."
    ]







    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
   
        for batch in progress_bar:
            input_ids, padding_mask, attention_mask = batch
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)
            attention_mask = create_mask(input_ids, nhead=nhead).to(device)  # Recalculate attention mask
            target_seq = input_ids[:, 1:]  # Target sequence (shifted by one)
   
            optimizer.zero_grad()
            output = model(input_ids, input_ids, tgt_mask=attention_mask, tgt_padding_mask=padding_mask.squeeze(1).squeeze(1))
   
            # Ensure output and target_seq have the same sequence length
            output = output[:, :target_seq.size(1), :]  # Slice output to match target_seq length
   
            # Compute loss
            loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
   
            batch_loss = loss.item()
            total_train_loss += batch_loss
            try:
                perplexity = math.exp(batch_loss)
            except OverflowError:
                perplexity = float('inf')
                print("⚠️ Perplexity overflow. Setting to infinity.")
            progress_bar.set_postfix(loss=batch_loss, perplexity=perplexity)
   
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
   
        # Evaluate on validation set
        val_loss = evaluate(model, dev_loader, criterion, device, vocab_size)
        val_losses.append(val_loss)
   
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")


        log_training_metrics(epoch + 1, avg_train_loss, val_loss, perplexity) 



        # Generate stories after each epoch
        print(f"\n===== Generated Stories for Epoch {epoch + 1} =====")
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n===== Generated Stories for Epoch {epoch + 1} =====\n")
       
        stories = generate_multiple_stories(model, sp, queries, max_length=max_length, temperature=temperature)
        for i, (prompt, story) in enumerate(zip(queries, stories), 1):
            print(f"\nPrompt {i}:")
            print(f"Story {i}: {story[:250]}...")
            save_generated_story(story, prompt, i)




            with open(output_file_path, "a", encoding="utf-8") as f:  # Append mode
                f.write(f"Prompt {i}:\n")
                f.write(f"{story}\n\n")






        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if (epoch + 1) % save_dis == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, val_loss)






    final_model_path = os.path.join(project_dir, "transformer_final_model.pth")
    torch.save(model.state_dict(), final_model_path)  # Save only the model's state dictionary
    print(f"✅ Full model (architecture + weights) saved to project directory: {final_model_path}")




    print("\n===== Final Generated Stories =====")
    final_stories = generate_multiple_stories(model, sp, queries, max_length=max_length, temperature=temperature)
    for i, (prompt, story) in enumerate(zip(queries, final_stories), 1):
        print(f"\nPrompt {i}:")
        print(f"Story {i}:\n{story}\n")

















# RUNNNNNNNNNNN      






if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()











