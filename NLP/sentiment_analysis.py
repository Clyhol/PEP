import torch
import torch.nn as nn
from torch.nn import functional as F
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from tqdm import tqdm


batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 0.0003
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed_dims = 384
n_heads = 6
n_layers = 6
dropout = 0.2
n_classes = 2 # positive or negative sentiment

# Load the IMDB dataset
train_iter, test_iter = IMDB() # type: ignore

# Tokenizer and vocabulary
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator((tokenizer(text) for label, text in train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)

def encode_batch(batch):
    labels, texts = zip(*batch)
    tokenized_texts = [tokenizer(text) for text in texts]
    encoded_texts = [torch.tensor(vocab(tokenized_text), dtype=torch.long) for tokenized_text in tokenized_texts]
    labels = torch.tensor([1 if label == "pos" else 0 for label in labels], dtype=torch.long)
    return labels, encoded_texts

# DataLoader
def collate_batch(batch):
    labels, texts = encode_batch(batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return labels, padded_texts, lengths

batch_size = 64
train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed_dims, head_size, bias=False)
        self.query = nn.Linear(n_embed_dims, head_size, bias=False)
        self.value = nn.Linear(n_embed_dims, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        No mask since we are not predicting the future tokens, so looking ahead is allowed.
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = q @ k.transpose(-2, -1) / ((C) ** 0.5)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embed_dims, n_embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_out = torch.cat([head(x) for head in self.heads], dim=-1)
        head_out = self.projection(head_out)
        return head_out

class FeedForward(nn.Module):
    def __init__(self, n_embed_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed_dims, 4 * n_embed_dims),
            nn.ReLU(),
            nn.Linear(4 * n_embed_dims, n_embed_dims),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed_dims, n_heads):
        super().__init__()
        head_size = n_embed_dims // n_heads
        self.self_attention = MultiHeadAttention(n_heads, head_size)
        self.feedforward = FeedForward(n_embed_dims)
        self.layer_norm1 = nn.LayerNorm(n_embed_dims)
        self.layer_norm2 = nn.LayerNorm(n_embed_dims)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feedforward(self.layer_norm2(x))
        return x

# Define the modified model for sentiment analysis
class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed_dims)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed_dims)
        self.blocks = nn.Sequential(*[Block(n_embed_dims, n_heads=n_heads) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(n_embed_dims)
        self.fc = nn.Linear(n_embed_dims, n_classes)
    
    def forward(self, text, lengths=None):
        B, T = text.size()
        
        if T > block_size:
            text = text[:, :block_size]  # truncate to block_size
            T = block_size
        
        token_embed = self.embedding_table(text)
        pos_embed = self.positional_embedding_table(torch.arange(T, device=text.device))
        tokens = token_embed + pos_embed
        tokens = self.blocks(tokens)
        tokens = self.final_layer_norm(tokens)
        tokens = tokens.mean(dim=1)  # Average pooling over the sequence length
        return self.fc(tokens)

    
model = SentimentAnalysisModel().to(device)

# Load the pre-trained weights, if available
model_path = "NLP\\pretrained_language_model.pth"
pretrained_weights = torch.load(model_path, map_location=device)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
max_epochs = 5
for epoch in tqdm(range(max_epochs)):
    model.train()
    for labels, texts, lengths in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(texts, lengths)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, texts, lengths in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            output = model(texts, lengths)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch+1}: Accuracy: {100 * correct / total:.2f}%')
    torch.save(model.state_dict(), 'sentiment_analysis_model.pth')