import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtext.datasets import IMDB
from torchtext.legacy.data import Field, LabelField, BucketIterator
from torchtext.data.utils import get_tokenizer

# hyperparameters
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 0.0003
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Prepare the data
tokenizer = get_tokenizer("basic_english")
TEXT = Field(tokenize=tokenizer, lower=True, batch_first=True)
LABEL = LabelField(dtype=torch.long, batch_first=True)

train_data, test_data = IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(split_ratio=0.8)

TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=batch_size, device=device
)

vocab_size = len(TEXT.vocab)
n_classes = len(LABEL.vocab)


class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(512, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.fc = nn.Linear(n_embd, n_classes)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)  # average over the sequence length
        logits = self.fc(x)
        return logits


model = SentimentAnalysisModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def evaluate(model, iterator):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in iterator:
            texts, labels = batch.text, batch.label
            outputs = model(texts)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return epoch_loss / len(iterator), correct / total


for iter in range(max_iters):
    model.train()
    epoch_loss = 0
    for batch in train_iterator:
        texts, labels = batch.text, batch.label
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        train_loss, train_acc = evaluate(model, train_iterator)
        valid_loss, valid_acc = evaluate(model, valid_iterator)
        print(
            f"Iter: {iter}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc:.4f}"
        )
