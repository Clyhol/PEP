import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 0.0003
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed_dims = 32

seed = 1337
torch.manual_seed(seed)  # seeded randomness

########### initialize dataset ############

with open("NLP\\shakespeare.txt", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(
    chars
)  # note that capital and small letters are treated as different characters

########### encode the text ############
# create dictionaries to convert characters to integers and vice versa
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# encode the text
# lambda functions are used as small throwaway functions
encoder = lambda string: [
    char_to_int[char] for char in string
]  # make a list of every encoded character in input string
decoder = lambda string: "".join([int_to_char[i]
                                  for i in string])  # reverse the encoding

########### create dataset ############
data = torch.tensor(
    encoder(text), dtype=torch.long
)  # this is a 1D vector with an integer for each character in the entire text

train_size = int(len(data) * 0.9)  # 90% of the data is used for training
train_data = data[0:train_size]
val_data = data[train_size:len(data)]


########### create dataloader ############
def get_batch(mode):
    if mode == "train":
        data = train_data
    elif mode == "val":
        data = val_data
    start_idx = torch.randint(
        0,
        len(data) - block_size, (batch_size, )
    )  # get batch_size number of randoms between 0 and (length of data - block_size)

    # these loops pick a start index from start_ids and store that + block_size characters in context and targets
    # targets is offset by one character from context
    context = torch.stack([data[i:i + block_size] for i in start_idx
                           ])  # shape: (batch_size, block_size)
    targets = torch.stack([data[i + 1:i + 1 + block_size] for i in start_idx
                           ])  # shape: (batch_size, block_size)
    context, targets = context.to(device), targets.to(device)
    return context, targets


########### estimate loss function ############
@torch.no_grad()  # gradient is not needed when evaluating. No backpropagation
def estimate_loss():
    out = {}
    model.eval()  # enter evaluation mode this disables dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()  # exit evaluation mode
    return out


########### initialize head ############
class Head(nn.Module):
    """
    This defines a single head which will be used in a multiheadded attention layer. The head is responsible for letting a token communicate with previous tokens in the sequence from the same batch. This is done using keys and querys, to find a value for the affinitiy between tokens.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed_dims, head_size, bias=False)  # linear transformation of token features
        self.query = nn.Linear(n_embed_dims, head_size, bias=False)  # linear transformation of token wants to attend to
        self.value = nn.Linear(n_embed_dims, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C) -> (B, T, head_size)
        q = self.query(x)  # (B, T, C) -> (B, T, head_size)
        v = self.value(x)  # (B, T, C) -> (B, T, head_size)

        # find attention weights
        weights = q @ k.transpose(-2,-1) / ((C)**0.5)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # softmax over the time dimension (x-axis)
        
        # apply attention weights to values
        v = self.value(x) # (B, T, C) -> (B, T, head_size)
        out = weights @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out
    

class MultiHeadAttention(nn.Module):
    """
    Creates a list of heads which are used to let tokens communicate with each other. The heads are then concatenated to form a single output. This is done to allow the model to learn different types of attention patterns.
    """
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embed_dims, n_embed_dims) # project the concatenated heads into the residual pathway

    def forward(self, x):
        head_out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, n_heads * head_size)
        head_out = self.projection(head_out)  # (B, T, n_heads * head_size) -> (B, T, n_embed_dims)
        return head_out

########### define feedforward layer ############
class FeedForward(nn.Module):
    """A simple feedforward layer with one hidden layer. This layer serves the purpose of doing doing a non-linear transformation of the self-attention output.
    """
    def __init__(self, n_embed_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed_dims, 4 * n_embed_dims), #expand the dimensionality of the input
            nn.ReLU(), # non-linear transformation
            nn.Linear(4 * n_embed_dims, n_embed_dims), # projection back to the residual pathway
        )
        
    def forward(self, x):
        return self.net(x)
        
########### define a block of the model ############
class Block(nn.Module):
    def __init__(self, n_embed_dims, n_heads):
        super().__init__()
        head_size = n_embed_dims // n_heads
        self.self_attention = MultiHeadAttention(n_heads, head_size)
        self.feedforward = FeedForward(n_embed_dims)
        self.layer_norm1 = nn.LayerNorm(n_embed_dims) # layer norm is used to normalize the output of the self-attention layer. This stabilizes the training process
        self.layer_norm2 = nn.LayerNorm(n_embed_dims)
        
    def forward(self, x):
        # layer normalization is applied before and after the self-attention and feedforward layers
        x = x + self.self_attention(self.layer_norm1(x)) # add x to the output of the self-attention layer (residual connection)
        x = x + self.feedforward(self.layer_norm2(x)) # add x to the output of the feedforward layer (residual connection)
        return x
########### initialize model ############
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed_dims)  # a lookup table where rows are plucked out based on the input token (one-hot encoded)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed_dims)  # lookup table for positional embeddings
        self.blocks = nn.Sequential(
            Block(n_embed_dims, n_heads=4),
            Block(n_embed_dims, n_heads=4),
            Block(n_embed_dims, n_heads=4),
            nn.LayerNorm(n_embed_dims),
        )
        self.lm_head = nn.Linear(n_embed_dims, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embed = self.embedding_table(idx)  # shape: (batch_size, block_size, n_embed_dims) OR (B, T, n_embed_dims)
        pos_embed = self.positional_embedding_table(torch.arange(T, device=device))  # get the position of the embedding in the sequence (used for self-attention)
        tokens = token_embed + pos_embed  # add the positional embedding to the token embedding
        tokens = self.blocks(tokens)  # apply self-attention
    
        logits = self.lm_head(tokens)  # (B, T, vocab_size)

        # failsafe if true identity of next token is not known
        if targets is None:
            loss = None
        else:
            # targets contain the identity of the next character, cross_entropy computes the quality of the prediction in logits
            B, T, C = logits.shape
            logits = logits.view(
                B * T,
                C)  # value up, B, and value down, T, matrices from 3blue1brown
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        # logits are scores for each token use to predict the next token e.g. certain characters are more likely to follow others
        return logits, loss

    def predict_next(self, idx, max_new_tokens):
        # idx is the context (B, T)
        for i in range(max_new_tokens):
            # crop idx to block_size, requiret for self-attention
            idx_crop = idx[:, -block_size:]
            # get predictions (logit is the output before applying an activation function)
            logits, loss = self.forward(idx_crop)  # currently feeding in the entire context, but only need the last token
            # store only the last prediction
            logits = logits[:, -1, :]
            # convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # pick sample
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append predicted token to context
            idx = torch.cat([idx, next_token], dim=1)
        return idx


model = BigramLanguageModel().to(device)

########### initialize training loop ############
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(model.predict_next(context, max_new_tokens=1000)[0].tolist()))
