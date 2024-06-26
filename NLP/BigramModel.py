import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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


########### initialize model ############
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )  # a lookup table where rows are plucked out based on the input token (one-hot encoded)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(
            idx)  # shape: (batch_size, block_size, vocab_size) OR (B, T, C)

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
        # idx is the context
        for i in range(max_new_tokens):
            # get predictions (logit is the output before applying an activation function)
            logits, loss = self.forward(
                idx
            )  # currently feeding in the entire context, but only need the last token
            # store only the last prediction
            logits = logits[:, -1, :]
            # convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # pick sample
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append predicted token to context
            idx = torch.cat([idx, next_token], dim=1)
        return idx


model = BigramLanguageModel(vocab_size).to(device)

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
print(decoder(model.predict_next(context, max_new_tokens=2000)[0].tolist()))
