import torch
import torch.nn as nn
from torch.nn import functional as F


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
