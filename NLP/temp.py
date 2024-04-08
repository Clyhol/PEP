with open("NLP\\alice.txt", encoding="utf-8") as f:
    text = f.read()

print("length of dataset:", len(text))

chars = sorted(set(text))
vocab_size = len(chars)
print("length of vocabulary:", vocab_size)
print("vocabulary:", "".join(chars))

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
