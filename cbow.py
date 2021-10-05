import sys
import torch
import torch.nn as nn

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMDEDDING_DIM = 100
MBEDDING_DIM = 128

file = sys.argv[1]
with open(file, 'r', encoding='utf-8') as file:
    raw_text = file.read().split();

print(f'Raw text: {" ".join(raw_text)}\n')

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
print(vocab)
vocab_size = len(vocab)
print(vocab_size)

word_to_ix = {word:ix for ix, word in enumerate(vocab)}
ix_to_word = {ix:word for ix, word in enumerate(vocab)}


print(word_to_ix)

data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = []
    for j in range(CONTEXT_SIZE):
        context.append(raw_text[i - CONTEXT_SIZE + j])
    for j in range(CONTEXT_SIZE):
        context.append(raw_text[i + j + 1])
        target = raw_text[i]
        data.append((context, target))

for row in data:
    print(row)

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        #out: 1 x embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, MBEDDING_DIM)
        self.activation_function1 = nn.ReLU()
        
        #out: 1 x vocab_size
        self.linear2 = nn.Linear(MBEDDING_DIM, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim = -1)
        

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_embedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1,-1)


model = CBOW(vocab_size, EMDEDDING_DIM)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for context, target in data:
    context_vector = make_context_vector(context, word_to_ix)
    print(f'X:{context_vector}, Y:[{word_to_ix[target]}]')

print("Creating neural network architecture:")
print(f'Default hidden dense layer of neural network {vocab_size} => {EMDEDDING_DIM}')
print(f'Second linear layer of neural network {EMDEDDING_DIM} => {MBEDDING_DIM}')
print(f'Third linear layer of neural network {MBEDDING_DIM} => {vocab_size}')
print("Now training in 50 epochs (iterations) ...")

#TRAINING
for epoch in range(50):
    print(f'Epoch (iteration): {epoch}')
    total_loss = 0

    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)
        log_probs = model(context_vector)
        total_loss += loss_function(log_probs, torch.tensor([word_to_ix[target]]))

 
    #optimize at the end of each epoch
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f'Total_loss: {total_loss}')

#TESTING
context = ['People','create','to', 'direct']
context_vector = make_context_vector(context, word_to_ix)
a = model(context_vector)

print(f'Test input:')
print(f'Context: {context}\n')
#Print result
print(f'Result:')
print(f'a[0]: {a[0]}')
print(f'argmax(a[0]): {torch.argmax(a[0])}')
pword = ix_to_word[torch.argmax(a[0]).item()]
print(f'Prediction: {pword}')
print(f'Word-embedding: {model.get_word_embedding(pword)}')
