import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import random
random.seed(42)

# Function for building a dataset
def build_dataset(words, block_size, stoi):
    """
    Constructs training data for a character-level language model.
    
    Args:
        words (list of str): List of words to process.
        block_size (int): The context window size.
        stoi (dict): Dictionary mapping characters to integer indices.

    Returns:
        X (torch.Tensor): Input tensor of shape (num_samples, block_size).
        Y (torch.Tensor): Target tensor of shape (num_samples,).
    """
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size  # Initialize context with padding (assuming 0 is a special token)
        
        for ch in w + '.':  # Append '.' as an end-of-word marker
            ix = stoi[ch]  # Convert character to index
            X.append(context)  # Store the current context as input
            Y.append(ix)  # Store the target character index
            
            # Slide the context window forward
            context = context[1:] + [ix]  

    X = torch.tensor(X)  # Convert list to tensor
    Y = torch.tensor(Y)  
    
    return X, Y


# Function for computing final losses on training, validation, and test sets
def compute_loss(X, Y, C, W1, b1, W2, b2):
    emb = C[X]  
    h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y)


# .................ENGLISH STAR NAMES.................

# ENGLISH DATASET
star_names = open('eng_star_names.txt', 'r').read().splitlines() # Getting names list form a .txt

print(star_names[:8])

# Build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(star_names))))
stoi = {s:i+1 for i,s in enumerate(chars)} # Create a character-to-index mapping (1-based)
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # Create an index-to-character mapping
print(itos)

# Shuffle dataset for training, validation, and test splits
random.shuffle(star_names)

# Define dataset splits (80% train, 10% validation, 10% test)
n1 = int(0.8 * len(star_names))
n2 = int(0.9 * len(star_names))

block_size = 3 # Context length: number of characters used to predict the next one

# Create datasets for training, validation, and testing
Xtr, Ytr = build_dataset(star_names[:n1], block_size, stoi)
Xdev, Ydev = build_dataset(star_names[n1:n2], block_size, stoi)
Xte, Yte = build_dataset(star_names[n2:], block_size, stoi)


# ENGLISH MODEL
embedding_dim = 10  # Each character is represented by a 10-dimensional vector

# Initialize model parameters with random values
g = torch.Generator().manual_seed(42)  # Set random seed for reproducibility
C = torch.randn((len(stoi), embedding_dim), generator=g)  # Character embedding matrix
W1 = torch.randn((block_size * embedding_dim, 200), generator=g)  # First layer weights
b1 = torch.randn(200, generator=g)  # First layer biases
W2 = torch.randn((200, len(stoi)), generator=g)  # Second layer weights
b2 = torch.randn(len(stoi), generator=g)  # Second layer biases


# List of parameters to be optimized
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True  # Enable gradient computation

# Learning rate schedule
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

# Lists to store training statistics
lri = []
lossi = []
stepi = []


# TRAINING
for i in range(200000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (64,))

  # forward pass
  emb = C[Xtr[ix]]  # Look up character embeddings
  h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1) # Hidden layer
  logits = h @ W2 + b2  # Output layer
  loss = F.cross_entropy(logits, Ytr[ix]) # Compute loss

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 100000 else 0.0001
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  stepi.append(i)
  lossi.append(loss.log10().item())

# Plot training loss over time
plt.plot(stepi, lossi)

loss_tr = compute_loss(Xtr, Ytr, C, W1, b1, W2, b2)
print("English model training loss:", loss_tr)

loss_v = compute_loss(Xdev, Ydev, C, W1, b1, W2, b2)
print("English model validation loss:", loss_v)

loss_te = compute_loss(Xte, Yte, C, W1, b1, W2, b2)
print("English model test loss:", loss_te)


# VISUALIZATION
# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')

# GENERATION
# sample from the model
g = torch.Generator().manual_seed(42 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size  # Start with an empty context
    while True:
        emb = C[torch.tensor([context])]  # Get embedding for the current context
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)  # Compute hidden state
        logits = h @ W2 + b2  # Compute output logits
        probs = F.softmax(logits, dim=1)  # Convert to probabilities
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()  # Sample a character
        context = context[1:] + [ix]  # Shift context
        out.append(ix)  # Store generated character
        
        if ix == 0:  # Stop if end-of-sequence marker is reached
            break

    print(''.join(itos[i] for i in out))  # Convert indices back to characters


# .................RUSSIAN STAR NAMES.................

# RUSSIAN DATASET
ru_star_names = open('russian_star_names.txt', 'r').read().splitlines() # Getting names list form a .txt

print(ru_star_names[:8])

# Build the vocabulary of characters and mappings to/from integers
ru_chars = sorted(list(set(''.join(ru_star_names))))
ru_stoi = {s:i+1 for i,s in enumerate(ru_chars)} # Create a character-to-index mapping (1-based)
ru_stoi['.'] = 0
ru_itos = {i:s for s,i in ru_stoi.items()} # Create an index-to-character mapping
print(ru_itos)

# Shuffle dataset for training, validation, and test splits
random.shuffle(ru_star_names)

# Define dataset splits (80% train, 10% validation, 10% test)
ru_n1 = int(0.8*len(ru_star_names))
ru_n2 = int(0.9*len(ru_star_names))

# Create datasets for training, validation, and testing
ru_Xtr, ru_Ytr = build_dataset(ru_star_names[:ru_n1], block_size, ru_stoi)
ru_Xdev, ru_Ydev = build_dataset(ru_star_names[ru_n1:ru_n2], block_size, ru_stoi)
ru_Xte, ru_Yte = build_dataset(ru_star_names[ru_n2:], block_size, ru_stoi)


# RUSSIAN MODEL
# Initialize model parameters with random values
ru_g = torch.Generator().manual_seed(42) # Set random seed for reproducibility
ru_C = torch.randn((80, 10), generator=ru_g) # Character embedding matrix
ru_W1 = torch.randn((30, 200), generator=ru_g) # First layer weights
ru_b1 = torch.randn(200, generator=ru_g) # First layer biases
ru_W2 = torch.randn((200,80), generator=ru_g) # Second layer weights
ru_b2 = torch.randn(80, generator=ru_g) # Second layer biases

# List of parameters to be optimized
ru_parameters = [ru_C, ru_W1, ru_b1, ru_W2, ru_b2]

for p in ru_parameters:
  p.requires_grad = True # Enable gradient computation

# Learning rate schedule
ru_lre = torch.linspace(-3, 0, 1000)
ru_lrs = 10**ru_lre

# Lists to store training statistics
ru_lri = []
ru_lossi = []
ru_stepi = []


# TRAINING
for i in range(200000):

  # minibatch construct
  ru_ix = torch.randint(0, ru_Xtr.shape[0], (64,))

  # forward pass
  ru_emb = ru_C[ru_Xtr[ru_ix]] # (32, 3, 2)
  ru_h = torch.tanh(ru_emb.view(-1, block_size * embedding_dim) @ ru_W1 + ru_b1) # (32, 100)
  ru_logits = ru_h @ ru_W2 + ru_b2 # (32, 27)
  ru_loss = F.cross_entropy(ru_logits, ru_Ytr[ru_ix])

  # backward pass
  for p in ru_parameters:
    p.grad = None
  ru_loss.backward()

  # update
  ru_lr = 0.1 if i < 100000 else 0.0001
  for p in ru_parameters:
    p.data += -ru_lr * p.grad

  # track stats
  ru_stepi.append(i)
  ru_lossi.append(ru_loss.log10().item())

plt.plot(ru_stepi, ru_lossi)


# Training loss
ru_loss_tr = compute_loss(ru_Xtr, ru_Ytr, ru_C, ru_W1, ru_b1, ru_W2, ru_b2)
print("Russian model training loss:", ru_loss_tr)
# Validation loss
ru_loss_v = compute_loss(ru_Xdev, ru_Ydev, ru_C, ru_W1, ru_b1, ru_W2, ru_b2)
print("Russian model validation loss:", ru_loss_v)
# Test loss
ru_loss_te = compute_loss(ru_Xte, ru_Yte, ru_C, ru_W1, ru_b1, ru_W2, ru_b2)
print("Russian model test loss:", ru_loss_te)


# VISUALIZATION
# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(ru_C[:,0].data, ru_C[:,1].data, s=200)
for i in range(ru_C.shape[0]):
    plt.text(ru_C[i,0].item(), ru_C[i,1].item(), ru_itos[i], ha="center", va="center", color='white')
plt.grid('minor')

# GENERATION
# sample from the model
ru_g = torch.Generator().manual_seed(42 + 10)

for _ in range(20):

    ru_out = []
    ru_context = [0] * block_size # initialize with all ...
    while True:
      ru_emb = ru_C[torch.tensor([ru_context])] # (1,block_size,d)
      ru_h = torch.tanh(ru_emb.view(1, -1) @ ru_W1 + ru_b1)
      ru_logits = ru_h @ ru_W2 + ru_b2
      ru_probs = F.softmax(ru_logits, dim=1)
      ru_ix = torch.multinomial(ru_probs, num_samples=1, generator=ru_g).item()
      ru_context = ru_context[1:] + [ru_ix]
      ru_out.append(ru_ix)
      if ru_ix == 0:
        break

    print(''.join(ru_itos[i] for i in ru_out))



# APPLYING RUSSIAN MODEL TO ENGLISH DATASET
for _ in range(20):

    ru_out1 = []
    ru_context1 = [0] * block_size 
    while True:
      emb = C[torch.tensor([ru_context1])] 
      ru_h1 = torch.tanh(emb.view(1, -1) @ ru_W1 + ru_b1) # Compute hidden layer activations using the Russian model weights (but English embeddings)
      ru_logits1 = ru_h1 @ ru_W2 + ru_b2
      ru_probs1 = F.softmax(ru_logits1, dim=1)
      ru_ix1 = torch.multinomial(ru_probs1, num_samples=1, generator=ru_g).item()
      ru_ix1 = min(ru_ix1, 58) # Ensure the index does not exceed the vocabulary size (58 is the max valid index in English characters set)
      ru_context1 = ru_context1[1:] + [ru_ix1]
      ru_out1.append(ru_ix1)
      if (ru_ix1 == 0):
        break

    # Convert generated indices to characters with the help of English itos
    print(''.join(itos[i] for i in ru_out1))

    # Russian model on English training dataset loss
emb1 = C[Xtr] # getting an embedding matrix of training dataset size
ru_h1 = torch.tanh(emb1.view(-1, 30) @ ru_W1 + ru_b1) 
ru_logits1 = ru_h1 @ ru_W2 + ru_b2
ru_loss1 = F.cross_entropy(ru_logits1, Ytr)
print("Russian model on English training dataset loss:", ru_loss1)



# APPLYING ENGLISH MODEL TO RUSSIAN DATASET
for _ in range(20):

    out1 = []
    context1 = [0] * block_size
    while True:
      ru_emb = ru_C[torch.tensor([context1])] 
      h1 = torch.tanh(ru_emb.view(1, -1) @ W1 + b1) # Compute hidden layer activations using the English model weights (but Russian embeddings)
      logits1 = h1 @ W2 + b2
      probs1 = F.softmax(logits1, dim=1)
      ix1 = torch.multinomial(probs1, num_samples=1, generator=g).item()
      context1 = context1[1:] + [ix1]
      out1.append(ix1)
      if (ix1 == 0):
        break

  # Convert generated indices to characters with the help of Russian itos
    print(''.join(ru_itos[i] for i in out1))

# English model on Russian training dataset loss

ru_emb1 = ru_C[ru_Xtr] # getting a Russian embedding matrix of training dataset size
h1 = torch.tanh(ru_emb1.view(-1, 30) @ W1 + b1) # (32, 100)
logits_tr1 = h1 @ W2 + b2 # (32, 27)
# Clip target indices to be within the valid range for the English model
ru_Ytr_clipped = torch.clamp(ru_Ytr, 0, 58)
loss_tr1 = F.cross_entropy(logits_tr1, ru_Ytr_clipped)

print("English model on Russian training dataset loss:", loss_tr1)