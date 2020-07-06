import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html



# Very useful tip: use the dir() function to dig deeper into an object
#  vars() might also be useful as well

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

# Note how positional encodings are stored in a buffer
# https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/8
# Storing in buffers means that it isn't stored in model.parameters()
# Therefore the optimizer won't try to update these values
# Fits the use case for a positional encoding


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Taking original data X and adding the positional encoding
        # for each vector
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


import torchtext
from torchtext.data.utils import get_tokenizer
# REMEMBER! Need to use the tokenizer associated with the model
# that you are using. Otherwise the one-hot encoding of presenting the
# tokens to the model will NOT be the system that the model learnt

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
# Look at some of the things produced using the SPECIFIC tokenizer algo
#print(TEXT.vocab.freqs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The next bit is to take text and then to stitch them up into batches
# This batching function, like the one seen in text_class.py, is used within
# the DataSet and subequently the DataLoader pytorch classes

# We have seen above how the vocabulary is built, now lets take 
# a look as to how we can view the data

#Since we have a single corpus, we only look at the first index
# Extract the first 11 TOKENS
print(train_txt[0].text[0:10])
# Notice above the special <sos> start of sentence and <eos> tokens we 
# specified earlier


# Now what we can do is convert these 11 tokens to NUMERICAL Values
# using the special text object
print(TEXT.numericalize([train_txt[0].text[0:10]]))
print(train_txt[0].text[0:10])

# THESE ARE INDEX POSITIONS!!!!
# note above how <eos> is assigned the same position number 3,
# this is why 3 is repeated a few times for this extract of tokens

# Since 3 refers to the INDEX in the lookup vocabulary table,
# let's see if it works
print(TEXT.vocab.itos[3])
# try for valkyria
print(TEXT.vocab.itos[3852])

# https://stackoverflow.com/questions/48915810/pytorch-contiguous


def batchify(data, bsz):
    # Convert text tokens into lookup table positions
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    # Calculates number of batches produced at given batch size
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    # Narrow allows you to split it up nicely into nested tensors
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # Slice it up and reformat it via the .view function
    # Batch size * nbatch
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10

# data = train_txt
# bsz = batch_size

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# Okay now we have the train/val/test data in INDEX-encoded batches
# How do we define our targets?


# get_batch() function generates the input and target sequence for the transformer model. 
# It subdivides the source data into chunks of length bptt. For the language modeling task,
#  the model needs the following words as Target. For example, with a bptt value of 2, 
#  we’d get the following two Variables for i = 0:

# It should be noted that the chunks are along dimension 0, consistent with the S dimension 
# in the Transformer model. The batch dimension N is along dimension 1.

# because the data is of the form batch_size * number_of_batches


bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


# Set up the instance of the network

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# Remember the nn.Embedding layer acts as a lookup table for all of your UNIQUE
# tokens that make up your vocab
# This is in the __init__() function of TransformerModel
# self.encoder = nn.Embedding(ntoken, ninp)
# Note how it is of the form vocabulary_size * embedding_size 


criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()