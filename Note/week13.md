# Week13



### Embedding

* 在自然語言處理（NLP）領域中，embedding 是將詞語或文本轉換為連續向量表示的技術
* 通過將詞語映射到一個連續的低維向量空間中，可以捕捉詞語之間的語義關係。例如，相似的詞語在嵌入空間中通常彼此更接近。這些嵌入向量可以在許多NLP任務中作為輸入特徵使用



### RNN

* 是一種在序列數據上進行處理的神經網絡模型，與傳統的前饋神經網絡不同，RNN具有記憶單元，使其能夠捕捉序列數據中的時間相依性
* 基本結構是一個循環的神經元單元。在每個時間步驟，RNN接收一個輸入向量和先前時間步驟的隱藏狀態作為輸入，並生成一個新的隱藏狀態和輸出。這個遞歸的過程使得RNN可以捕捉到序列中的上下文信息，並在不同時間步驟共享權重
* 傳統的RNN在處理長期依賴關係時存在「梯度消失」或「梯度爆炸」的問題，這導致模型難以學習長序列之間的關係。為了解決這個問題，出現了一些RNN的變種，如長短期記憶網絡（LSTM）和門控循環單元（GRU），它們通過引入門控機制和記憶單元，有效地處理長序列和梯度問題

> RNN

```python
# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 8   # 所有詞向量變成幾維度
hidden_size = 32  # 隱藏層神經元數量
num_layers = 1
num_epochs = 20 # 原為 5
num_samples = 1000     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

def load_data(train_file):
    global corpusObj, ids, vocab_size, num_batches
    corpusObj = Corpus()
    ids = corpusObj.get_data(train_file, batch_size)
    print('ids.shape=', ids.shape)
    print('dictionary=', corpusObj.dictionary.idx2word)
    vocab_size = len(corpusObj.dictionary)
    print('vocab_size=', vocab_size)
    num_batches = ids.size(1) // seq_length

# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, method, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        method = method.upper()
        # seq2seq最基本的手法，一開始都會做這步驟
        self.embed = nn.Embedding(vocab_size, embed_size)  # 把所有詞變為詞向量
        if method == "RNN":
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        elif method == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise Exception(f'RNNLM: method={method} not supported!')
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def get_embed(self, x):
        return self.embed(x)
    
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate 
        out, h = self.rnn(x, h)
        
        # Reshape output to (batch_size*seq_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, h

def train(corpus, method):
    global model
    model = RNNLM(method, vocab_size, embed_size, hidden_size, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        # Set initial hidden // and cell states (for LSTM)
        states = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        
        for i in range(0, ids.size(1) - seq_length, seq_length):
            # Get mini-batch inputs and targets
            inputs = ids[:, i:i+seq_length].to(device) # 輸入為目前詞 (1-Batch)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device) # 輸出為下個詞 (1-Batch)
            
            # Forward pass
            states = states.detach() # states 脫離 graph
            outputs, states = model(inputs, states) # 用 model 計算預測詞
            loss = criterion(outputs, targets.reshape(-1)) # loss(預測詞, 答案詞)
            
            # Backward and optimize
            optimizer.zero_grad() # 梯度歸零
            loss.backward() # 反向傳遞
            clip_grad_norm_(model.parameters(), 0.5) # 切斷，避免梯度爆炸
            optimizer.step() # 向逆梯度方向走一步

            step = (i+1) // seq_length
            if step % 100 == 0:
                print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

    # Save the model checkpoints
    # torch.save(model.state_dict(), 'model.ckpt')
    torch.save(model, f'{corpus}_{method}.pt')

def test(corpus, method):
    # Test the model
    model = torch.load(f'{corpus}_{method}.pt')
    with torch.no_grad():
        with open(f'{corpus}_{method}.txt', 'w', encoding='utf-8') as f:
            # Set intial hidden ane cell states
            state = torch.zeros(num_layers, 1, hidden_size).to(device)

            # Select one word id randomly # 這裡沒有用預熱
            prob = torch.ones(vocab_size)
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

            for i in range(num_samples):
                # Forward propagate RNN 
                output, state = model(input, state)

                # Sample a word id
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()

                # Fill input with sampled word id for the next time step
                input.fill_(word_id)

                # File write
                word = corpusObj.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i+1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, f'{corpus}_{method}.txt'))

def show_embed(corpus, method):
    model = torch.load(f'{corpus}_{method}.pt')
    for idx, word in corpusObj.dictionary.idx2word.items():
        input = torch.LongTensor(1)
        input[0] = idx
        embed = model.get_embed(input)[0]
        print(f'{idx}:{word} {embed}')

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for idx1, word1 in corpusObj.dictionary.idx2word.items():
        v1 = torch.LongTensor(1)
        v1[0] = idx1
        embed1 = model.get_embed(v1)
        print(f'{idx1}:{word1}')
        for idx2, word2 in corpusObj.dictionary.idx2word.items():
            v2 = torch.LongTensor(1)
            v2[0] = idx2
            embed2 = model.get_embed(v2)
            print(f'  similarity {idx2}:{word2} {cos(embed1,embed2).item()}')
    

if len(sys.argv) < 3:
    print('usage: python main.py <corpus> (train or test)')
    exit()

corpus = sys.argv[1]  # 輸入的語料庫
method = sys.argv[2]  # 方法(RNN or GRU)
job = sys.argv[3]  # tain or test

load_data(f'{corpus}.txt')
if job == 'train':
    train(corpus, method)
elif job == 'test':
    test(corpus, method)
elif job == 'embed':
    show_embed(corpus, method)
```

