# Week10



### 語言處理

* ChatGPT 與傳統語法理論不同，程式語言目前都是使用type2語法

* type3語法描述

  ```
  S -> a A    # 只能有一個非終端項目  # 所以不能 S -> A B
  A -> bB | aA
  B -> b | bB
  ```



### AI 分類

* 分類問題
  * nn.CrossEntorpyLoss 做最後的損失函數
  * torch.optim.Adam 自動調整衝量的梯度下降法
* 回歸問題
  * 數值化問題



### 生成語法

> eliza.py

```python
'''
以下為本程式回答問題時使用的 Q&A 規則，例如對於以下 Q&A 規則物件

: 'Q':"想 | 希望", 'A':"為何想*呢?|真的想*?|那就去做阿?為何不呢?",

代表的是，當您輸入的字串中有「想」或「希望」這樣的詞彙時，
程式就會從 'A': 欄位中的回答裏隨機選出一個來回答。

回答語句中的 * 代表比對詞彙之後的字串，舉例而言、假如您說：

    我想去巴黎

那麼我們的程式從這四個可能的規則中隨機挑出一個來產生答案，產生的答案可能是：

為何想去巴黎呢?
真的想去巴黎?
那就去做阿?
為何不呢?

Eliza 就是一個這麼簡單的程式而已。
'''

import re
import math
import random as R
# Q&A 陣列宣告
qa_list = [
{ 'Q':"謝謝", 'A':"不客氣!" },
{ 'Q':"對不起 | 抱歉 | 不好意思", 'A':"別說抱歉 !|別客氣，儘管說 !" },
{ 'Q':"可否 | 可不可以", 'A':"你確定想*?" },
{ 'Q':"我想", 'A':"你為何想*?" },
{ 'Q':"我要", 'A':"你為何要*?" },
{ 'Q':"你是", 'A':"你認為我是*?" },
{ 'Q':"認為 | 以為", 'A':"為何說*?" },
{ 'Q':"感覺", 'A':"常有這種感覺嗎?" },
{ 'Q':"為何不", 'A':"你希望我*!" },
{ 'Q':"是否", 'A':"為何想知道是否*?" },
{ 'Q':"不能", 'A':"為何不能*?|你試過了嗎?|或許你現在能*了呢?" },
{ 'Q':"我是", 'A':"你好，久仰久仰!" },
{ 'Q':"甚麼 | 什麼 | 何時 | 誰 | 哪裡 | 如何 | 為何 | 因何", 'A':"為何這樣問?|為何你對這問題有興趣?|你認為答案是甚麼呢?|你認為如何呢?|你常問這類問題嗎?|這真的是你想知道的嗎?|為何不問問別人?|你曾有過類似的問題嗎?|你問這問題的原因是甚麼呢?" },
{ 'Q':"原因", 'A':"這是真正的原因嗎?|還有其他原因嗎?" }, 
{ 'Q':"理由", 'A':"這說明了甚麼呢?|還有其他理由嗎?" },
{ 'Q':"你好 | 嗨 | 您好", 'A':"你好，有甚麼問題嗎?" },
{ 'Q':"或許", 'A':"你好像不太確定?" },
{ 'Q':"不曉得 | 不知道", 'A':"為何不知道?|在想想看，有沒有甚麼可能性?" },
{ 'Q':"不想 | 不希望", 'A':"有沒有甚麼辦法呢?|為何不想*呢?|那你希望怎樣呢?" }, 
{ 'Q':"想 | 希望", 'A':"為何想*呢?|真的想*?|那就去做阿?為何不呢?" },
{ 'Q':"不", 'A':"為何不*?|所以你不*?" },
{ 'Q':"請", 'A':"我該如何*呢?|你想要我*嗎?" },
{ 'Q':"你", 'A':"你真的是在說我嗎?|別說我了，談談你吧!|為何這麼關心我*?|不要再說我了，談談你吧!|你自己*" },
{ 'Q':"總是 | 常常", 'A':"能不能具體說明呢?|何時?" },
{ 'Q':"像", 'A':"有多像?|哪裡像?" },
{ 'Q':"對", 'A':"你確定嗎?|我了解!" },
{ 'Q':"朋友", 'A':"多告訴我一些有關他的事吧!|你認識他多久了呢?" },
{ 'Q':"電腦", 'A':"你說的電腦是指我嗎?" }, 
{ 'Q':"難過", 'A':"別想它了|別難過|別想那麼多了|事情總是會解決的"},
{ 'Q':"高興", 'A':"不錯ㄚ|太棒了|這樣很好ㄚ"},
{ 'Q':"是阿|是的", 'A':"甚麼事呢?|我可以幫助你嗎?|我希望我能幫得上忙!" },
{ 'Q':"", 'A':"我了解|我能理解|還有問題嗎 ?|請繼續說下去|可以說的更詳細一點嗎?|這樣喔! 我知道!|然後呢? 發生甚麼事?|再來呢? 可以多說一些嗎|接下來呢? |可以多告訴我一些嗎?|多談談有關你的事，好嗎?|想多聊一聊嗎|可否多告訴我一些呢?" }
]

def answer(say):
	for qa in qa_list: # 對於每一個 QA
		qList = qa['Q'].split("|") # 取出 Q 部分，分割成一個一個的問題字串 q
		aList = qa['A'].split("|") # 取出回答 A 部分，分割成一個一個的回答字串 q
		for q in qList: # 對於每個問題字串 q
			if q.strip() == "": # 如果是最後一個「空字串」的話，那就不用比對，直接任選一個回答。
				return R.choice(aList) # 那就從答案中任選一個回答
			m = re.search("(.*)"+q.strip()+"([^?.;]*)", say)
			if m: # 比對成功的話
				tail = m.group(2) # 就取出句尾
				# 將問句句尾的「我」改成「你」，「你」改成「我」。
				tail = tail.replace("我", "#").replace("你", "我").replace("#", "你")
				return R.choice(aList).replace('*', tail) # 然後將 * 改為句尾進行回答
	return "然後呢？" # 如果發生任何錯誤，就回答「然後呢？」來混過去。


def eliza():
	print('你好，我是 Eliza ! ')
	while (True):
		say = input('> ') # 取得使用者輸入的問句。
		if say == 'bye':
			break
		ans = answer(say)
		print(ans)

eliza()
```

* 分類問題

  ```
  nn.CrossEntorpyLoss 做最後的損失函數
  torch.optim.Adam 自動調整衝量的梯度下降法
  ```

* perplexity : 衡量函數是否收斂的指標

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
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 3 # 原為 5
num_samples = 1000     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

def load_data(train_file):
    global corpusObj, ids, vocab_size, num_batches
    corpusObj = Corpus()
    ids = corpusObj.get_data(train_file, batch_size)
    print('ids.shape=', ids.shape)
    vocab_size = len(corpusObj.dictionary)
    print('vocab_size=', vocab_size)
    num_batches = ids.size(1) // seq_length

# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, method, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        method = method.upper()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if method == "RNN":
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        elif method == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise Exception(f'RNNLM: method={method} not supported!')
        self.linear = nn.Linear(hidden_size, vocab_size)
        
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

if len(sys.argv) < 3:
    print('usage: python main.py <corpus> (train or test)')
    exit()

corpus = sys.argv[1]
method = sys.argv[2]
job = sys.argv[3]

load_data(f'{corpus}.txt')
if job == 'train':
    train(corpus, method)
elif job == 'test':
    test(corpus, method)
```





