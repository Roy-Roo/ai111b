# Week3



* AI = 優化 = 給分數 + 搜尋
* AI 是使用 01010...01 存在電腦裡面並做運算，可以用在數學式或是路徑搜尋(圖像可以使用鄰接矩陣，或是鄰接陣列表達)

1. 梯度下降法
2. 仿真算法



### 遺傳演算法

* 為一種仿真算法
* 分數越高越好，找高點 : fitness

* 物競天擇，適者生存，如果父和母很好，那就會產生好的子代
* 有一大群生物，根據分數排好，然後抽取物種(分數越高被抽到機率越高，通常最高的機率會比最低大兩倍，如果差太多會有overfitting的問題)，生下子代

```
父: 0110|0111   (1|2)
母: 1101|1001   (A|B)
子: 0110|1001   (1|B)
```



> geneticAlgorithm.py

```python
import random
import math

class GeneticAlgorithm:
    def __init__(self): 
        self.population = []    # 族群
        self.mutationRate = 0.1 # 突變率

    def run(self, size, maxGen) :  # 遺傳演算法主程式
        self.population = self.newPopulation(size) # 產生初始族群
        for t in range(maxGen):  # 最多產生 maxGen 代
            print("============ generation", t, "===============")
            self.population = self.reproduction() # 產生下一代
            self.dump() # 印出目前族群
  
    def newPopulation(self, size): 
        newPop=[]
        for _ in range(size): 
            chromosome = self.randomChromosome() # 隨機產生新染色體
            newPop.append({'chromosome':chromosome, 
                           'fitness':self.calcFitness(chromosome)})
        newPop.sort(key = lambda c: c['fitness']) # 對整個族群進行排序
        return newPop
  
    # 輪盤選擇法: 隨機選擇一個個體 -- 落點在 i*i ~ (i+1)*(i+1) 之間都算是 i
    def selection(self): 
        n = len(self.population)
        shoot  = random.randint(0, (n*n/2)-1)
        select = math.floor(math.sqrt(shoot*2))
        return self.population[select]
  
    # 產生下一代
    def reproduction(self):
        newPop = []
        for i in range(len(self.population)): 
            parent1 = self.selection()['chromosome'] # 選取父親
            parent2 = self.selection()['chromosome'] # 選取母親
            chromosome = self.crossover(parent1, parent2) # 父母交配，產生小孩
            prob = random.random()
            if prob < self.mutationRate: # 有很小的機率
                chromosome = self.mutate(chromosome) # 小孩會突變
            newPop.append({ 'chromosome':chromosome, 'fitness':self.calcFitness(chromosome) })  # 將小孩放進下一代族群裡
        newPop.sort(key = lambda c: c['fitness']) # 對新一代根據適應性（分數）進行排序
        return newPop
  
    def dump(self):  # 印出一整代成員
        for i in range(len(self.population)):
            print(i, self.population[i])
```

> keyGa.py

```python
from geneticAlgorithm import GeneticAlgorithm
import random

class KeyGA(GeneticAlgorithm):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def randomChromosome(self): # 隨機產生一個染色體 (一個 16 位元的 01 字串)
        bits=[]
        for _ in range(len(self.key)):
            bit = str(random.randint(0,1))
            bits.append(bit)
        return ''.join(bits)
  
    def calcFitness(self, c): # 分數是和 key 一致的位元個數
        fitness=0  # 分數
        for i in range(len(self.key)):
            fitness += 1 if c[i]==self.key[i] else 0
        return fitness
  
    def crossover(self, c1, c2):
        cutIdx = random.randint(0, len(c1)-1)
        head   = c1[0:cutIdx]
        tail   = c2[cutIdx:]
        return head + tail
    
    def mutate(self, chromosome): # 突變運算
        i=random.randint(0, len(chromosome)-1) # 選擇突變點
        cMutate = chromosome[0:i]+random.choice(['0','1'])+chromosome[i+1:] # 在突變點上隨機選取 0 或 1
        return cMutate # 傳回突變後的染色體

# 執行遺傳演算法，企圖找到 key，最多執行20代，每代族群都是一百人
kga = KeyGA("1010101010101010")
kga.run(100, 20)
```

