# Week 1



* 用電腦模擬神經細胞，簡化成簡單的數學運算



### 梯度下降法

* 計算量會變大，速度會越來越慢



### Gym

* 強化學習套件，由 OpenAI 所釋出
* 提供強化學習測試環境，類似 imagenet 在影像辨識中的角色



### 爬山演算法

* 使用迴圈查看附近有沒有比X更好的解，如果有就更換，直到旁邊的解都比現在差時，就回傳X的值
* 只能找到「局部最佳解」(local optimal)，當整個空間有很多山頂的時候，這種方法會爬到其中一個山頂就停了，並不一定會爬到最高的山頂

> 簡易版本

```python
Algorithm HillClimbing(f, x)
  x = 隨意設定一個解。
  while (x 有鄰居 x' 比 x 更高)
    x = x';
  end
  return x;
end
```

> 通用爬山框架

```python
# 通常僅需修改height、neighbor即可
def hillClimbing(x, height, neighbor, max_fail=10000): # 預設失敗10000會離開迴圈
    fail = 0 # 失敗次數
    while True:
        nx = neighbor(x)
        if height(nx)>height(x):
            x = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return x
```

> root.py

```python
from hillClimbing import hillClimbing
from random import uniform
from math import sqrt 

def complexNorm(x):
    return sqrt(x.real**2+x.imag**2)

def polynomialEval(a, x):
    n = len(a)
    r = 0
    for i in range(n):
        r += a[i]*(x**i)
    return r

def polyHeight(a, x):
    return -1.0*complexNorm(polynomialEval(a, x))

def complexNeighbor(x, h=0.001):
    dx = uniform(-h, h)*1+uniform(-h, h)*1j
    return x+dx

def polynomialRoot(a, h=0.001):  # 呼叫爬山演算法，起始點為0
    return hillClimbing(
        0+0j, 
        lambda x:polyHeight(a,x),  # 高度演算法? x在哪裡填入
        complexNeighbor  # 鄰居演算法
    )

# 0j
print(f'polynomial_eval(x*2-2x+1)=', polynomialEval([1,-2,1], 1.0+0j)) 
# (2.8263503200760516e-06+0.9999913543152901j)
print(f'polynomial_root(x*2+0x+1)=', polynomialRoot([1,0,1]))
# (-1.8929617994654685e-05+0.9999961982473808j)
print(f'polynomial_root(x**4-3x**2-4)=', polynomialRoot([-4, 0, -3, 0, 1]))
```

> 利用爬山排課表

* 給出分數的高度(算分數)和鄰居函數(搜尋)(找鄰居的方法，紀錄附近的值)

```python
from random import random, randint, choice
from solution import Solution
import numpy as np

courses = [
{'teacher': '  ', 'name':'　　', 'hours': -1},
{'teacher': '甲', 'name':'機率', 'hours': 2},
{'teacher': '甲', 'name':'線代', 'hours': 3},
{'teacher': '甲', 'name':'離散', 'hours': 3},
{'teacher': '乙', 'name':'視窗', 'hours': 3},
{'teacher': '乙', 'name':'科學', 'hours': 3},
{'teacher': '乙', 'name':'系統', 'hours': 3},
{'teacher': '乙', 'name':'計概', 'hours': 3},
{'teacher': '丙', 'name':'軟工', 'hours': 3},
{'teacher': '丙', 'name':'行動', 'hours': 3},
{'teacher': '丙', 'name':'網路', 'hours': 3},
{'teacher': '丁', 'name':'媒體', 'hours': 3},
{'teacher': '丁', 'name':'工數', 'hours': 3},
{'teacher': '丁', 'name':'動畫', 'hours': 3},
{'teacher': '丁', 'name':'電子', 'hours': 4},
{'teacher': '丁', 'name':'嵌入', 'hours': 3},
{'teacher': '戊', 'name':'網站', 'hours': 3},
{'teacher': '戊', 'name':'網頁', 'hours': 3},
{'teacher': '戊', 'name':'演算', 'hours': 3},
{'teacher': '戊', 'name':'結構', 'hours': 3},
{'teacher': '戊', 'name':'智慧', 'hours': 3}
]

teachers = ['甲', '乙', '丙', '丁', '戊']

rooms = ['A', 'B']

slots = [
'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27',
'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37',
'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47',
'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57',
]

cols = 7

def randSlot() :
    return randint(0, len(slots)-1)

def randCourse() :
    return randint(0, len(courses)-1)


class SolutionScheduling(Solution) :
    def neighbor(self):    # 單變數解答的鄰居函數。
        fills = self.v.copy()
        choose = randint(0, 1)
        if choose == 0: # 任選一個改變 
            i = randSlot()
            fills[i] = randCourse()
        elif choose == 1: # 任選兩個交換
            i = randSlot()
            j = randSlot()
            t = fills[i]
            fills[i] = fills[j]
            fills[j] = t
        return SolutionScheduling(fills)                  # 建立新解答並傳回。

    def height(self) :      # 高度函數
        courseCounts = [0] * len(courses)
        fills = self.v
        score = 0
        # courseCounts.fill(0, 0, courses.length)
        for si in range(len(slots)):
            courseCounts[fills[si]] += 1
            #                        連續上課:好                   隔天:不好     跨越中午:不好
            if si < len(slots)-1 and fills[si] == fills[si+1] and si%7 != 6 and si%7 != 3:
                score += 0.1
            if si % 7 == 0 and fills[si] != 0: # 早上 8:00: 不好
                score -= 0.12
        
        for ci in range(len(courses)):
            if (courses[ci]['hours'] >= 0):
                score -= abs(courseCounts[ci] - courses[ci]['hours']) # 課程總時數不對: 不好
        return score

    def str(self) :    # 將解答轉為字串，以供印出觀察。
        outs = []
        fills = self.v
        for i in range(len(slots)):
            c = courses[fills[i]]
            if i%7 == 0:
                outs.append('\n')
            outs.append(slots[i] + ':' + c['name'])
        return 'height={:f} {:s}\n\n'.format(self.height(), ' '.join(outs))
    
    @classmethod
    def init(cls):
        fills = [0] * len(slots)
        for i in range(len(slots)):
            fills[i] = randCourse()
        return SolutionScheduling(fills)
```



### 模擬退火法(鑄刀法)

* 找谷底，在高溫的時候可以允許亂跑，低溫的時候不允許亂跑

```python
from solution import Solution
from random import random, randint

class SolutionArray(Solution):
    def neighbor(self):           #  多變數解答的鄰居函數。
        nv = self.v.copy()        #  nv=v.clone()=目前解答的複製品
        i = randint(0, len(nv)-1) #  隨機選取一個變數
        if (random() > 0.5):      #  擲骰子決定要往左或往右移
            nv[i] += self.step
        else:
            nv[i] -= self.step
        return SolutionArray(nv)  #  傳回新建的鄰居解答。

    def energy(self): #  能量函數
        x, y, z =self.v
        return x*x+3*y*y+z*z-4*x-3*y-5*z+8 #  (x^2+3y^2+z^2-4x-3y-5z+8)

    def str(self):    #  將解答轉為字串的函數，以供列印用。
        return "energy({:s})={:f}".format(str(self.v), self.energy())
```

