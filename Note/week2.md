# Week2



### 爬山演算法

* 基於啟發式搜尋 (heuristic search) 的最佳化演算法，用於求解在某個函數空間中的最大值或最小值。這個演算法模擬登山過程，從當前位置出發，尋找相對於當前位置高峰的方向前進，重複這個過程，直到到達局部極大值或全域最大值
* 算分數+搜尋，如果只有搜尋，那就是改良法
* 容易陷入局部最優解而無法跳出
* 幾乎所有問題都能用爬山演算法解決，缺點則是速度太慢
* 在達到相同目的的效果上，參數越小越好，因為太多參數會 overfitting，而且參數小的泛用性會比較高

> 雙變數

```python
import random

def hillClimbing(f, x, y, h=0.01):
    failCount = 0                    # 失敗次數歸零
    while (failCount < 10000):       # 如果失敗次數小於一萬次就繼續執行
        fxy = f(x, y)                # fxy 為目前高度
        dx = random.uniform(-h, h)   # dx 為左右偏移量
        dy = random.uniform(-h, h)   # dy 為前後偏移量
        if f(x+dx, y+dy) >= fxy:     # 如果移動後高度比現在高
            x = x + dx               #   就移過去
            y = y + dy
            print('x={:.3f} y={:.3f} f(x,y)={:.3f}'.format(x, y, fxy))
            failCount = 0            # 失敗次數歸零
        else:                        # 若沒有更高
            failCount = failCount + 1#   那就又失敗一次
    return (x,y,fxy)                 # 結束傳回 （已經失敗超過一萬次了）

def f(x, y):
    return -1 * ( x*x -2*x + y*y +2*y - 8 )

hillClimbing(f, 0, 0)
```

> 通用框架

```python
def hillClimbing(x, height, neighbor, max_fail=10000):
    fail = 0
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

> 用物件導向來寫爬山演算法(主程式)

```python
def hillClimbing(s, maxGens, maxFails):   # 爬山演算法的主體函數
    print("start: ", s.str())             # 印出初始解
    fails = 0                             # 失敗次數設為 0
    # 當代數 gen<maxGen，且連續失敗次數 fails < maxFails 時，就持續嘗試尋找更好的解。
    for gens in range(maxGens):
        snew = s.neighbor()               #  取得鄰近的解
        sheight = s.height()              #  sheight=目前解的高度
        nheight = snew.height()           #  nheight=鄰近解的高度
        if (nheight >= sheight):          #  如果鄰近解比目前解更好
            print(gens, ':', snew.str())  #    印出新的解
            s = snew                      #    就移動過去
            fails = 0                     #    移動成功，將連續失敗次數歸零
        else:                             #  否則
            fails = fails + 1             #    將連續失敗次數加一
        if (fails >= maxFails):
            break
    print("solution: ", s.str())          #  印出最後找到的那個解
    return s                              #    然後傳回。
```



### 模擬退火法

* 自冶金學的專有名詞退火。退火是將材料加熱後再經特定速率冷卻，目的是增大晶粒的體積，並且減少晶格中的缺陷。

> annealing.py

```python
import math
import random

def P(e, enew, T): # 模擬退火法的機率函數
    if (enew < e):
        return 1
    else:
        return math.exp((e-enew)/T)

def annealing(s, maxGens) : # 模擬退火法的主要函數
    sbest = s                              # sbest:到目前為止的最佳解
    ebest = s.energy()                     # ebest:到目前為止的最低能量
    T = 100                                # 從 100 度開始降溫
    for gens in range(maxGens):            # 迴圈，最多作 maxGens 這麼多代。
        snew = s.neighbor()                # 取得鄰居解
        e    = s.energy()                  # e    : 目前解的能量
        enew = snew.energy()               # enew : 鄰居解的能量
        T  = T * 0.995                     # 每次降低一些溫度
        if P(e, enew, T)>random.random():  # 根據溫度與能量差擲骰子，若通過
            s = snew                       # 則移動到新的鄰居解
            print("{} T={:.5f} {}".format(gens, T, s.str())) # 印出觀察

        if enew < ebest:                 # 如果新解的能量比最佳解好，則更新最佳解。
            sbest = snew
            ebest = enew
    
    print("solution: {}", sbest.str())     # 印出最佳解
    return sbest                           # 傳回最佳解
```

> solution.py

```python
class Solution: # 解答的物件模版 (類別)
    def __init__(self, v, step = 0.01):
        self.v = v       # 參數 v 為解答的資料結構
        self.step = step # 每一小步預設走的距離

    # 以下兩個函數至少需要覆蓋掉一個，否則會無窮遞迴
    def height(self): # 爬山演算法的高度函數
        return -1*self.energy()               # 高度 = -1 * 能量

    def energy(self): # 尋找最低點的能量函數
        return -1*self.height()               # 能量 = -1 * 高度
```

> solutionArray.py

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

    def energy(self): #  能量函數，要解的目標方程式
        x, y, z =self.v
        return x*x+3*y*y+z*z-4*x-3*y-5*z+8 #  (x^2+3y^2+z^2-4x-3y-5z+8)

    def str(self):    #  將解答轉為字串的函數，以供列印用。
        return "energy({:s})={:f}".format(str(self.v), self.energy())
```

