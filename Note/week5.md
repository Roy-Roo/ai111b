# Week5



### 梯度下降法

* 要找最低點(類似模擬退火法)
* 梯度為斜率的方向

> vecGradient.py

* 梯度(偏微分方向)下降，朝逆梯度方向走，直到斜率為0，步數太小會走太久走不到，步數太大則會亂跳

```python
step = 0.01

# 我們想找函數 f 的最低點
def f(p):
    [x,y] = p
    return x * x + y * y

# df(f, p, k) 為函數 f 對變數 k 的偏微分: df / dp[k]
# 例如在上述 f 範例中 k=0, df/dx, k=1, df/dy
def df(f, p, k):
    p1 = p.copy()
    p1[k] += step
    return (f(p1) - f(p)) / step

# 函數 f 在點 p 上的梯度
def grad(f, p):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k)
    return gp

[x,y] = [1,3]
print('x=', x, 'y=', y)
print('df(f(x,y), 0) = ', df(f, [x, y], 0))
print('df(f(x,y), 1) = ', df(f, [x, y], 1))
print('grad(f)=', grad(f, [x,y]))
```



### 偏微分

* 一個多變數的函數對其中一個變數（導數）微分，而保持其他變數恆定



### 反傳遞

> karpathy/micrograd/blob/master/micrograd/engine.py

* 正傳遞傳遞的是值
* 反傳遞傳遞的是梯度

```python
# 正傳遞 : out = self.data * other.data
# 反傳遞 : self.grad += other.data * out.grad
#         other.grad += self.data * out.grad
#         out._backward = _backward
def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
```



