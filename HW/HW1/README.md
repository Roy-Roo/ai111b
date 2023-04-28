# HW 1



### 題目

> 請寫一個程式可以做線性回歸
>
> 線性回歸： 給定一些 (x,y) 值，請找出最好的 a,b 值，使得
>
> y = a + bx
>
> 與所有點的 y 軸距離總和最小。

 

### 程式碼

```python
import matplotlib.pyplot as plt
import numpy as np

# 題目給定的(x, y)座標
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

# 預測線性回歸的值, y = a + bx 
def predict(a, b,  xt):
	return a + b * xt

# 計算均方差
def MSE(a, b, x, y):
	total = 0
	for i in range(len(x)):
		total += (y[i]-predict(a,b, x[i]))**2
	return total

# 均方差計算的損失
def loss(a, b):
	return MSE(a, b, x, y)

# p = [0.0, 0.0]
# plearn = optimize(loss, p, max_loops=3000, dump_period=1)
# 嘗試在四個方向上更新 p[0], p[1], 如果新位置的損失更小就移動到新位置
def optimize():
    # 請修改這個函數，自動找出讓 loss 最小的 p 
    p = [0.0, 0.0]
    h = 0.001
    
    while (True):
        if (loss(p[0] + h, p[1]) < loss(p[0], p[1])):
            p[0] = p[0] + h
        elif (loss(p[0] - h, p[1]) < loss(p[0], p[1])):
            p[0] = p[0] - h
        elif (loss(p[0], p[1] + h) < loss(p[0], p[1])):
            p[1] = p[1] + h
        elif (loss(p[0], p[1] - h) < loss(p[0], p[1])):
            p[1] = p[1] - h
        else:
            break
    return p
    # p = [2,1]  這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)
    # p = [3,2] # 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)
    
    # return p

p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0]+p[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
```



### 結果

![](D:\大學\大三\大三下\人工智慧\HW\HW1\HW1.jpg)

![](D:\大學\大三\大三下\人工智慧\HW\HW1\HW1-2.jpg)



### 參考

* [regression.py](https://github.com/ccc111b/py2cs/blob/master/03-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7/A3-%E7%BF%92%E9%A1%8C/01-regression/regression.py)

* [hillClimbing2.py](https://github.com/ccc111b/py2cs/blob/master/03-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7/01-%E5%84%AA%E5%8C%96/01-%E7%88%AC%E5%B1%B1%E6%BC%94%E7%AE%97%E6%B3%95/02-%E9%9B%99%E8%AE%8A%E6%95%B8%E5%87%BD%E6%95%B8%E7%9A%84%E7%88%AC%E5%B1%B1/hillClimbing2.py)

* [iwantall2333](https://github.com/iwantall2333/ai111b/blob/main/%E7%BF%92%E9%A1%8C1.py)
* [stereomp3](https://github.com/stereomp3/ai111b/blob/main/AI/work/climbToLine/climbToLine.py)

* ChatGPT
