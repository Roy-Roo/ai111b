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