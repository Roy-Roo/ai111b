# Week15



### Classification

* 是一種監督式學習任務，它將輸入數據分配到預先定義的類別或標籤中的一個。分類問題的目標是根據給定的訓練數據集，構建一個模型，能夠將新的未標籤數據映射到正確的類別

| 術語 | 解釋                               |
| ---- | ---------------------------------- |
| TN   | 實際類別為負，且模型正確地預測為負 |
| TP   | 實際類別為正，且模型正確地預測為正 |
| FN   | 實際類別為正，但模型錯誤地預測為負 |
| FP   | 實際類別為負，但模型錯誤地預測為正 |

* Precision = `TP / (TP + FP)` 
  * 評估模型的準確性和預測正確性，特別是在類別不平衡的情況下。較高的精確率表示模型對於將負樣本誤分類為正樣本的能力較低
* Recall(FPR) = `TP / (TP + FN)` 
  * 評估模型的敏感性和能夠檢測出正樣本的能力，特別是在類別不平衡的情況下。較高的召回率表示模型對於錯誤地將正樣本分類為負樣本的風險較低
* F1 - Score = `2 * (Precision + Recall) / (Precision + Recall)`
  * 評估模型在保持精確性的同時能夠有效檢測出真實正樣本的能力。較高的F1值表示模型在精確率和召回率之間取得了較好的平衡
  * 只考慮了正樣本的預測能力，對於負樣本的預測能力未納入考慮
* Accuracy = `(TP + TN) / (TP + TN + FP + TN)` 
  * 評估模型整體的預測能力，特別是在類別平衡的情況下。較高的準確率表示模型對於所有類別的預測都較為準確
  * 準確率在處理類別不平衡的分類問題時可能存在偏誤。當某一類別的樣本數量較少或重要性較高時，準確率可能無法提供全面的評估



> main.py

```python
import ml
# 要使用哪一個就把目前的註解掉，然後再開啟想要用的
#  x_train,x_test,y_train,y_test=ml.load_csv('../csv/cmc.csv', 'ContraceptiveMethodUsed')
# 這裡直接分訓練集8成，測試集2成
x_train,x_test,y_train,y_test=ml.load_csv('../csv/iris.csv', 'target')
# from decision_tree_classifier import train_classifier
# from mlp_classifier import learn_classifier
# from sgd_classifier import learn_classifier
# from gnb_classifier import learn_classifier
from knn_classifier import learn_classifier
# from svm_classifier import learn_classifier
# from ovr_classifier import learn_classifier
# from random_forest_classifier import learn_classifier
classifier = learn_classifier(x_train, y_train)
print('=========== train report ==========')
ml.report(classifier, x_train, y_train)  # 報告訓練集
print('=========== test report ==========')
ml.report(classifier, x_test, y_test)  # 會映出混淆矩陣
```



### Softmax

* 在機器學習中廣泛應用於多類別分類模型，例如神經網絡中的分類層，其中Softmax函數將神經元的輸出轉換為類別的概率。通過將Softmax函數應用於模型的輸出，可以得到每個類別的概率分佈，從而進行預測和分類

> test01.py

```python
import numgd as ngd
import soft as so
import numpy as np

x = np.array([0.3, 0.5, 0.2])
y = np.array([0.0, 1.0, 0.0])
print('x =', x)
print('y =', y)

s = so.softmax(x)
print('s = softmax(x) =', s)

print('jacobian_softmax(s)=\n', so.jacobian_softmax(s))
print('cross_entropy(y, s)=', so.cross_entropy(y, s))

def num_gradient_cross_entropy(y, s):
    return ngd.grad(lambda s:so.cross_entropy(y, s), s)

print('    gradient_cross_entropy(y, s)=', so.gradient_cross_entropy(y, s))
print('num_gradient_cross_entropy(y, s)=', num_gradient_cross_entropy(y, s))

def loss(y, x):
    s = so.softmax(x)
    return so.cross_entropy(y, s)

def num_error_softmax_input(y, x):
    return ngd.grad(lambda x:loss(y, x), x)

print('    error_softmax_input(y, s)=', so.error_softmax_input(y, s))  # 驗證公式
print('num_error_softmax_input(y, x)=', num_error_softmax_input(y, x))
```

> mnist.py

```python
from macrograd import Tensor

from keras.datasets import mnist
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = np.asarray(x_train, dtype=np.float32) / 255.0
test_images = np.asarray(x_test, dtype=np.float32) / 255.0
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
y_train = keras.utils.to_categorical(y_train)

def forward(X,Y,W):
    y_predW = X.matmul(W)
    probs = y_predW.softmax()
    loss = probs.cross_entropy(Y)
    return loss

batch_size = 32
steps = 20000

X = Tensor(train_images); Y = Tensor(y_train) # 全部資料
# new initialized weights for gradient descent
Wb = Tensor(np.random.randn(784, 10))
for step in range(steps):
    ri = np.random.permutation(train_images.shape[0])[:batch_size]
    Xb, yb = Tensor(train_images[ri]), Tensor(y_train[ri]) # Batch 資料
    lossb = forward(Xb, yb, Wb)  # cross entropy
    lossb.backward()
    if step % 1000 == 0 or step == steps-1:
        loss = forward(X, Y, Wb).data/X.data.shape[0]
        print(f'loss in step {step} is {loss}')
    Wb.data = Wb.data - 0.01*Wb.grad # update weights, 相當於 optimizer.step()
    Wb.grad = 0  # 梯度歸0
```



### Cluster

* 無監督學習的方法，其目標是根據資料的相似性將資料點分組。這些分組稱為群集或簇。每個群集內的資料點應該彼此相似，而不同群集之間的資料點則應該有明顯的區別。

> cluster_data1.py

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
n = 300
X, y = datasets.make_blobs(n_samples=n, centers=4, cluster_std=0.60, random_state=0)
# X, y = datasets.make_moons(n_samples=n, noise=0.1)
# X, y = datasets.make_circles(n_samples=n, noise=0.1, factor=0.5)
# X, y = np.random.rand(n, 2), None
plt.scatter(X[:, 0], X[:, 1]) # , s=50
plt.show()
```

