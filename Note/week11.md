# Week11



* 目前無法把全部東西都交給 AI 來做，AI 會亂講話，這是 Transformer 模型的缺點



### LangChain

* 補強 Open AI 不會做的東西，自己補上套件，搜尋使用正規表達式
* 新的神經網路不確定因素太多，導致會輸出意想不到的答案，而舊的可以讓我們有固定答案，才可以放到一些自動化流程上



### 格狀語法

* 格狀與法是語言的核心，每個動詞或是名詞都有多個語意，主語、賓語(受詞)、地點等等，可以使用格狀語法，讓程式碼更了解輸入者的語意或想法



### SCIgen

* 自動生成論文被期刊錄用



### textgen

* 傳統作法，不使用生成語法，用名人名言來生成一段文章

> textgen.py

```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, re
import random,readJSON

data = readJSON.读JSON文件("data.json")
名人名言 = data["famous"] # a 代表前面垫话，b代表后面垫话
前面垫话 = data["before"] # 在名人名言前面弄点废话
后面垫话 = data['after']  # 在名人名言后面弄点废话
废话 = data['bosh'] # 代表文章主要废话来源

xx = "学生会退会"

重复度 = 2

def 洗牌遍历(列表):
    global 重复度
    池 = list(列表) * 重复度
    while True:
        random.shuffle(池)
        for 元素 in 池:
            yield 元素

下一句废话 = 洗牌遍历(废话)
下一句名人名言 = 洗牌遍历(名人名言)

def 来点名人名言():
    global 下一句名人名言
    xx = next(下一句名人名言)
    xx = xx.replace(  "a",random.choice(前面垫话) )
    xx = xx.replace(  "b",random.choice(后面垫话) )
    return xx

def 另起一段():
    xx = ". "
    xx += "\r\n"
    xx += "    "
    return xx

if __name__ == "__main__":
    xx = input("请输入文章主题:")
    for x in xx:
        tmp = str()
        while ( len(tmp) < 6000 ) :
            分支 = random.randint(0,100)
            if 分支 < 5:
                tmp += 另起一段()
            elif 分支 < 20 :
                tmp += 来点名人名言()
            else:
                tmp += next(下一句废话)
        tmp = tmp.replace("x",xx)
        print(tmp)
```

