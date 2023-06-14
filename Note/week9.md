# Week9



### 演算法種類

* 優化 = AI = 算分數 + 搜尋
  * 算分數 : lost = 能量 = - 高度

* 爬山

* 遺傳
* 梯度下降法(找能量最低)
* 反傳遞
* 模型
  * MLP
  * CNN
  * RNN
  * Transformer 內的 `Attention` 概念 => open AI
  * ChatGPT



### AI下棋

* 1997 AI深藍 西洋棋
* 下棋是使用算分數做處理
  * 給定各種類型的棋分數，每一手都掃描所有棋盤，取分數最高的當作做落子點，計算棋子的攻擊力和防禦力(攻擊得幾分+防禦讓對方少幾分)

```
ooooo  ->  1000分
oooo  ->  30分
ooo  ->  10分
oo ->  3分
o ->  1分
```

> gomoku.py

* ref : [五子棋遊戲，單機命令列版 -- 作者：陳鍾誠](https://github.com/cccbook/py2cs/tree/master/03-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7/07-%E9%9B%BB%E8%85%A6%E4%B8%8B%E6%A3%8B)

* 五子棋下棋程式碼，16*16的棋盤，下棋打入橫軸縱軸

```python
'''
五子棋遊戲，單機命令列版 -- 作者：陳鍾誠

人對人下  ：python gomoku.py P P
人對電腦  ：python gomoku.py P C
電腦對電腦：python gomoku.py C C
'''

import sys
import time
#  棋盤物件
class Board:

    def __init__(self, rMax, cMax):
        self.m = [None] * rMax
        self.rMax = rMax
        self.cMax = cMax
        for r in range(rMax):
            self.m[r] = [None] * cMax
            for c in range(cMax):
                self.m[r][c] = '-'

    #  將棋盤格式化成字串
    def __str__(self):
        b = []
        b.append('  0 1 2 3 4 5 6 7 8 9 a b c d e f')
        for r in range(self.rMax):
            b.append('{:x} {:s} {:x}'.format(r, ' '.join(self.m[r]), r))
            # r.toString(16) + ' ' + self.m[r].join(' ') + ' ' + r.toString(16) + '\n'

        b.append('  0 1 2 3 4 5 6 7 8 9 a b c d e f')
        return '\n'.join(b)

    #  顯示棋盤
    def show(self):
        print(str(self))

#  以下為遊戲相關資料與函數
#  zero = [ 0, 0, 0, 0, 0]
#  inc  = [-2,-1, 0, 1, 2]
#  dec  = [ 2, 1, 0,-1,-2]  
# 利用下面的資訊可以講查橫軸縱軸
z9 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
i9 = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
d9 = [4, 3, 2, 1, 0, -1, -2, -3, -4]
z5 = [0, 0, 0, 0, 0]
i2 = i9[2:-2]
d2 = d9[2:-2]

#  檢查在 (r, c) 這一格，規則樣式 (dr, dc) 是否被滿足
#  dr, dc 的組合可用來代表「垂直 | , 水平 - , 下斜 \ , 上斜 /」。
def patternCheck(board, turn, r, c, dr, dc):
    for i in range(len(dr)):
        tr = round(r + dr[i])
        tc = round(c + dc[i])
        if tr < 0 or tr >= board.rMax or tc < 0 or tc >= board.cMax:
            return False
        v = board.m[tr][tc]
        if (v != turn):
            return False
    
    return True

#  檢查是否下 turn 這個子的人贏了。
def winCheck(board, turn):
    win = False
    tie = True
    for r in range(board.rMax):
        for c in range(board.cMax):
            tie = False if board.m[r][c] == '-' else tie
            win = True if patternCheck(board, turn, r, c, z5, i2) else win #  水平 -
            win = True if patternCheck(board, turn, r, c, i2, z5) else win #  垂直 |
            win = True if patternCheck(board, turn, r, c, i2, i2) else win #  下斜 \
            win = True if patternCheck(board, turn, r, c, i2, d2) else win #  上斜 /
    if (win):
        print('{} 贏了！'.format(turn))  #  如果贏了就印出贏了
        sys.exit() #  然後離開。

    if (tie):
        print('平手')
        sys.exit(0) #  然後離開。

    return win

attackScores = [0, 3, 10, 30, 100, 500]  # 0子、1子、2...
guardScores = [0, 2, 9, 25, 90, 400]  # 防守分數
attack = 1
guard = 2

def getScore(board, r, c, turn, mode):
    score = 0
    mScores = attackScores if mode == attack else guardScores
    board.m[r][c] = turn
    for start in range(5):
        for len1 in reversed(range(5)):
            length = len1 + 1
            zero = z9[start: start + length]  # 從start開始向後取length
            inc  = i9[start: start + length]
            dec  = d9[start: start + length]
            if patternCheck(board, turn, r, c, zero, inc):
                score += mScores[length] #  得分：垂直 |
            if patternCheck(board, turn, r, c, inc, zero):
                score += mScores[length] #  得分：水平 -
            if patternCheck(board, turn, r, c, inc, inc):
                score += mScores[length] #  得分：下斜 \
            if patternCheck(board, turn, r, c, inc, dec):
                score += mScores[length] #  得分：上斜 /

    if r == 0 or r == board.rMax:
        score = score - 1
    if c == 0 or c == board.cMax:
        score = score - 1
    board.m[r][c] = '-'
    return score

def peopleTurn(board, turn):
    try:
        xy = input('將 {} 下在: '.format(turn))
        r = int(xy[0], 16) #  取得下子的列 r (row)  
        c = int(xy[1], 16) #  取得下子的行 c (column)
        if r < 0 or r > board.rMax or c < 0 or c > board.cMax: #  檢查是否超出範圍
            raise Exception('(row, col) 超出範圍!') #  若超出範圍就丟出例外，下一輪重新輸入。
        if board.m[r][c] != '-': #  檢查該位置是否已被佔據
            raise Exception('({}{}) 已經被佔領了!'.format(xy[0], xy[1])) #  若被佔據就丟出例外，下一輪重新輸入。
        board.m[r][c] = turn #  否則、將子下在使用者輸入的 (r,c) 位置
    except Exception as error:
        print(error)
        peopleTurn(board, turn)

def computerTurn(board, turn):
    best = {'r': 0, 'c': 0, 'score': -1}
    for r in range(board.rMax):
        for c in range(board.cMax):
            if (board.m[r][c] != '-'):
                continue
            enermy = 'o' if turn == 'x' else 'x'
            attackScore = getScore(board, r, c, turn, attack)  #  攻擊分數
            guardScore = getScore(board, r, c, enermy, guard)   #  防守分數
            score = attackScore + guardScore
            if r==8 and c==8: # 電腦若是第一手應該下 (8,8)
                score += 1
            if score > best['score']:
                best['r'] = r
                best['c'] = c
                best['score'] = score

    board.m[best['r']][best['c']] = turn #  將子下在分數最高的位置

def chess(o, x):
    b = Board(16, 16) #  建立棋盤
    b.show()            #  顯示棋盤
    while (True):
        if o.upper()=='P':
            peopleTurn(b, 'o')
        else:
            computerTurn(b, 'o')
        b.show()         #  顯示棋盤現況
        winCheck(b, 'o') #  檢查下了這子之後是否贏了！
        time.sleep(2)
        if x.upper()=='P':
            peopleTurn(b, 'x')
        else:
            computerTurn(b, 'x')
        b.show()
        winCheck(b, 'x')
        time.sleep(2)  # 休息兩秒，不要讓程式碼跑太快

o, x = sys.argv[1], sys.argv[2]  # 取得下棋的人是
chess(o, x)
```

![](https://github.com/cccbook/py2cs/blob/master/03-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7/07-%E9%9B%BB%E8%85%A6%E4%B8%8B%E6%A3%8B/12-chess/img/Minimax.jpg?raw=true)

```python
# depth: 目前還要下幾層，maximizingPlayer 取大還是取小，這個會算很久
function minimax(node, depth, maximizingPlayer)  
    if depth = 0 or node is a terminal node
        return the heuristic value of node
    if maximizingPlayer
        bestValue := -∞
        for each child of node
            val := minimax(child, depth - 1, FALSE))
            bestValue := max(bestValue, val);
        return bestValue
    else
        bestValue := +∞
        for each child of node
            val := minimax(child, depth - 1, TRUE))
            bestValue := min(bestValue, val);
        return bestValue

(* Initial call for maximizing player *)
minimax(origin, depth, TRUE)  # 目前應該下哪個節點
```

