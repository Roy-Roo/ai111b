# Week6



### 人工智慧學習資料

1. [karpathy github](https://github.com/karpathy)
   * 他把人工智慧的東西寫的很簡單，如果要學習GPT可以看nanoGPT和minGPT

2. https://mofanpy.com/
   * 這個網站有教很多人工智慧，有興趣可以看看

3. https://gymnasium.farama.org/
   * 可以學習強化學習如何做，通常都是使用馬可夫決策理論

4. https://github.com/openai/baselines
   * 學習強化學習底層實作



> graph_search.py

* 深度和廣度優先搜尋 

```python
def enqueue(a, o):
    a.insert(0, o)

def dequeue(a):
    return a.pop()

g = { #  graph: 被搜尋的網路
    '1': {'n':['2','5'], 'v':0}, #  n: neighbor (鄰居), v: visited (是否被訪問過)
    '2': {'n':['3','4'], 'v':0},
    '3': {'n':['4','5','6'], 'v':0},
    '4': {'n':['5','6'], 'v':0},
    '5': {'n':['6'], 'v':0},
    '6': {'n':[], 'v':0}
}

def init(g): #  初始化、設定 visited 為 0
    for i in g:
        g[i]['v'] = 0

def dfs(g, node): #  深度優先搜尋
    if g[node]['v']!=0:           #  如果已訪問過，就不再訪問
        return
    print(node, '=> ', end = '')  #  否則、印出節點
    g[node]['v'] = 1              #    並設定為已訪問
    neighbors = g[node]['n']      # 取出鄰居節點
    for n in neighbors:           #  對於每個鄰居
        dfs(g, n)                 #    逐一進行訪問

queue=['1'] #  BFS 用的 queue, 起始點為 1。

def bfs(g, q): #  廣度優先搜尋
    if len(q)==0:                 #  如果 queue 已空，則返回。
        return
    node = dequeue(q)             #  否則、取出 queue 的第一個節點。
    if g[node]['v'] == 0:         #  如果該節點尚未拜訪過。
        g[node]['v'] = 1          #  標示為已拜訪
    else:                         #  否則 (已訪問過)
        return                    #  不繼續搜尋，直接返回。
    print(node, '=> ', end = '')  #  印出節點
    neighbors = g[node]['n']      #  取出鄰居。
    for n in neighbors: #  對於每個鄰居
        if not g[n]['v']:         #  假如該鄰居還沒被拜訪過
            enqueue(q, n)         #    就放入 queue 中
    bfs(g, q)

print('dfs:', end = '')
init(g)
dfs(g, '1') # 呼叫深度優先搜尋。
print('')
```



### 習題

> 以深度優先搜尋解決老鼠走迷宮問題

```javascript
var log = console.log;

function matrixPrint(m) {
  for(var i=0;i<m.length;i++)
    log(m[i]);
}

function strset(s, i, c) {
  return s.substr(0, i) + c + s.substr(i+1);
}

function findPath(m, x, y) {
  log("=========================");
  log("x="+x+" y="+y);
  matrixPrint(m);
  if (x>=6||y>=8) return false;
  if (m[x][y] == '*') return false;
  if (m[x][y] == '+') return false;
  if (m[x][y] == ' ') m[x] = strset(m[x], y, '.');
  if (m[x][y] == '.' && (x == 5 || y==7)) 
    return true;
  if (y<7&&m[x][y+1]==' ') //向右
    if (findPath(m, x,y+1)) return true;
  if(x<5&&m[x+1][y]==' ') //向下
    if (findPath(m, x+1,y)) return true;
  if(y>0&&m[x][y-1]==' ') //向左
    if (findPath(m, x,y-1)) return true;
  if(x>0&&m[x-1][y]==' ') //向上
    if (findPath(m, x-1,y)) return true;
  m[x][y]='+';
  return false;
}

var m =["********", 
        "** * ***",
        "     ***",
        "* ******",
        "*     **",
        "***** **"];
	
findPath(m, 2, 0);
log("=========================");
matrixPrint(m);
```



> 《狼、羊、甘藍菜》過河的問題

```javascript
var c = console;
var objs = ["人", "狼", "羊", "菜"];
var state= [   0,  0 ,   0,    0 ];

function neighbors(s) {
    var side = s[0];
    var next = [];
    checkAdd(next, move(s,0));
    for (var i=1; i<s.length; i++) {
        if (s[i]===side)
          checkAdd(next, move(s, i));
    }
    return next;
}

function checkAdd(next, s) {
    if (!isDead(s)) {
        next.push(s);
    }
}

function isDead(s) {
    if (s[1]===s[2] && s[1]!==s[0]) return true; // 狼吃羊
    if (s[2]===s[3] && s[2]!==s[0]) return true; // 羊吃菜
    return false;
}

// 人帶著 obj 移到另一邊
function move(s, obj) {
    var newS = s.slice(0); // 複製一份陣列
    var side = s[0];
    var anotherSide = (side===0)?1:0;
    newS[0] = anotherSide;
    newS[obj] = anotherSide;
    return newS; 
}

var visitedMap = {};

function visited(s) {
    var str = s.join('');
    return (typeof visitedMap[str] !== 'undefined');
}

function isSuccess(s) {
    for (var i=0; i<s.length; i++) {
      if (s[i]===0) return false;        
    }
    return true;
}

function state2str(s) {
    var str = "";
    for (var i=0; i<s.length; i++) {
        str += objs[i]+s[i]+" ";
    }
    return str;
}

var path = [];

function printPath(path) {
    for (var i=0; i<path.length; i++)
      c.log(state2str(path[i]));
}

function dfs(s) {
  if (visited(s)) return;
  path.push(s);
//  c.log('visit:', state2str(s));
  if (isSuccess(s)) {
      c.log("success!");
      printPath(path);
      return;
  }
  visitedMap[s.join('')] = true;
  var neighborsList = neighbors(s); 
  for (var i in neighborsList) { 
    dfs(neighborsList[i]);
  }
  path.pop();
}

dfs(state);
```



> 八個皇后問題

```python
def queens(n, i, a, b, c):
    if i < n:
        for j in range(n):
            if j not in a and i+j not in b and i-j not in c:
                yield from queens(n, i+1, a+[j], b+[i+j], c+[i-j])
    else:
        yield a

for solution in queens(8, 0, [], [], []):
    print(solution)
```

