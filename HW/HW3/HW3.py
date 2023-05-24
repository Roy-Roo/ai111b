ans = 0
col = [False] * 10  # 列的佔用情況，col[i]為True表示第i列已被佔用
x1 = [False] * 20  # 正對角線的佔用情況，x1[r+i]為True表示正對角線上的位置(r, i)已被佔用
x2 = [False] * 20  # 反對角線的佔用情況，x2[r-i+8]為True表示反對角線上的位置(r, i)已被佔用

def check(r, i):
    # 檢查位置(r, i)是否可以放置皇后
    return not col[i] and not x1[r+i] and not x2[r-i+8]

def dfs(r):
    global ans
    if r == 8:
        ans += 1  # 找到一組合法的解，答案計數器加1
        return
    for i in range(8):
        if check(r, i):
            col[i] = x1[r+i] = x2[r-i+8] = True  # 佔用位置(r, i)
            dfs(r+1)  # 遞迴搜索下一行
            col[i] = x1[r+i] = x2[r-i+8] = False  # 回溯，釋放位置(r, i)

dfs(0)  # 從第0行開始搜索
print(ans)
