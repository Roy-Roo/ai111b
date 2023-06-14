# Week16	

強化學習與 gym

### 強化學習

* 通過獎勵信號來指導學習過程，通過試錯的方式，根據當前的狀態和行動，獲得環境給予的回報，並逐漸調整策略以獲得更高的回報。
* 在強化學習中，並不需要事先標註的訓練資料，而是通過與環境的互動來進行訓練和學習。



### Gym

* 提供了一個統一的介面和一系列的環境，供開發者和研究者在強化學習算法的設計和評估中使用
* 開發者可以通過統一的API來設置環境、執行行動、獲得觀測和獎勵等，並且可以根據需要自定義環境和算法



### Q-Learning

* 是一種強化學習算法，用於解決Markov Decision Process（MDP）中的問題。它通過學習一個Q值函數來指導智能體的行動選擇，以最大化預期的長期回報。
* 是一種模型無關的強化學習算法，不需要事先了解或建模環境的動態。它通過與環境的互動來學習最佳策略，適用於具有離散狀態和行動空間的問題，如迷宮遊戲、棋盤遊戲等

> qLearn1.py

```python
import gym
import numpy as np

# 定义环境
env = gym.make("MountainCar-v0")

# 定义 Q 表
q_table = np.zeros((20, 20, 3))

# 定义超参数
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# 定义训练循环
for episode in range(10000):
    # 初始化环境
    state = env.reset()
    done = False
    total_reward = 0

    # 开始 episode
    while not done:
        # 选择动作
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state[0]][state[1]])
        
        # 执行动作，获得下一个状态、奖励和是否结束的信息
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # 更新 Q 表
        q_table[state[0]][state[1]][action] = (1 - alpha) * q_table[state[0]][state[1]][action] + alpha * (reward + gamma * np.max(q_table[next_state[0]][next_state[1]]))

        # 更新状态
        state = next_state
    
    # 打印结果
    print("Episode {}: Total reward = {}".format(episode, total_reward))
```

