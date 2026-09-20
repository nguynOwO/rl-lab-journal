import numpy as np

walls = [[1, 1], [2, 3]]
lakes = [[3, 1]]
exit = (4, 4)

state = (0, 0)

action_ids = (0, 1, 2, 3)
actions = {0: (1, 0),  # up
           1: (-1, 0), # down
           2: (0, -1), # left
           3: (0, 1)}  # right
probs = [0.25, 0.25, 0.25, 0.25]

stoc_policy = [0,  # good
               -1, # left
               1]  # right
stoc_policy_probs = [0.8, 0.1, 0.1]
gamma = 0.95   
reward = 0

def step(s, a):
    global reward
    dx, dy = a
    x, y = s
    nx, ny = (x + dx, y + dy)
    ns = (nx, ny)
    if ns in walls or ns in lakes: 
        reward -= 1
        return False
    if nx < 0 or nx > 4 or ny < 0 or ny > 4: 
        return False
    return True

slide = 0
for i in range(100000):
    reward -= 0.04
    action_idx = np.random.choice(action_ids, p = probs)
    stoc = np.random.choice(stoc_policy, p = stoc_policy_probs)
    if stoc == 0: 
        dx, dy = actions[action_idx]
        if(step(state, (dx, dy))):
            state = (state[0] + dx, state[1] + dy)
    elif (step(state, (0, stoc))):
        state = (state[0], state[1] + stoc)

    if stoc != 0: slide += 1

    if (state == 'exit'):
        print(f"Found the exit with the reward of {reward:.2f} at step {i}")
        break
    
print(slide/100000)