import numpy as np

class GridWorldEnv():
    def __init__(self, width = 5, height = 5):
        self.width = width
        self.height = height
        self.state = (0, 0)
        self.walls = {(1, 1), (2, 3)}
        self.lakes = {(3, 1)}
        self.exit = (4, 4)
        self.exstates =  self.walls | self.lakes | {self.exit}

        self.action_ids = (0, 1, 2, 3)
        self.action = {0: (1, 0),  # up
                       1: (-1, 0), # down
                       2: (0, -1), # left
                       3: (0, 1)}  # right

        self.stoc_policy = (0, -1, 1) # correct, slide left, slide right
        self.stoc_probs = (0.8, 0.1, 0.1)

        self.gamma = 0.95

    def reset(self):
        self.state = (0, 0)
        return self.state 

    def get_all_states(self):
            #all_nstates = [(x, y) not in self.exstates for x in range(self.width) for y in range(self.height)] # all states without terminal states and walls
            all_states = [(x, y)  for x in range(self.width) for y in range(self.height) if (x, y) not in self.walls]
            return all_states
    
    """
    def step(self, a):
        reward = 0
        reward -= 0.04
        done = 0
        dx, dy = a 
        stoc = np.random.choice(self.stoc_policy, p = self.stoc_probs)
        x, y = self.state
        nx, ny = (x + dx, y + dy + stoc)
        next_state = (nx, ny)
        if nx < 0 or nx >= self.width  or ny < 0 or ny >= self.height or next_state in self.walls:
            next_state = self.state 
        if next_state in self.lakes:
            reward -= 1
            done = 1
        elif next_state == self.exit:
            reward += 1
            done = 1

        self.state = next_state
        return next_state, reward, done
    """
