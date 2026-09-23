import numpy as np
from collections import defaultdict

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

    def _perpendiculars(self, dx, dy):
        left = (-dy, dx)
        right = (dy, -dx)
        return left, right
        
    def transition_model(self, state, a):
            outcomes = defaultdict(float) # key: (next_state, reward), values: cumulative prob
            dx, dy = a
            left, right = self._perpendiculars(dx, dy)
            directions = [(dx, dy), left, right]
    
            for (mdx, mdy), p in zip(directions, self.stoc_probs):
                x, y = state
                nx, ny = (x + mdx, y + mdy)
                ns = (nx, ny) # next_state
                reward = 0
                reward -= 0.04
                done = 0
    
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or ns in self.walls:
                    ns = state
                if ns in self.lakes:
                    done = 1
                    reward -= 1
                elif ns == self.exit:
                    done = 1
                    reward += 1
                
                outcomes[(ns, reward)] += p
    
            return outcomes

