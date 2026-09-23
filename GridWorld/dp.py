from gridworld import GridWorldEnv

import numpy as np 
from collections import defaultdict

class DPAgent():
    def __init__(self, env):
        self.env = env
        self.gamma = self.env.gamma 

        self.states = self.env.get_all_states()
        self.action = self.env.action

        self.V = {s: 0.0 for s in self.states}

        self.policy = self._init_policy() # key: state, value: (idx: prob)

    def _init_policy(self):
        return {
            s: {0: 0.25, 1: 0.25, 2: 0.25, 3:0.25}
            for s in self.states
        }

    def policy_evaluation(self, policy, theta = 1e-6):
        while True:
            delta = 0

            for state, action in policy.items():
                if state in self.env.lakes or state == self.env.exit:
                    continue

                v_old = self.V[state]
                new_v = 0.0

                for a_id, a_prob in action.items():
                    if a_prob == 0:
                        continue

                    outcomes = self.env.transition_model(state, self.env.action[a_id])
                    new_v += sum(a_prob * ns_p * (r + self.gamma * self.V[ns])
                                for (ns, r), ns_p in outcomes.items()
                    )

                self.V[state] = new_v
                delta = max(delta, abs(v_old - self.V[state]))

            if delta < theta:
                break

    def policy_improvement(self):
        policy_stable = True 
        for s in self.states:
            if s in self.env.lakes or s == self.env.exit:
                continue 

            old_best_a = max(self.policy[s], key = self.policy[s].get)
            q_values = defaultdict(float) # dict: (key: a, value: q_val)

            for a_id, a_vec in self.env.action.items():
                outcomes = self.env.transition_model(s, a_vec)
                q_sa = sum(ns_p * (r + self.gamma * self.V[ns])
                           for (ns, r), ns_p in outcomes.items()
                )
                q_values[a_id] += q_sa

            best_a = max(q_values, key = q_values.get)
            if (best_a != old_best_a):
                policy_stable = False

            for a_id in self.env.action:
                self.policy[s][a_id] = 1.0 if a_id == best_a else 0.0

        return policy_stable      

    def policy_iteration(self, theta = 1e-6):
        n_iter = 0
        while True:
            self.policy_evaluation(self.policy, theta)
            stable = self.policy_improvement()
            n_iter += 1
            if stable:
                break
        return n_iter
    
    def value_iteration(self, theta = 1e-6):
            n_iter = 0
            while True:
                delta = 0
                for state in self.states:
                    if state == self.env.exit or state in self.env.lakes:
                        continue 

                    v_old = self.V[state]
                    q_values = []
                    for a_id, a_vec in self.action.items():
                        outcomes = self.env.transition_model(state, a_vec)
                        q_sa = sum(ns_p * (r + self.gamma * self.V[ns])
                                   for (ns, r), ns_p in outcomes.items())
                        q_values.append(q_sa)

                    self.V[state] = max(q_values)
                    delta = max(delta, abs(v_old - self.V[state]))

                n_iter += 1
                if delta < theta:
                    break         

            return n_iter
    
    def extract_policy(self):
        policy_dict = {}
        for s in self.states:
            if s in self.env.lakes or s == self.env.exit:
                continue

            best_a, best_q = None, -float('inf')
            for a_id, a_vec in self.env.action.items():
                outcomes = self.env.transition_model(s, a_vec)
                q_sa = sum(ns_p * (r + self.gamma * self.V[ns])
                           for (ns, r), ns_p in outcomes.items())
                if q_sa > best_q:
                    best_q, best_a = q_sa, a_id

            policy_dict[s] = best_a
        return policy_dict
