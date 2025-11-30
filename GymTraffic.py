import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GymTrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # --- System Parameters ---
        self.max_queue = 18
        self.arrival_prob_road1 = 0.28
        self.arrival_prob_road2 = 0.40
        self.max_episode_steps = 1800
        self.green_departure_prob = 0.9

        # State variables (will be initialized in reset)
        self.q1 = None
        self.q2 = None
        self.g = None          # green road: 1 or 2
        self.delta = None      # time since last switch
        self.current_step = None

        # --- Gym-specific definitions ---
        # Observation space: (q1, q2, g, delta)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.max_queue + 1),  # q1
            spaces.Discrete(self.max_queue + 1),  # q2
            spaces.Discrete(2),                  # g ∈ {1, 2} → mapped as {0, 1}
            spaces.Discrete(11)                  # delta ∈ [0, 10]
        ))

        # Action space: {0: keep green, 1: switch}
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q1 = np.random.randint(0, 11)  # uniform from 0 to 10
        self.q2 = np.random.randint(0, 11)
        self.g = np.random.choice([1, 2])  # randomly choose initial green
        self.delta = 10  # allow switching immediately at start
        self.current_step = 0

        return (self.q1, self.q2, self.g - 1, self.delta), {}  # g ∈ {0, 1}

    def step(self, action):
        # Step counter
        self.current_step += 1

        # --- Action decoding ---
        # Action 1 (switch) is only allowed when delta == 10
        if action == 1 and self.delta == 10:
            self.g = 1 if self.g == 2 else 2  # toggle road
            self.delta = 0
        else:
            action = 0  # force keep if invalid
            self.delta = min(self.delta + 1, 10)

        # --- Determine departure probability ---
        if self.delta == 0:
            dep_prob = self.green_departure_prob
        else:
            dep_prob = self.green_departure_prob * (1 - (self.delta ** 2) / 100)

        # --- Sample arrivals ---
        a1 = np.random.rand() < self.arrival_prob_road1
        a2 = np.random.rand() < self.arrival_prob_road2

        # --- Sample departures ---
        departure = np.random.rand() < dep_prob

        # --- Update queues ---
        if self.g == 1:
            self.q1 = min(self.max_queue, max(0, self.q1 - departure + a1))
            self.q2 = min(self.max_queue, self.q2 + a2)
        else:
            self.q2 = min(self.max_queue, max(0, self.q2 - departure + a2))
            self.q1 = min(self.max_queue, self.q1 + a1)

        # --- Form state ---
        state = (self.q1, self.q2, self.g - 1, self.delta)

        # --- Reward ---
        reward = - (self.q1 + self.q2)

        # --- Episode ends? ---
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {}

        return state, reward, terminated, truncated, info

