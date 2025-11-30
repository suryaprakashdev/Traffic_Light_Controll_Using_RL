import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from GymTraffic import GymTrafficEnv

def TestPolicy(env, policy):
    max_q = 20  # Truncate large queue values for policy indexing
    state, _ = env.reset()

    queue1_history = []
    queue2_history = []
    action_history = []

    done = False
    total_steps = 0
    total_queue = 0

    while not done:
        q1, q2, g, delta = state
        # Cap the queue lengths for indexing into the policy
        q1_index = min(q1, max_q)
        q2_index = min(q2, max_q)

        # Select action from policy
        action = policy[q1_index, q2_index, g, delta]
        action_history.append(action)

        # Take a step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        done = truncated  # Episode ends only if truncated after 1800 steps

        # Store queue lengths for plotting
        queue1_history.append(state[0])
        queue2_history.append(state[1])
        total_queue += state[0] + state[1]
        total_steps += 1

    # --- Plotting the queue lengths ---
    plt.figure(figsize=(12, 6))
    plt.plot(queue1_history, label='Queue Length Road 1 (East-West)', linewidth=2)
    plt.plot(queue2_history, label='Queue Length Road 2 (North-South)', linewidth=2)
    plt.xlabel('Time Slot (Seconds)', fontsize=12)
    plt.ylabel('Queue Length', fontsize=12)
    plt.title('Queue Lengths Over Time', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    avg_queue_length = total_queue / total_steps
    print(f"\nAverage Total Queue Length Over Episode: {avg_queue_length:.2f}")
    print("\nActions Taken by Agent (0 = continue, 1 = switch):")
    print(action_history)

# ---------------------------------------------------

# Choose which policy to test
env = GymTrafficEnv()

# policy = np.load('policy1.npy')  # Load SARSA policy
# policy = np.load('policy2.npy')  # Load Expected SARSA policy
policy = np.load('policy3.npy')  # Load Value Function SARSA policy

TestPolicy(env, policy)

env.close()
