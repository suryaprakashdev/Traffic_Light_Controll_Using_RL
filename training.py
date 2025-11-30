import numpy as np
from GymTraffic import GymTrafficEnv

# Exploration helper
def epsilon_greedy(q_values, epsilon, nu):
    if np.random.rand() < (1 - epsilon):
        return np.argmax(q_values)
    else:
        norm_q = q_values / (nu + np.sum(np.abs(q_values)))
        exp_q = np.exp(norm_q)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice([0, 1], p=probs)

# ---------------- SARSA ----------------
def SARSA(env, beta, Nepisodes, alpha):
    max_q = 20
    nu = 1e-8
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    Q = np.zeros((max_q + 1, max_q + 1, 2, 11, 2))

    for ep in range(Nepisodes):
        state, _ = env.reset()
        q1, q2, g, delta = [min(state[0], max_q), min(state[1], max_q), state[2], state[3]]
        action = epsilon_greedy(Q[q1, q2, g, delta], epsilon, nu)

        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated
            nq1, nq2, ng, ndelta = [min(next_state[0], max_q), min(next_state[1], max_q), next_state[2], next_state[3]]
            next_action = epsilon_greedy(Q[nq1, nq2, ng, ndelta], epsilon, nu)

            Q[q1, q2, g, delta, action] += alpha * (
                reward + beta * Q[nq1, nq2, ng, ndelta, next_action] - Q[q1, q2, g, delta, action]
            )

            q1, q2, g, delta = nq1, nq2, ng, ndelta
            action = next_action

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    policy = np.argmax(Q, axis=-1)
    return policy

# ---------------- Expected SARSA ----------------
def ExpectedSARSA(env, beta, Nepisodes, alpha):
    max_q = 20
    nu = 1e-8
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    Q = np.zeros((max_q + 1, max_q + 1, 2, 11, 2))

    for ep in range(Nepisodes):
        state, _ = env.reset()
        q1, q2, g, delta = [min(state[0], max_q), min(state[1], max_q), state[2], state[3]]

        done = False
        while not done:
            action = epsilon_greedy(Q[q1, q2, g, delta], epsilon, nu)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated
            nq1, nq2, ng, ndelta = [min(next_state[0], max_q), min(next_state[1], max_q), next_state[2], next_state[3]]

            # Compute expected Q using softmax over normalized Q-values
            norm_q = Q[nq1, nq2, ng, ndelta] / (nu + np.sum(np.abs(Q[nq1, nq2, ng, ndelta])))
            exp_q = np.exp(norm_q)
            probs = exp_q / np.sum(exp_q)
            expected_q = np.dot(probs, Q[nq1, nq2, ng, ndelta])

            Q[q1, q2, g, delta, action] += alpha * (
                reward + beta * expected_q - Q[q1, q2, g, delta, action]
            )

            q1, q2, g, delta = nq1, nq2, ng, ndelta

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    policy = np.argmax(Q, axis=-1)
    return policy

# ---------------- Value-based SARSA ----------------
def ValueFunctionSARSA(env, beta, Nepisodes, alpha):
    max_q = 20
    nu = 1e-8
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    V = np.zeros((max_q + 1, max_q + 1, 2, 11))

    for ep in range(Nepisodes):
        state, _ = env.reset()
        q1, q2, g, delta = [min(state[0], max_q), min(state[1], max_q), state[2], state[3]]

        done = False
        while not done:
            # Compute q(x,a) from V(x′)
            q_vals = np.zeros(2)
            for a in [0, 1]:
                env_state = (q1, q2, g, delta)
                env2 = GymTrafficEnv()
                env2.q1, env2.q2, env2.g, env2.delta = q1, q2, g + 1, delta
                env2.current_step = 0
                next_state, reward, _, _, _ = env2.step(a)
                nq1, nq2, ng, ndelta = [min(next_state[0], max_q), min(next_state[1], max_q), next_state[2], next_state[3]]
                q_vals[a] = reward + beta * V[nq1, nq2, ng, ndelta]

            action = epsilon_greedy(q_vals, epsilon, nu)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated
            nq1, nq2, ng, ndelta = [min(next_state[0], max_q), min(next_state[1], max_q), next_state[2], next_state[3]]

            V[q1, q2, g, delta] += alpha * (reward + beta * V[nq1, nq2, ng, ndelta] - V[q1, q2, g, delta])
            q1, q2, g, delta = nq1, nq2, ng, ndelta

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Deriving greedy policy from q(x,a)
    policy = np.zeros((max_q + 1, max_q + 1, 2, 11), dtype=int)
    for q1 in range(21):
        for q2 in range(21):
            for g in range(2):
                for delta in range(11):
                    q_vals = []
                    for a in [0, 1]:
                        env2 = GymTrafficEnv()
                        env2.q1, env2.q2, env2.g, env2.delta = q1, q2, g + 1, delta
                        env2.current_step = 0
                        next_state, reward, _, _, _ = env2.step(a)
                        nq1, nq2, ng, ndelta = [min(next_state[0], max_q), min(next_state[1], max_q), next_state[2], next_state[3]]
                        q_vals.append(reward + beta * V[nq1, nq2, ng, ndelta])
                    policy[q1, q2, g, delta] = np.argmax(q_vals)
    return policy
#----------------------------------------
if __name__ == "__main__":
    env = GymTrafficEnv()

    Nepisodes = 2000   # Number of training episodes
    alpha = 0.1        # Learning rate
    beta = 0.997       # Discount factor

    print("Training SARSA...")
    policy1 = SARSA(env, beta, Nepisodes, alpha)
    np.save('policy1.npy', policy1)
    print("Saved policy1.npy")

    print("Training Expected SARSA...")
    policy2 = ExpectedSARSA(env, beta, Nepisodes, alpha)
    np.save('policy2.npy', policy2)
    print("Saved policy2.npy")

    print("Training ValueFunction SARSA...")
    policy3 = ValueFunctionSARSA(env, beta, Nepisodes, alpha)
    np.save('policy3.npy', policy3)
    print("Saved policy3.npy")

    env.close()

