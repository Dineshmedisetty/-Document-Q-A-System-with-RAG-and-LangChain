import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

class Environment:
    def __init__(self, probs):
        self.probs = probs  # success probabilities for each arm
    def step(self, action):
        # Pull arm and get stochastic reward (1 for success, 0 for failure)
        return 1 if (np.random.random() < self.probs[action]) else 0

class Agent:
    def __init__(self, nActions, max_steps):
        self.nActions = nActions
        self.max_steps = max_steps
        self.step_count = 0  # Track how many steps have occurred
        self.n = np.zeros(nActions, dtype=np.int64)  # action counts n(a)
        self.Q = np.zeros(nActions, dtype=np.float64)  # value Q(a)

    def update_Q(self, action, reward):
        self.n[action] += 1
        self.Q[action] += (1.0 / self.n[action]) * (reward - self.Q[action])

    def get_action(self):
        # Linearly decay epsilon from 1.0 to 0.1
        epsilon = max(1.0 - (0.9 * self.step_count / self.max_steps), 0.1)
        self.step_count += 1

        if np.random.random() < epsilon:
            return np.random.randint(self.nActions)  # explore
        else:
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))  # exploit

# Start multi-armed bandit simulation
def experiment(probs, N_episodes):
    env = Environment(probs)
    agent = Agent(len(env.probs), N_episodes)  # pass max steps
    actions, rewards = [], []
    for episode in range(N_episodes):
        action = agent.get_action()  # sample policy
        reward = env.step(action)  # take step + get reward
        agent.update_Q(action, reward)  # update Q
        actions.append(action)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)

# Settings
probs = [0.10, 0.50, 0.60, 0.80, 0.10,
         0.25, 0.60, 0.45, 0.75, 0.65]  # bandit arm probabilities
N_experiments = 10000
N_steps = 100
save_fig = True
output_dir = os.path.join(os.getcwd(), "output")

# Run multi-armed bandit experiments
print("Running multi-armed bandits with nActions = {}, epsilon decay from 1.0 to 0.1".format(len(probs)))
R = np.zeros((N_steps,))
A = np.zeros((N_steps, len(probs)))
for i in range(N_experiments):
    actions, rewards = experiment(probs, N_steps)
    if (i + 1) % (N_experiments // 100) == 0:
        print("[Experiment {}/{}] ".format(i + 1, N_experiments) +
              "n_steps = {}, ".format(N_steps) +
              "reward_avg = {:.4f}".format(np.sum(rewards) / len(rewards)))
    R += rewards
    for j, a in enumerate(actions):
        A[j][a] += 1

# Plot reward results
R_avg = R / float(N_experiments)
plt.plot(R_avg, ".")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.grid()
plt.xlim([1, N_steps])
if save_fig:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "rewards.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()

# Plot action results
for i in range(len(probs)):
    A_pct = 100 * A[:, i] / N_experiments
    steps = list(np.array(range(len(A_pct))) + 1)
    plt.plot(steps, A_pct, "-",
             linewidth=4,
             label="Arm {} ({:.0f}%)".format(i + 1, 100 * probs[i]))
plt.xlabel("Step")
plt.ylabel("Count Percentage (%)")
leg = plt.legend(loc='upper left', shadow=True)
plt.xlim([1, N_steps])
plt.ylim([0, 100])
for legobj in leg.get_lines():
    legobj.set_linewidth(4.0)
if save_fig:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "actions.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()
