import jax
import jax.numpy as jnp
from flax import nnx

import optax

import gymnasium as gym

import numpy as np

from typing import NamedTuple, Dict
from collections import deque
import random

import matplotlib.pyplot as plt

class QNetwork(nnx.Module):
    """
    Neural network for predicting Q-values for each action given the state
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        self.fc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        x = self.fc1(state)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        q_values = self.fc3(x)
        return q_values

class StateTransition(NamedTuple):
    """
    Tuple to hold state transition information (s, a, r, s', done)
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    """
    Stores state-transitions.
    """
    def __init__(self, max_len: int = 100000):
        self.buffer = deque(maxlen=max_len)

    def push(self, state_transition: StateTransition):
        self.buffer.append(state_transition)

    def sample(self, batch_size: int) -> Dict[str, jax.Array]:
        sampled_state_transitions = random.sample(self.buffer, batch_size)
        return {
            'states': jnp.array([t.state for t in sampled_state_transitions]),
            'actions': jnp.array([t.action for t in sampled_state_transitions]),
            'rewards': jnp.array([t.reward for t in sampled_state_transitions]),
            'next_states': jnp.array([t.next_state for t in sampled_state_transitions]),
            'dones': jnp.array([t.done for t in sampled_state_transitions], dtype=jnp.float32),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)

class DDQNAgent(nnx.Module):
    """
    Agent for Double Deep Q-Network
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        buffer_size: int = 100000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        gamma: float = 0.99,
        seed: int = 24
    ):
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gamma = gamma
        self.action_dim = action_dim

        # Networks
        network_rngs = nnx.Rngs(seed)
        self.q_network = QNetwork(network_rngs, state_dim, action_dim, hidden_dim)
        self.target_network = nnx.clone(self.q_network)

        # Optimizer
        self.optimizer = nnx.Optimizer(self.q_network, optax.adam(learning_rate), wrt=nnx.Param)
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, epsilon: float) -> int:
        """Select action using epsilon greedy policy"""
        # Explore
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        # Exploit
        q_values = self.q_network(jnp.asarray(state, jnp.float32))
        return int(jnp.argmax(q_values))
    
    def store_state_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(StateTransition(state, action, reward, next_state, done))

    def epsilon_at(self, step: int) -> float:
        """Linear anneal epsilon_start -> epsilon_end over epsilon_decay_steps."""
        frac = min(step / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)    

    def train_step(self, update_target_network: bool) -> float:
        """
        Go through train step in DDQN algorithm.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self._update_q_network(batch)
        if update_target_network:
            self._update_target_network()
        return float(loss)

    @nnx.jit
    def _update_q_network(self, batch: Dict[str, jax.Array]) -> jax.Array:
        """
        Compute loss and perform gradient descent on Q-network.
        """
        def loss_fn(q_network):
            # Expected values
            q_values_all = q_network(batch['states'])
            q_values_selected = q_values_all[jnp.arange(len(batch['actions'])), batch['actions']]
            # Target values
            target_q_values = self.target_network(batch['next_states'])
            targets = batch['rewards'] + self.gamma * jnp.max(target_q_values, axis=-1) * (1 - batch['dones'])
            # Errors
            td_errors = q_values_selected - targets
            loss = jnp.mean(td_errors**2)
            return loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(self.q_network)
        self.optimizer.update(self.q_network, grads)
        return loss

    def _update_target_network(self):
        q_params = nnx.state(self.q_network, nnx.Param)
        # target_params = nnx.state(self.target_network, nnx.Param)

        # new_target_params = jax.tree.map(lambda q, _: q, q_params, target_params)
        nnx.update(self.target_network, q_params)

def train(num_episodes: int = 300, max_steps_per_episode: int = 500, update_target_network_steps: int = 100, seed: int = 24):
    # Initialize env
    random.seed(seed)
    np.random.seed(seed)
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # Initialize agent
    agent = DDQNAgent(state_dim=state_dim, action_dim=action_dim, seed=seed)
    # Epsiode loop
    episode_rewards = []
    loss_steps = []        # global step index for each logged loss
    losses = []            # scalar TD loss, one per training step
    global_step = 0
    for episode in range(1, num_episodes+1):
        state, _ = env.reset(seed=seed if episode == 1 else None)
        episode_reward = 0
        # rollout
        for step in range(max_steps_per_episode):
            global_step+=1
            epsilon = agent.epsilon_at(global_step)
            # Sample
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_state_transition(state, action, reward, next_state, done)

            # Train
            update_target_network = (global_step % update_target_network_steps) == 0
            loss = agent.train_step(update_target_network)
            if loss is not None:
                loss_steps.append(global_step)
                losses.append(loss)
            # update reward and state
            episode_reward += reward
            state = next_state
            if done:
                break
        episode_rewards.append(episode_reward)
        if episode % 25 == 0:
            recent_loss = np.mean(losses[-100:]) if losses else float("nan")
            print(f"episode {episode:4d} | return(last 25) = {np.mean(episode_rewards[-25:]):6.1f}"
                  f" | eps = {epsilon:.3f} | loss(last 100) = {recent_loss:.4f}", flush=True)

    env.close()
    return agent, episode_rewards, loss_steps, losses

def moving_average(x, window: int) -> np.ndarray:
    """Trailing mean over `window`; length max(len(x) - window + 1, 0)."""
    x = np.asarray(x, dtype=float)
    w = min(window, len(x))
    if w == 0:
        return np.empty(0)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(episode_rewards, loss_steps, losses, save_path="training_curves.png",
                  reward_window: int = 20, loss_window: int = 200) -> plt.Figure:
    """Plot episode return and TD loss, with trailing moving averages."""
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- episode return ----
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax_r.plot(episodes, episode_rewards, color="tab:blue", alpha=0.3, lw=1, label="episode return")
    ma = moving_average(episode_rewards, reward_window)
    if ma.size:
        ax_r.plot(episodes[len(episodes) - ma.size:], ma, color="tab:blue", lw=2,
                  label=f"moving avg ({reward_window})")
    ax_r.set(xlabel="episode", ylabel="return", title="Episode return")
    ax_r.grid(alpha=0.3)
    ax_r.legend(loc="upper left")

    # ---- TD loss ----
    if losses:
        steps = np.asarray(loss_steps)
        vals = np.asarray(losses, dtype=float)
        stride = max(1, vals.size // 4000)      # thin the raw trace: 38k points is a blob
        ax_l.plot(steps[::stride], vals[::stride], color="tab:red", alpha=0.18, lw=0.5,
                  label="TD loss (per step)")
        ma = moving_average(vals, loss_window)
        if ma.size:
            ax_l.plot(steps[vals.size - ma.size:], ma, color="tab:red", lw=2,
                      label=f"moving avg ({loss_window})")
        ax_l.set_yscale("log")                 # MSE spans orders of magnitude
    ax_l.set(xlabel="training step", ylabel="mean squared TD error", title="TD loss (log scale)")
    ax_l.grid(alpha=0.3)
    ax_l.legend(loc="upper left")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

if __name__=="__main__":
    agent, episode_rewards, loss_steps, losses = train()
    plot_training(episode_rewards, loss_steps, losses, save_path="training_curves.png")