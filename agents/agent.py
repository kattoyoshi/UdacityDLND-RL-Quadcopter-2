# TODO: your agent here!

import random
from collections import namedtuple, deque
import numpy as np
import copy
from agents.actor_critic import Actor, Critic

class DDPG_Agent():
    '''Reinforcement Learning agent that learns using Deep Deterministic Policy Gradients(DDPG).'''
    
    def __init__(self, 
                 task,
                 learning_rate_actor,
                 learning_rate_critic,
                 gamma=0.99,
                 tau = 0.01,
                 buffer_size = 100000,
                 batch_size = 64,
                 exploration_mu = 0,
                 exploration_theta = 0.15,
                 exploration_sigma = 0.2):
        '''
        # Arguments
            task: instance of the Task class
            learning_rate_actor: learning_rate of the actor network
            learning_rate_critic: learning_rate of the critc network
            gamma: discount rate
            tau: soft update coef.
            buffer_size: int
                size of replay buffer
            batch_size: int
                size of batch extract from replay buffer
        '''
        ### Task (environment) information ###
        self.task = task
        
        # state space
        self.state_size = task.state_size

        # action space
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        ### Actor ###
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, learning_rate_actor)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, learning_rate_actor)
        # Initialize target model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        ### Critic ###
        self.critic_local = Critic(self.state_size, self.action_size, learning_rate_critic)
        self.critic_target = Critic(self.state_size, self.action_size, learning_rate_critic)
        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        
        ### Noise process for exploration ###
        self.exploration_mu = exploration_mu
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        ### Replay memory (Experience Replay) ###
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        ### Algorithm parameters ###
        self.gamma = gamma
        self.tau = tau
        
        
    def reset_episode(self):
        '''
        Reset environments. 
        Run this function before starting epsode.
        '''
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        '''
        One-step experience of the agent.
        Store a (S_t, A_t, R_t+1, S_t+1) experience tuple to replay-buffer.
        Update policy function and value function using batch of experience tuples.
        
        # Arguments
            action: A_t
            reward: R_t+1
            next_state: S_t+1
            done: boolean flag
                The indicator whether end of the episode or not.
        '''
        
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        '''Returns actions for given state(s) as per current policy.'''
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        """
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
        buffer_size: maximum size of buffer
        batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
