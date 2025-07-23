"""
Scenario 1: Simple Package Collection
Implements a Q-learning agent to collect a single package in the Four-Rooms environment.
Supports both epsilon-greedy and Boltzmann exploration strategies.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import argparse
import os

class QLearningAgent:
    """Q-learning agent with configurable exploration strategy"""
    
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_strategy='epsilon_greedy'):
        """
        Initialize Q-learning agent
        
        Args:
            num_actions: Number of possible actions (4 for grid world)
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            exploration_strategy: 'epsilon_greedy' or 'boltzmann'
        """
        self.q_table = {}  # State-action value dictionary
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.exploration_strategy = exploration_strategy
        
        # Exploration parameters
        self.epsilon = 1.0  # Initial exploration rate (epsilon-greedy)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate per episode
        
        # Boltzmann exploration parameters
        self.temperature = 1.0  # Initial temperature
        self.temp_decay = 0.995  # Temperature decay rate

    def get_state_key(self, state):
        """
        Convert state to immutable tuple for Q-table key
        
        Args:
            state: Tuple containing (x_pos, y_pos, packages_remaining)
        Returns:
            Immutable state representation
        """
        return (state[0], state[1], state[2])

    def choose_action(self, state):
        """
        Select action using specified exploration strategy
        
        Args:
            state: Current environment state
        Returns:
            action: Selected action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        
        if self.exploration_strategy == 'epsilon_greedy':
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)  # Explore
            return np.argmax(self.q_table[state_key])  # Exploit
        else:
            # Boltzmann exploration
            q_values = self.q_table[state_key]
            exp_q = np.exp(q_values / self.temperature)
            probabilities = exp_q / np.sum(exp_q)
            return np.random.choice(range(self.num_actions), p=probabilities)

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-values using temporal difference learning
        
        Args:
            state: Current state before action
            action: Action taken
            reward: Immediate reward received
            next_state: Resulting state after action
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize next state if not seen before
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)
        
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error
        
        # Decay exploration parameters
        if self.exploration_strategy == 'epsilon_greedy':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.temperature = max(0.1, self.temperature * self.temp_decay)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Scenario 1: Simple Package Collection')
    parser.add_argument('-stochastic', action='store_true', 
                       help='Enable 20% action failure probability')
    parser.add_argument('-boltzmann', action='store_true',
                       help='Use Boltzmann exploration instead of epsilon-greedy')
    parser.add_argument('-episodes', type=int, default=500,
                       help='Number of training episodes')
    return parser.parse_args()

def main():
    """Main training loop for Scenario 1"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory for saving results
    os.makedirs('output', exist_ok=True)
    
    # Initialize FourRooms environment ('simple' scenario)
    fourRoomsObj = FourRooms('simple', args.stochastic)
    
    # Create Q-learning agent with specified exploration strategy
    exploration = 'boltzmann' if args.boltzmann else 'epsilon_greedy'
    agent = QLearningAgent(4, exploration_strategy=exploration)  # 4 possible actions
    
    rewards = []  # Track rewards per episode
    
    # Training loop
    for episode in range(args.episodes):
        # Reset environment for new episode
        fourRoomsObj.newEpoch()
        state = (*fourRoomsObj.getPosition(), fourRoomsObj.getPackagesRemaining())
        total_reward = 0
        
        # Run episode until package is collected
        while not fourRoomsObj.isTerminal():
            # Agent selects action
            action = agent.choose_action(state)
            
            # Execute action in environment
            cell_type, new_pos, packages_left, is_terminal = fourRoomsObj.takeAction(action)
            
            # Calculate reward
            reward = -0.1  # Small penalty per step to encourage efficiency
            if cell_type > 0:  # Package collected
                reward = 100  # Large positive reward
            
            # Update agent's knowledge
            next_state = (*new_pos, packages_left)
            agent.update_q_table(state, action, reward, next_state)
            
            # Transition to next state
            state = next_state
            total_reward += reward
        
        # Record episode reward
        rewards.append(total_reward)
    
    # Save visualization of final path
    stochastic_suffix = '_stochastic' if args.stochastic else ''
    exploration_suffix = '_boltzmann' if args.boltzmann else '_epsilon'
    fourRoomsObj.showPath(-1, savefig=f'output/scenario1_path{exploration_suffix}{stochastic_suffix}.png')
    
    # Plot and save learning curve
    plt.figure()
    plt.plot(rewards)
    plt.title(f'Scenario 1 Rewards ({exploration}{stochastic_suffix})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(f'output/scenario1_rewards{exploration_suffix}{stochastic_suffix}.png')
    plt.close()

if __name__ == "__main__":
    main()