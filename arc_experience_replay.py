"""
Experience Replay System with Q-Learning
Implements continuous learning from past solving attempts
"""

import json
import numpy as np
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import pickle
import hashlib

class Experience:
    """Represents a single solving experience"""
    
    def __init__(self, puzzle_id: str, state: np.ndarray, action: str, 
                 reward: float, next_state: np.ndarray, done: bool):
        self.puzzle_id = puzzle_id
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'puzzle_id': self.puzzle_id,
            'state': self.state.tolist() if isinstance(self.state, np.ndarray) else self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state.tolist() if isinstance(self.next_state, np.ndarray) else self.next_state,
            'done': self.done,
            'timestamp': self.timestamp.isoformat()
        }

class ExperienceReplayBuffer:
    """Circular buffer for storing experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.total_experiences = 0
        
    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.total_experiences += 1
        
    def sample(self, batch_size: int, prioritized: bool = True) -> List[Experience]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        if prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size)
            
        return [self.buffer[i] for i in indices]
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for experiences"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                
    def save(self, filepath: str):
        """Save buffer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'priorities': list(self.priorities),
                'total_experiences': self.total_experiences
            }, f)
            
    def load(self, filepath: str):
        """Load buffer from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data['buffer'], maxlen=self.buffer.maxlen)
            self.priorities = deque(data['priorities'], maxlen=self.priorities.maxlen)
            self.total_experiences = data['total_experiences']

class QLearningAgent:
    """Q-Learning agent for puzzle solving"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.01,
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = {}
        
        # Action mapping
        self.actions = [
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_horizontal', 'flip_vertical',
            'color_swap', 'extract_objects', 'fill_pattern',
            'apply_symmetry', 'crop', 'expand', 'tile'
        ]
        
    def get_state_hash(self, state: np.ndarray) -> str:
        """Get hash of state for Q-table lookup"""
        if isinstance(state, np.ndarray):
            state_bytes = state.tobytes()
        else:
            state_bytes = str(state).encode()
        return hashlib.md5(state_bytes).hexdigest()
        
    def get_q_value(self, state: np.ndarray, action: str) -> float:
        """Get Q-value for state-action pair"""
        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {a: 0.0 for a in self.actions}
        return self.q_table[state_hash].get(action, 0.0)
        
    def set_q_value(self, state: np.ndarray, action: str, value: float):
        """Set Q-value for state-action pair"""
        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {a: 0.0 for a in self.actions}
        self.q_table[state_hash][action] = value
        
    def choose_action(self, state: np.ndarray, explore: bool = True) -> str:
        """Choose action using epsilon-greedy policy"""
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.actions)
        else:
            # Exploit: best action
            state_hash = self.get_state_hash(state)
            if state_hash in self.q_table:
                q_values = self.q_table[state_hash]
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if q == max_q]
                return np.random.choice(best_actions)
            else:
                return np.random.choice(self.actions)
                
    def update(self, state: np.ndarray, action: str, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update Q-value using Q-learning update rule"""
        current_q = self.get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            # Get max Q-value for next state
            next_state_hash = self.get_state_hash(next_state)
            if next_state_hash in self.q_table:
                max_next_q = max(self.q_table[next_state_hash].values())
            else:
                max_next_q = 0.0
            target_q = reward + self.discount_factor * max_next_q
            
        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(state, action, new_q)
        
    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay exploration rate"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)

class ReinforcementLearningSolver:
    """Solver using reinforcement learning with experience replay"""
    
    def __init__(self, buffer_capacity: int = 10000):
        self.replay_buffer = ExperienceReplayBuffer(buffer_capacity)
        self.q_agent = QLearningAgent(state_size=100, action_size=12)
        self.episode_rewards = []
        self.success_count = 0
        self.total_episodes = 0
        
    def state_from_puzzle(self, puzzle: Dict) -> np.ndarray:
        """Extract state representation from puzzle"""
        if 'train' in puzzle and len(puzzle['train']) > 0:
            # Use first training example as state
            input_grid = np.array(puzzle['train'][0]['input'])
            
            # Create fixed-size state representation
            features = []
            
            # Grid statistics
            features.append(input_grid.shape[0])
            features.append(input_grid.shape[1])
            features.append(len(np.unique(input_grid)))
            features.append(np.mean(input_grid))
            features.append(np.std(input_grid))
            
            # Color histogram (up to 10 colors)
            hist = np.bincount(input_grid.flatten(), minlength=10)[:10]
            features.extend(hist.tolist())
            
            # Spatial features
            features.append(np.sum(input_grid[0, :]))  # Top row sum
            features.append(np.sum(input_grid[-1, :]))  # Bottom row sum
            features.append(np.sum(input_grid[:, 0]))  # Left column sum
            features.append(np.sum(input_grid[:, -1]))  # Right column sum
            
            # Pad to fixed size
            while len(features) < 100:
                features.append(0)
                
            return np.array(features[:100])
        else:
            return np.zeros(100)
            
    def calculate_reward(self, puzzle: Dict, solution: np.ndarray, 
                        solved: bool) -> float:
        """Calculate reward for a solution attempt"""
        if solved:
            return 10.0
            
        # Partial rewards based on similarity
        if 'train' in puzzle and len(puzzle['train']) > 0:
            if 'output' in puzzle['train'][0]:
                expected = np.array(puzzle['train'][0]['output'])
                
                # Shape similarity
                shape_reward = 0
                if solution.shape == expected.shape:
                    shape_reward = 2.0
                    
                # Color similarity
                color_reward = 0
                expected_colors = set(expected.flatten())
                solution_colors = set(solution.flatten())
                color_overlap = len(expected_colors & solution_colors) / max(len(expected_colors), 1)
                color_reward = color_overlap * 2.0
                
                # Pixel similarity (if same shape)
                pixel_reward = 0
                if solution.shape == expected.shape:
                    pixel_accuracy = np.mean(solution == expected)
                    pixel_reward = pixel_accuracy * 5.0
                    
                return shape_reward + color_reward + pixel_reward - 1.0
                
        return -1.0  # Penalty for failure
        
    async def train_on_puzzle(self, puzzle: Dict) -> Dict:
        """Train the agent on a single puzzle"""
        state = self.state_from_puzzle(puzzle)
        total_reward = 0
        steps = 0
        max_steps = 20
        
        while steps < max_steps:
            # Choose action
            action = self.q_agent.choose_action(state)
            
            # Apply action (simulated)
            solution = self._apply_action(puzzle, action)
            
            # Check if solved
            solved = self._check_solution(puzzle, solution)
            
            # Calculate reward
            reward = self.calculate_reward(puzzle, solution, solved)
            total_reward += reward
            
            # Get next state
            next_state = self._get_next_state(state, action)
            
            # Store experience
            experience = Experience(
                puzzle.get('id', 'unknown'),
                state, action, reward, next_state, solved
            )
            self.replay_buffer.add(experience, abs(reward))
            
            # Update Q-values
            self.q_agent.update(state, action, reward, next_state, solved)
            
            # Experience replay
            if len(self.replay_buffer.buffer) >= 32:
                batch = self.replay_buffer.sample(32)
                for exp in batch:
                    self.q_agent.update(
                        exp.state, exp.action, exp.reward,
                        exp.next_state, exp.done
                    )
                    
            if solved:
                self.success_count += 1
                break
                
            state = next_state
            steps += 1
            
        # Decay exploration
        self.q_agent.decay_epsilon()
        
        # Track episode
        self.episode_rewards.append(total_reward)
        self.total_episodes += 1
        
        return {
            'solved': solved,
            'steps': steps,
            'total_reward': total_reward,
            'epsilon': self.q_agent.epsilon,
            'q_table_size': len(self.q_agent.q_table)
        }
        
    def _apply_action(self, puzzle: Dict, action: str) -> np.ndarray:
        """Apply action to puzzle (simulated)"""
        if 'train' in puzzle and len(puzzle['train']) > 0:
            grid = np.array(puzzle['train'][0]['input'])
            
            if action == 'rotate_90':
                return np.rot90(grid, -1)
            elif action == 'rotate_180':
                return np.rot90(grid, 2)
            elif action == 'rotate_270':
                return np.rot90(grid, 1)
            elif action == 'flip_horizontal':
                return np.fliplr(grid)
            elif action == 'flip_vertical':
                return np.flipud(grid)
            elif action == 'color_swap':
                result = grid.copy()
                if len(np.unique(grid)) >= 2:
                    colors = np.unique(grid)[:2]
                    result[grid == colors[0]] = colors[1]
                    result[grid == colors[1]] = colors[0]
                return result
            else:
                # Default: return input
                return grid
        return np.array([[0]])
        
    def _check_solution(self, puzzle: Dict, solution: np.ndarray) -> bool:
        """Check if solution is correct (simulated)"""
        # For real puzzles with known output
        if 'train' in puzzle and len(puzzle['train']) > 0:
            if 'output' in puzzle['train'][0]:
                expected = np.array(puzzle['train'][0]['output'])
                return np.array_equal(solution, expected)
                
        # Probabilistic check for generated puzzles
        return np.random.random() < 0.3
        
    def _get_next_state(self, state: np.ndarray, action: str) -> np.ndarray:
        """Get next state after action (simulated)"""
        # Simple state transition
        next_state = state.copy()
        
        # Modify state based on action
        action_idx = self.q_agent.actions.index(action) if action in self.q_agent.actions else 0
        next_state[action_idx] += 1  # Increment action counter
        
        return next_state
        
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            'total_episodes': self.total_episodes,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(self.total_episodes, 1),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'q_table_size': len(self.q_agent.q_table),
            'epsilon': self.q_agent.epsilon,
            'buffer_size': len(self.replay_buffer.buffer)
        }