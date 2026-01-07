import numpy as np
import os

MODEL_FILE = "q_table.npy"
BINS_X = np.linspace(-2.4, 2.4, 6)
BINS_X_DOT = np.linspace(-3.0, 3.0, 6)
BINS_THETA = np.linspace(-0.25, 0.25, 12)
BINS_THETA_DOT = np.linspace(-3.0, 3.0, 12)

class QLearningAgent:
    def __init__(self):
        self.alpha = 0.1         
        self.gamma = 0.99          
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        
        self.q_table = None
        self.epsilon = self.epsilon_start
        self.episode_count = 0
        self.is_training = True
        
        self.q_shape = (len(BINS_X)+1, len(BINS_X_DOT)+1, len(BINS_THETA)+1, len(BINS_THETA_DOT)+1, 2)
        
        self.load_model()

    def get_state_index(self, state_vector):
        x, x_dot, theta, theta_dot = state_vector
        idx_x = np.digitize(x, BINS_X)
        idx_x_dot = np.digitize(x_dot, BINS_X_DOT)
        idx_theta = np.digitize(theta, BINS_THETA)
        idx_theta_dot = np.digitize(theta_dot, BINS_THETA_DOT)
        return (idx_x, idx_x_dot, idx_theta, idx_theta_dot)

    def get_action(self, state_disc):
        if self.is_training and np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state_disc])

    def train(self, state_disc, action, reward, next_state_disc, done):
        if not self.is_training:
            return

        old_value = self.q_table[state_disc + (action,)]
        next_max = np.max(self.q_table[next_state_disc])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_disc + (action,)] = new_value
        
        if done:
            self.episode_count += 1
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            print(f"Loading brain from: {MODEL_FILE}")
            try:
                data = np.load(MODEL_FILE, allow_pickle=True).item()
                self.q_table = data.get('q_table')
                self.episode_count = data.get('episode', 0)
                print(f"Loaded brain with {self.episode_count} episodes.")
                self.is_training = False
                self.epsilon = self.epsilon_end
            except Exception:
                print("Model incompatible or missing. Starting fresh.")
                self.reset_model()
        else:
            print("No model found. Starting fresh.")
            self.reset_model()

    def save_model(self):
        state_to_save = {
            'q_table': self.q_table,
            'episode': self.episode_count
        }
        np.save(MODEL_FILE, state_to_save)
        print(f"Model saved! (Episode {self.episode_count})")

    def reset_model(self):
        self.q_table = np.zeros(self.q_shape)
        self.epsilon = self.epsilon_start
        self.episode_count = 0
        self.is_training = True
        print("Model RESET.")
