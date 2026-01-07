import math
import matplotlib

try:
    matplotlib.use('QtAgg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from environment import HammerEnvironment
from agent import QLearningAgent

class InvertedPendulumApp:
    def __init__(self):

        self.env = HammerEnvironment()
        self.agent = QLearningAgent()
        
        self.state_real = self.env.reset()
        self.state_disc = self.agent.get_state_index(self.state_real)
        self.steps_survived = 0
        self.max_steps = 0
        self.simulation_speed = 1
        
        self.initial_episodes = self.agent.episode_count
        
        self.setup_plot()

    def setup_plot(self):
        plt.style.use('dark_background')
        plt.rcParams['keymap.save'] = []
        plt.rcParams['keymap.fullscreen'] = []

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Proiect IA - Q-Learning')

        self.ax.set_xlim(-3.0, 3.0)
        self.ax.set_ylim(-1.0, 2.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax.axhline(y=-0.26, color='#7f8c8d', linewidth=2)
        
        self.cart_w, self.cart_h = 0.5, 0.3
        self.cart = patches.Rectangle((0,0), self.cart_w, self.cart_h, fc='#3498db', ec='white', linewidth=2, zorder=10)
        self.ax.add_patch(self.cart)
        
        r = 0.08
        self.wheel_l = patches.Circle((0,0), r, fc='#95a5a6', ec='white', linewidth=1)
        self.wheel_r = patches.Circle((0,0), r, fc='#95a5a6', ec='white', linewidth=1)
        self.ax.add_patch(self.wheel_l)
        self.ax.add_patch(self.wheel_r)

        self.pole_line, = self.ax.plot([], [], color='#ecf0f1', linewidth=4, zorder=5)
        self.head_w, self.head_h = 0.25, 0.12
        self.hammer_head = patches.Polygon([[0,0]], fc='#e74c3c', ec='white', linewidth=1, zorder=12)
        self.ax.add_patch(self.hammer_head)
        self.pole_joint = patches.Circle((0, 0), 0.05, fc='#f1c40f', zorder=15)
        self.ax.add_patch(self.pole_joint)

        self.status_text = self.ax.text(-2.8, 1.7, '', fontsize=10, color='white', family='monospace')
        self.control_text = self.ax.text(1.5, 1.7, '', fontsize=12, color='#f1c40f', fontweight='bold', ha='center')
        self.ax.text(0, -0.8, '[T] Train/Demo | [S] Save | [R] Reset | [+/-] Speed', fontsize=10, color='#95a5a6', ha='center')

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 't':
            self.agent.is_training = not self.agent.is_training
            print(f"Training Mode: {self.agent.is_training}")
        elif event.key == 's':

            import os
            import numpy as np
            MODEL_FILE = "q_table.npy"
            
            should_save = True
            if os.path.exists(MODEL_FILE):
                try:
                    old_data = np.load(MODEL_FILE, allow_pickle=True).item()
                    old_episodes = old_data.get('episode', '?')
                    
                    from PySide6.QtWidgets import QMessageBox
                    parent = self.fig.canvas.manager.window
                    text = (f"A saved model already exists with {old_episodes} episodes.\n\n"
                            f"Overwrite it with the current model ({self.agent.episode_count} episodes)?")
                    
                    reply = QMessageBox.question(parent, 'Confirm Overwrite', text,
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    
                    if reply != QMessageBox.Yes:
                        should_save = False
                except:
                    pass

            if should_save:
                self.agent.save_model()
        elif event.key == 'r':
            try:
                from PySide6.QtWidgets import QMessageBox
                parent = self.fig.canvas.manager.window
                reply = QMessageBox.question(parent, 'Confirm Reset', 
                                            "Are you sure you want to wipe the memory?\n\nThis will reset the agent to zero immediately.\n(Your saved file will only be overwritten if you press 'S' afterwards.)",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply != QMessageBox.Yes:
                    return
            except ImportError:
                pass

            self.agent.reset_model()
            self.steps_survived = 0
            self.max_steps = 0
            self.state_real = self.env.reset()
            self.state_disc = self.agent.get_state_index(self.state_real)
        elif event.key == '+' or event.key == '=':
            if self.simulation_speed < 100:
                self.simulation_speed += 1
        elif event.key == '-' and self.simulation_speed > 1:
            self.simulation_speed -= 1

    def update(self, frame):
        current_action = 0
        
        for _ in range(int(self.simulation_speed)):
            current_action = self.agent.get_action(self.state_disc)
            
            next_state_real, reward, done = self.env.step(current_action)
            next_state_disc = self.agent.get_state_index(next_state_real)
            
            self.agent.train(self.state_disc, current_action, reward, next_state_disc, done)
            
            self.state_real = next_state_real
            self.state_disc = next_state_disc
            self.steps_survived += 1

            if done:
                if self.steps_survived > self.max_steps:
                    self.max_steps = self.steps_survived
                
                self.state_real = self.env.reset()
                self.state_disc = self.agent.get_state_index(self.state_real)
                self.steps_survived = 0
                
                self.pole_line.set_color('#c0392b')
                break 
            else:
                self.pole_line.set_color('#ecf0f1')

        x, x_dot, theta, theta_dot = self.state_real

        self.cart.set_x(x - self.cart_w/2)
        self.cart.set_y(-self.cart_h/2)
        
        w_y = -self.cart_h/2 - 0.08 + 0.05
        self.wheel_l.center = (x - self.cart_w/3, w_y)
        self.wheel_r.center = (x + self.cart_w/3, w_y)
        self.pole_joint.center = (x, 0)

        pole_x = x + (self.env.length * 2) * math.sin(theta)
        pole_y = (self.env.length * 2) * math.cos(theta)
        self.pole_line.set_data([x, pole_x], [0, pole_y])

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        vec_r_x, vec_r_y = cos_t, -sin_t
        vec_u_x, vec_u_y = sin_t, cos_t
        
        dx_w = (self.head_w / 2) * vec_r_x
        dy_w = (self.head_w / 2) * vec_r_y
        dx_h = (self.head_h / 2) * vec_u_x
        dy_h = (self.head_h / 2) * vec_u_y
        
        corners = [
            (pole_x - dx_w - dx_h, pole_y - dy_w - dy_h),
            (pole_x + dx_w - dx_h, pole_y + dy_w - dy_h),
            (pole_x + dx_w + dx_h, pole_y + dy_w + dy_h),
            (pole_x - dx_w + dx_h, pole_y - dy_w + dy_h),
        ]
        self.hammer_head.set_xy(corners)

        mode_str = "TRAINING" if self.agent.is_training else "DEMO / TEST"
        force_dir = ">>> RIGHT >>>" if current_action == 1 else "<<< LEFT <<<"
        self.control_text.set_text(force_dir)
        
        status = (
            f"MODE: {mode_str}\n"
            f"Episode   : {self.agent.episode_count}\n"
            f"Epsilon   : {self.agent.epsilon:.3f}\n"
            f"Steps     : {self.steps_survived}\n"
            f"Record    : {self.max_steps}\n"
            f"Speed     : {int(self.simulation_speed * 100)}%\n"
            f"Velocity  : {x_dot:.2f}\n"
            f"Angle Vel : {theta_dot:.2f}"
        )
        self.status_text.set_text(status)
        
        return self.cart, self.pole_line, self.hammer_head

    def run(self):
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, self.show_startup_message)
        except:
            pass
            
        ani = animation.FuncAnimation(self.fig, self.update, frames=None, interval=20, blit=False, cache_frame_data=False)
        plt.show()

    def show_startup_message(self):
        try:
            from PySide6.QtWidgets import QMessageBox
            parent = self.fig.canvas.manager.window
            
            if self.initial_episodes > 0:
                QMessageBox.information(parent, 'Brain Loaded', 
                                      f"Successfully loaded brain with {self.initial_episodes} episodes of experience.\n\nThe controller is currently in DEMO mode.")
            else:
                QMessageBox.information(parent, 'Fresh Start',
                                      "No previous training data found.\n\nStarting a new brain from scratch (Episode 0).")
        except:
            pass

if __name__ == "__main__":
    app = InvertedPendulumApp()
    app.run()
