
import numpy as np
from typing import List
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from enum import Enum

class Direction(Enum):
    LEFT = 0
    RIGHT = 1

bins_x = np.linspace(-3.0, 3.0, 10)
bins_theta = np.linspace(-1.5, 1.5, 20)


nr_stari_x = len(bins_x) + 1
nr_stari_theta = len(bins_theta) + 1
nr_actiuni = 2

q_table = np.zeros((nr_stari_x, nr_stari_theta, nr_actiuni))

print(q_table.shape)

max_steps = 0


def get_state(state_real: List):
    x_real = state_real[0]
    theta_real = state_real[2]

    #scadem 1 pentru a avea indecsi de la 0

    index_x = np.digitize(x_real, bins_x)
    index_theta = np.digitize(theta_real, bins_theta)

    #ne asiguram ca index_x ramane in [0,10] si index_theta ramane in [0,20]

    index_x = np.clip(index_x, 0, nr_stari_x - 1)
    index_theta = np.clip(index_theta, 0, nr_stari_theta - 1)

    return (index_x, index_theta)


class HammerEnvironment:
    def __init__(self):
        self.gravity = 9.8
        self.length = 0.5
        self.mass_cart = 1.0
        self.mass_hammer = 0.13

        self.total_mass = self.mass_cart + self.mass_hammer

        self.polemass_length = self.mass_hammer * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.theta_threshold_radians = math.pi / 2
        self.x_threshold = 3.0

        self.reset()

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        #pozitie carucior, viteza carucior, unghi bara/pendul, viteza unghiulara bara/pendul
        self.steps_survived = 0
        return self.state

    def step(self, action):
        x_pos, x_speed, theta, theta_angular = self.state

        if action == Direction.RIGHT.value:
            force = self.force_mag
        else:
            force = -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_angular * theta_angular * sintheta) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.mass_hammer * costheta * costheta / self.total_mass))

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x_pos = x_pos + self.tau * x_speed
        theta = theta + self.tau * theta_angular

        x_speed = x_speed + self.tau * xacc
        theta_angular = theta_angular + self.tau * thetaacc

        self.state = (x_pos, x_speed, theta, theta_angular)

        done = bool(
            x_pos < -self.x_threshold
            or x_pos > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward_theta = 1.0 - (abs(theta) / self.theta_threshold_radians)
            reward_x = 1.0 - (abs(x_pos) / self.x_threshold)
            reward = (reward_theta * 0.8) + (reward_x * 0.2)  # 8
        else:
            reward = -100.0

        return self.state, reward, done


# Training

ALPHA = 0.1
GAMMA = 0.98
EPSILON = 1.0
EPSILON_DECAY = 0.995

env = HammerEnvironment()

print("Antrenamentul cu 1000 de episoade:")

for episod in range(1000):

    state_real = env.reset()
    #pornim de la valori random nu din pozitie 0,0,0,0 pt ca agentul sa poata invata

    state_disc = get_state(state_real)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < EPSILON:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(q_table[state_disc])

        next_state_real, reward, done = env.step(action)
        next_state_disc = get_state(next_state_real)

        old_value = q_table[state_disc + (action,)]
        next_max = np.max(q_table[next_state_disc])

        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[state_disc + (action,)] = new_value

        state_disc = next_state_disc
        total_reward += reward

    if EPSILON > 0.01:
        EPSILON *= EPSILON_DECAY

    if (episod + 1) % 100 == 0:
        print(f"Episod: {episod + 1}, Scor Total: {total_reward:.2f}, Epsilon: {EPSILON:.2f}")



EPSILON = 0.0
state_real = env.reset()
state_disc = get_state(state_real)
frame_count = 0

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-1.0, 1.5)
ax.set_aspect('equal')
plt.title("Rezultat Q-Learning: Echilibrare Automata")

ax.axhline(y=0, color='black', linewidth=1)

#Desenare obiect

cart = patches.Rectangle((-0.2, -0.1), 0.4, 0.2, fc='#2ecc71', ec='black', zorder=5)
ax.add_patch(cart)

# Ciocanul este format din coada (linie) si cap (dreptunghi)
hammer_handle, = ax.plot([], [], 'k-', linewidth=4, zorder=4)
hammer_head = patches.Rectangle((0, 0), 0.2, 0.15, fc='#e74c3c', ec='black', zorder=6)
ax.add_patch(hammer_head)

info_text = ax.text(-2.8, 1.2, '', fontsize=12)


def update(frame):
    global state_real, state_disc, frame_count, max_steps

    action = np.argmax(q_table[state_disc])

    next_state_real, reward, done = env.step(action)
    next_state_disc = get_state(next_state_real)

    state_real = next_state_real
    state_disc = next_state_disc
    frame_count += 1

    x, x_speed, theta, theta_speed = state_real

    cart.set_x(x - 0.2)

    hammer_len = 0.8
    end_x = x + hammer_len * math.sin(theta)
    end_y = hammer_len * math.cos(theta)

    hammer_handle.set_data([x, end_x], [0, end_y])

    hammer_head.set_xy((end_x - 0.1, end_y - 0.075))

    if done:
        state_real = env.reset()
        state_disc = get_state(state_real)

        if frame_count > max_steps:
            max_steps = frame_count
            print(max_steps)

        frame_count = 0
        cart.set_facecolor('red')  # Flash roșu la reset

    else:
        cart.set_facecolor('#2ecc71')  # Verde cand e ok

    info_text.set_text(f"Unghi: {math.degrees(theta):.1f}° | Pasi Rezistati: {frame_count} | Pasi Rezistati Maxim: {max_steps}")

    return cart, hammer_handle, hammer_head, info_text


# Pornire Animație
ani = animation.FuncAnimation(fig, update, frames=None, interval=20, blit=False)
plt.show()