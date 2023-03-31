## Librairies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import random
from scipy.special import softmax
from maze import new_lab

colors = ["black", "blue", "orange", "red", "purple", "yellow", "green"]
bounds = [0, 1, 2, 3, 4, 5, 6, 7]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


# env configuration
SIZE = 10
HM_EPISODES = 400
TIME = 5 * SIZE * SIZE


# parameters
epsilon = 0.5  # randomness
EPS_DECAY = 0.99  # Every episode will be epsilon*EPS_DECAY
LEARNING_RATE = 0.4
DISCOUNT = 0.8
proba_pitfall = 0.7
fire_prob_spread = 0.3

# rewards
MOVING_PENALTY = 1
WALL_PENALTY = 2000
FIRE_PENALTY = 5
PITFALL_PENALTY = 1
SAVING_REWARD = 10
ESCAPING_REWARD = 20

# element of env
possible_moves = [(0, 1), (1, 0), (-1, 0), (0, -1)]
entrances = [(0, 0)]
exits = [(9, 9)]


class Player:
    def __init__(self, coord):
        self.x = coord[0]
        self.y = coord[1]
        self.cumulative_reward = 0

    def get_coord(self):
        return (self.x, self.y)

    def set_coord(self, coord):
        self.x, self.y = coord

    def reset(self, coord):
        self.x = coord[0]
        self.y = coord[1]
        self.cumulative_reward = 0

    def possible_moves(self):
        moves = []
        for i in range(4):
            if (
                self.x + possible_moves[i][0] >= 0
                and self.x + possible_moves[i][0] < SIZE
                and self.y + possible_moves[i][1] >= 0
                and self.y + possible_moves[i][1] < SIZE
            ):
                moves.append(i)
        return moves

    def move(self, num):

        self.x += possible_moves[num][0]
        self.y += possible_moves[num][1]

        # If player is out of bounds
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


def create_env(player_start, survivor, pitfall, fire):
    player = Player(player_start)

    maze = new_lab(entrances, exits, survivor, pitfall, fire, size=(SIZE, SIZE))

    maze[maze == 0] = -MOVING_PENALTY
    maze[maze == 1] = -WALL_PENALTY
    maze[maze == 2] = SAVING_REWARD
    maze[maze == 3] = -PITFALL_PENALTY
    maze[maze == 4] = -FIRE_PENALTY

    return player, maze


def qlearning(
    eps_factor=EPS_DECAY,
    learning_rate=LEARNING_RATE,
    prob_random_choice=epsilon,
    discount=DISCOUNT,
    size=SIZE,
    decision_making="stoch",
    nb_episodes=HM_EPISODES,
    nb_moves=TIME,
    player_start=(0, 0),
    survivor=2,
    pitfall=0,
    fire=0,
    fire_prob_spread=fire_prob_spread,
):
    def update_fire(maze, fire_prob_spread, move):
        for (y, x) in np.argwhere(maze == -FIRE_PENALTY):
            for h in range(-2, 3):
                for w in range(-2, 3):
                    if (
                        w**2 + h**2 <= 4
                        and y + h >= 0
                        and y + h < SIZE
                        and x + w >= 0
                        and x + w < SIZE
                    ):

                        proba = random.random()
                        if proba < fire_prob_spread * 1.001**move:
                            maze[(y + h, x + w)] = -FIRE_PENALTY + 1

    # before starting the Q learning algorithm, we should define the state space

    Q = np.zeros((size, size, 4))

    episode = 0
    awards = []
    paths = []
    Qs = []
    awards = []
    moves = []

    player, maze_original = create_env(player_start, survivor, pitfall, fire)

    while episode < nb_episodes:

        # reset the env
        player.reset(entrances[np.random.choice(len(entrances), replace=False)])
        maze = np.copy(maze_original)
        move = 0
        Path = []

        while move < nb_moves:
            obs = player.get_coord()
            Path.append(obs)

            if obs in entrances and maze[obs] == ESCAPING_REWARD:
                break

            # action selection
            if np.random.random() > prob_random_choice:
                # epsilon greedy
                if decision_making == "greedy":
                    action = np.argmax(
                        [Q[obs][i] for i in player.possible_moves()]
                    )  # action with the highest q value

                # softmax
                elif decision_making == "stoch":
                    p = softmax(
                        [Q[obs][i] for i in player.possible_moves()]
                    )  # probabilities for different actions
                    action = np.random.choice(player.possible_moves(), p=p)

            else:
                action = np.random.choice(player.possible_moves())

            # updating the player position
            player.move(action)

            # calcul of the reward
            reward = maze[obs]
            player.cumulative_reward += reward

            # updating the q table
            Q[obs][action] = (1 - learning_rate) * Q[obs][action] + learning_rate * (
                reward + discount * np.max(Q[obs])
            )

            # update fire postion
            if fire > 0:
                update_fire(maze, fire_prob_spread, move)

            if reward == SAVING_REWARD:
                maze[obs] = -MOVING_PENALTY

                for (x, y) in exits:
                    maze[x, y] = ESCAPING_REWARD
            elif reward not in [
                -WALL_PENALTY,
                -FIRE_PENALTY,
                -PITFALL_PENALTY,
                ESCAPING_REWARD,
            ]:
                maze[obs] += -MOVING_PENALTY

            move += 1
            if moves == nb_moves:
                player.cumulative_reward -= 10000

        awards.append(player.cumulative_reward)
        paths.append(Path)
        Qs.append(Q)
        moves.append(move)
        episode += 1
        prob_random_choice = prob_random_choice * eps_factor

    plt.clf()
    plt.plot(awards)
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")

    plt.show()
    show_map(np.copy(maze_original), paths[awards.index(max(awards))])

    return maze_original, paths, awards, Qs, moves


def show_map(
    maze,
    path=[],
):

    # modification of data for rendering
    maze[maze == -MOVING_PENALTY] = 0

    maze[maze == -WALL_PENALTY] = 1
    maze[maze == -PITFALL_PENALTY] = 4
    maze[maze == -FIRE_PENALTY] = 4.5
    maze[maze == -FIRE_PENALTY + 1] = 4.5
    maze[maze == SAVING_REWARD] = 7
    for i, j in exits:
        maze[i, j] = 2

    fig = plt.figure(figsize=(8, 8))
    im = plt.imshow(maze, interpolation="none", cmap=cmap, norm=norm, animated=True)

    def animatefct(i):
        if i < len(path):
            if maze[path[i]] == 0:
                maze[path[i]] = 5.5
        im.set_array(maze)
        return [im]

    anim = FuncAnimation(fig, animatefct, frames=len(path), repeat=True)
    plt.axis("off")
    anim.save("maze.gif", fps=20)


def show_path(award_list):
    plt.plot(award_list)


maze, paths, awards, Qs, moves = qlearning()
