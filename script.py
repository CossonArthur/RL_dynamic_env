## Librairies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.special import softmax
from maze import new_lab

colors = ["black", "blue", "orange", "red", "purple", "yellow", "green"]
bounds = [0, 1, 2, 3, 4, 5, 6, 7]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


# env configuration
SIZE = 10
NB_EPISODES = 100
NB_MAX_MOVES = 5 * SIZE * SIZE


# parameters
BETA = 0.1  # softmax parameter
EPSILON = 0.5  # probability of random choice
EPS_FACTOR = 0.99  # Every episode will be epsilon*EPS_DECAY
LEARNING_RATE = 0.4  # learning rate
DISCOUNT = 0.8  # discount factor
proba_pitfall = 0.7  # probability of having a pitfall
fire_prob_spread = 0.3  # probability of fire spreading

# rewards
MOVING_PENALTY = 1
WALL_PENALTY = 20000
FIRE_PENALTY = 50
PITFALL_PENALTY = 50
SAVING_REWARD = 100
ESCAPING_REWARD = 200

# element of env
allowed_moves = [(0, 1), (1, 0), (-1, 0), (0, -1)]
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
        for i in range(len(allowed_moves)):
            if (
                self.x + allowed_moves[i][0] >= 0
                and self.x + allowed_moves[i][0] < SIZE
                and self.y + allowed_moves[i][1] >= 0
                and self.y + allowed_moves[i][1] < SIZE
            ):
                moves.append(i)
        return moves

    def move(self, num):

        self.x += allowed_moves[num][0]
        self.y += allowed_moves[num][1]

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


def train(decision_making="stoch", player_start=(0, 0), survivor=2, pitfall=0, fire=0):

    # before starting the Q learning algorithm, we should define the state space

    Q = np.zeros((SIZE, SIZE, 4))
    awards = []
    paths = []
    Qs = []
    moves = []

    player, maze_original = create_env(player_start, survivor, pitfall, fire)

    for episode in range(NB_EPISODES):

        # reset the env
        player.reset(entrances[np.random.choice(len(entrances), replace=False)])
        maze = np.copy(maze_original)
        Path = []

        for move in range(NB_MAX_MOVES):
            obs = player.get_coord()
            Path.append(obs)

            if obs in exits and maze[obs] == ESCAPING_REWARD:
                break

            # action selection
            if np.random.random() > EPSILON * EPS_FACTOR ** (move * episode):
                # epsilon greedy
                if decision_making == "greedy":
                    action = np.argmax(
                        [
                            Q[obs][i]
                            for i in np.random.permutation(player.possible_moves())
                        ]
                    )  # action with the highest q value
                # softmax
                elif decision_making == "stoch":
                    p = softmax(
                        [BETA * Q[obs][i] for i in player.possible_moves()]
                    )  # probabilities for different actions
                    action = np.random.choice(player.possible_moves(), p=p)

            else:
                action = np.random.choice(player.possible_moves())
                j = len(player.possible_moves())
                while (
                    maze[
                        (
                            obs[0] + allowed_moves[action][0],
                            obs[1] + allowed_moves[action][1],
                        )
                    ]
                    == -WALL_PENALTY
                    and j > 0
                ):
                    action = np.random.choice(player.possible_moves())
                    j -= 1

            # updating the player position
            player.move(action)

            # calcul of the reward
            reward = maze[player.get_coord()]
            player.cumulative_reward += reward

            # updating the q table
            Q[obs][action] = (1 - LEARNING_RATE) * Q[obs][action] + LEARNING_RATE * (
                reward + DISCOUNT * np.max(Q[player.get_coord()])
            )

            # update fire postion
            if fire > 0:
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

                                proba = np.random.random()
                                if proba < fire_prob_spread * 1.001**move:
                                    maze[(y + h, x + w)] = -FIRE_PENALTY + 1

            if reward == SAVING_REWARD:
                maze[player.get_coord()] = -MOVING_PENALTY

                for (x, y) in exits:
                    maze[x, y] = ESCAPING_REWARD

            elif reward not in [
                -WALL_PENALTY,
                -FIRE_PENALTY,
                -PITFALL_PENALTY,
                ESCAPING_REWARD,
            ]:
                maze[player.get_coord()] += -MOVING_PENALTY

            if moves == NB_MAX_MOVES:
                player.cumulative_reward = -99999999

        awards.append(player.cumulative_reward)
        paths.append(Path)
        Qs.append(np.copy(Q))
        moves.append(move)

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
            maze[path[i]] = 5.5
        im.set_array(maze)
        return [im]

    anim = FuncAnimation(fig, animatefct, frames=len(path), repeat=True)
    plt.axis("off")
    anim.save("maze.gif", fps=20)


def show_path(award_list):
    plt.plot(award_list)


maze, paths, awards, Qs, moves = train()
