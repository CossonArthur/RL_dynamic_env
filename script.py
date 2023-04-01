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
NB_EPISODES = 200
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
FIRE_PENALTY = 70
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
        self.Q = np.zeros((SIZE, SIZE, len(allowed_moves)))
        self.BestQ = None
        self.best_reward = -np.inf

    def get_coord(self):
        return (self.x, self.y)

    def set_coord(self, coord):
        self.x, self.y = coord

    def reset(self, coord, Q=None):
        if not Q and self.best_reward < self.cumulative_reward:
            self.best_reward = self.cumulative_reward
            self.BestQ = np.copy(Q)

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


def create_env(survivor, pitfall, fire):
    player = Player(entrances[0])

    maze = new_lab(entrances, exits, survivor, pitfall, fire, size=(SIZE, SIZE))

    maze[maze == 0] = -MOVING_PENALTY
    maze[maze == 1] = -WALL_PENALTY
    maze[maze == 2] = SAVING_REWARD
    maze[maze == 3] = -PITFALL_PENALTY
    maze[maze == 4] = -FIRE_PENALTY

    return player, maze


def real_env(maze):
    # modify the maze by mobing the survivors, fires and pitfall near from original position

    # survivor

    return maze


def qlearning(player, maze, episode, decision_making, survivor, fire):

    path = []

    for move in range(NB_MAX_MOVES):
        obs = player.get_coord()
        path.append(obs)

        if obs in exits and maze[obs] == ESCAPING_REWARD:
            break

        # action selection
        if np.random.random() > EPSILON * EPS_FACTOR ** (episode + np.sqrt(move)):
            # epsilon greedy
            if decision_making == "greedy":
                action = np.argmax(
                    [
                        player.Q[obs][i]
                        for i in np.random.permutation(player.possible_moves())
                    ]
                )  # action with the highest q value
            # softmax
            elif decision_making == "stoch":
                p = softmax(
                    [BETA * player.Q[obs][i] for i in player.possible_moves()]
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
        player.Q[obs][action] = (1 - LEARNING_RATE) * player.Q[obs][
            action
        ] + LEARNING_RATE * (reward + DISCOUNT * np.max(player.Q[player.get_coord()]))

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

        if reward == SAVING_REWARD or survivor == 0:
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

        if move == NB_MAX_MOVES:
            player.cumulative_reward = -np.inf

    return player, path


def train(Maze, player, decision_making="stoch", survivor=2, pitfall=0, fire=0):

    # before starting the Q learning algorithm, we should define the state space

    for episode in range(NB_EPISODES):
        print("Episode {} on {}".format(episode + 1, NB_EPISODES), end="\r")

        # reset the env
        player.reset(entrances[np.random.choice(len(entrances), replace=False)])
        maze = np.copy(Maze)

        # add pitfalls
        if pitfall > 0:
            for (y, x) in np.argwhere(maze == -PITFALL_PENALTY):
                if np.random.random() > proba_pitfall:
                    maze[y, x] = -WALL_PENALTY

        player, _ = qlearning(player, maze, episode, decision_making, fire, survivor)

    print("")
    return player


def test(
    Maze,
    player,
    shots=1,
    decision_making="stoch",
    fire=0,
    survivor=2,
):

    for _ in range(shots):
        player.reset(entrances[np.random.choice(len(entrances), replace=False)])
        maze = np.copy(Maze)
        player, _ = qlearning(player, maze, 0, decision_making, fire, survivor)

    player.reset(entrances[np.random.choice(len(entrances), replace=False)])
    maze = np.copy(Maze)

    return qlearning(player, maze, 0, decision_making, fire, survivor)


def model(
    doTest=True,
    survivor=2,
    pitfall=0,
    fire=0,
):

    player, Maze = create_env(survivor, pitfall, fire)
    show_map(Maze, [])
    player.Q = np.zeros((SIZE, SIZE, len(allowed_moves)))

    print("Training...")
    player = train(Maze, player, "stoch", survivor, pitfall, fire)
    print("Training done")

    if doTest:
        print("Testing...")

        # Maze = real_env(Maze, survivor, pitfall, fire)

        player, path = test(Maze, player, 1, "stoch", survivor, fire)
        print("Testing done")

        show_map(Maze, path)


#%%
def show_map(
    maze,
    path=[],
):

    # modification of data for rendering
    for (y, x) in np.argwhere(maze == -FIRE_PENALTY):
        for h in range(-2, 3):
            for w in range(-2, 3):
                if (
                    w**2 + h**2 <= 4
                    and 0 <= y + h < len(maze)
                    and 0 <= x + w < len(maze[0])
                ):
                    maze[(y + h, x + w)] = 2

    maze[maze == -MOVING_PENALTY] = 0
    maze[maze == -WALL_PENALTY] = 1
    maze[maze == -PITFALL_PENALTY] = 4
    maze[maze == -FIRE_PENALTY] = 2
    maze[maze == SAVING_REWARD] = 7

    for i, j in exits:
        maze[i, j] = 2

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    im = plt.imshow(maze, interpolation="none", cmap=cmap, norm=norm, animated=True)

    if path != []:

        def animatefct(i):
            maze[maze == 7.1] = 5
            if i < len(path):
                maze[path[i]] = 7.1
            im.set_array(maze)
            return [im]

        anim = FuncAnimation(fig, animatefct, frames=len(path), repeat=True)
        anim.save("maze.gif", fps=20)
    else:
        plt.show()
