## Librairies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.special import softmax
from maze import new_lab, canEscape

colors = ["black", "blue", "orange", "red", "purple", "yellow", "green"]
bounds = [0, 1, 2, 3, 4, 5, 6, 7]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


# env configuration
SIZE = 10
NB_EPISODES = 500
NB_MAX_MOVES = 5 * SIZE * SIZE


# parameters
BETA = 0.1  # softmax parameter
EPSILON = 0.3  # probability of random choice
EPS_FACTOR = 0.999  # Every episode will be epsilon*EPS_DECAY
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
LOOP_PENALTY = 100

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
        self.best_Q = None
        self.best_reward = -np.inf

    def get_coord(self):
        return (self.x, self.y)

    def set_coord(self, coord):
        self.x, self.y = coord

    def set_Q(self, Q):
        self.Q = np.copy(Q)

    def reset(self, coord):

        if self.best_reward <= self.cumulative_reward:
            self.best_reward = self.cumulative_reward
            self.best_Q = np.copy(self.Q)

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


def real_env(Maze):
    # modify the maze by mobing the survivors, fires and pitfall near from original position

    def shuffle_position(maze):
        for (y, x) in np.argwhere(maze == SAVING_REWARD):
            shift = allowed_moves[np.random.choice(len(allowed_moves))]
            if 0 <= y + shift[0] < SIZE and 0 <= x + shift[1] < SIZE:
                maze[y, x] = -MOVING_PENALTY
                maze[y + shift[0], x + shift[1]] = SAVING_REWARD

        for (y, x) in np.argwhere(maze == -PITFALL_PENALTY):
            shift = allowed_moves[np.random.choice(len(allowed_moves))]
            if 0 <= y + shift[0] < SIZE and 0 <= x + shift[1] < SIZE:
                maze[y, x] = -MOVING_PENALTY
                maze[y + shift[0], x + shift[1]] = -PITFALL_PENALTY

        for (y, x) in np.argwhere(maze == -FIRE_PENALTY):
            shift = allowed_moves[np.random.choice(len(allowed_moves))]
            if 0 <= y + shift[0] < SIZE and 0 <= x + shift[1] < SIZE:
                maze[y, x] = -MOVING_PENALTY
                maze[y + shift[0], x + shift[1]] = -FIRE_PENALTY

        return maze

    maze = np.copy(Maze)

    maze = shuffle_position(maze)
    while not canEscape(maze, entrances, exits):
        maze = shuffle_position(maze)

    return maze


def qlearning(player, maze, episode, decision_making, survivor, fire):

    path = []
    if survivor == 0:
        for (x, y) in exits:
            maze[x, y] = ESCAPING_REWARD

    for move in range(NB_MAX_MOVES):
        obs = player.get_coord()
        path.append(obs)

        if obs in exits and maze[obs] == ESCAPING_REWARD:
            break

        # action selection
        if np.random.random() > EPSILON * EPS_FACTOR ** (episode):
            # epsilon greedy
            if decision_making == "greedy":
                action = player.possible_moves()[
                    np.argmax([player.Q[obs][i] for i in player.possible_moves()])
                ]  # action with the highest q value
            # softmax
            elif decision_making == "stoch":
                p = softmax(
                    [BETA * player.Q[obs][i] for i in player.possible_moves()]
                )  # probabilities for different actions
                action = np.random.choice(player.possible_moves(), p=p)

        else:
            action = np.random.choice(player.possible_moves())

        # updating the player position
        player.move(action)

        # loop detection
        # if (
        #     len(path) > 1
        #     and path[-2] == player.get_coord()
        #     and maze[player.get_coord()] == -MOVING_PENALTY
        # ):
        #     maze[player.get_coord()] = -LOOP_PENALTY

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

        if reward == SAVING_REWARD:
            maze[player.get_coord()] = -MOVING_PENALTY

            for (x, y) in exits:
                maze[x, y] = ESCAPING_REWARD

        if move == NB_MAX_MOVES:
            player.cumulative_reward = -np.inf

    return player, path


def train(Maze, player, decision_making="stoch", survivor=2, pitfall=0, fire=0):

    # before starting the Q learning algorithm, we should define the state space
    rewards = []

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
        rewards.append(player.cumulative_reward)

    print("")
    plt.plot(rewards)
    print(np.argmax(rewards))
    plt.show()

    return player


def test(
    Maze,
    player,
    shots=1,
    decision_making="stoch",
    fire=0,
    survivor=2,
):

    paths = []

    for _ in range(shots):
        player.reset(entrances[np.random.choice(len(entrances), replace=False)])
        player, path = qlearning(
            player, np.copy(Maze), np.inf, decision_making, fire, survivor
        )
        paths.append(path)

    player.reset(entrances[np.random.choice(len(entrances), replace=False)])
    player, path = qlearning(
        player, np.copy(Maze), np.inf, decision_making, fire, survivor
    )
    paths.append(path)

    return player, paths


def model(
    doTest=True,
    shots=1,
    survivor=2,
    pitfall=2,
    fire=1,
):

    player, Maze = create_env(survivor, pitfall, fire)
    show_maze(Maze, [])
    paths = []

    print("Training...")
    player = train(Maze, player, "stoch", survivor, pitfall, fire)
    print("Training done")

    if doTest:
        print("Testing...")

        Maze = real_env(Maze)

        player, paths = test(Maze, player, shots, "stoch", survivor, fire)
        print("Testing done")

        show_maze(Maze, paths[-1])

    return player, Maze, paths


#%%
def show_maze(
    Maze,
    path=[],
):

    maze = np.copy(Maze)

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
    maze[maze == ESCAPING_REWARD] = 3
    maze[maze == SAVING_REWARD] = 7

    for i, j in exits:
        maze[i, j] = 2

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    im = plt.imshow(maze, interpolation="none", cmap=cmap, norm=norm, animated=True)

    if path != []:

        def animatefct(i):
            maze[maze == 7.1] = 5
            if i < len(path) - 1:
                maze[path[i]] = 7.1
            elif i == len(path) - 1:
                maze[path[i]] = 5
            im.set_array(maze)
            return [im]

        anim = FuncAnimation(fig, animatefct, frames=len(path), repeat=True)
        anim.save("maze.gif", fps=20)
    else:
        plt.show()
