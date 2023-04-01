"""
Module containing the functions to create a maze

Meaning of the values in the maze
-----------------------------------------------
0 : corridor
1 : wall
2 : survivor
3 : pitfall
4 : fire


"""

import numpy as np
from queue import Queue
import matplotlib.pyplot as plt

m_free = 0
m_wall = 1
m_survivor = 2
m_pitfall = 3
m_fire = 4


def new_lab(entrees, sorties, survivor, pitfall, fire, size=(10, 10), alpha=0.7):
    """
    Create a maze with the given entrances and exits which can be escaped

    Parameters
    ----------
    entrees : list of tuples
        List of the coordinates of the entrances
    sorties : list of tuples
        List of the coordinates of the exits
    size :s tuple
        n x m size of the maze
    alpha : float
        Probability of a cell to be a corridor
    survivor : float
        Number of survivors in the maze
    pitfall : float
        Number of pitfalls in the maze
    fire : float
        Number of fires in the maze

    Return
    -------
    labyrinthe : numpy array
        The maze

    """

    labyrinthe = creation_lab(entrees, sorties, size, alpha, survivor, pitfall, fire)

    ## Parcours labyrinthe
    while not canEscape(labyrinthe, entrees, sorties):
        labyrinthe = creation_lab(
            entrees, sorties, size, alpha, survivor, pitfall, fire
        )

    return labyrinthe


def creation_lab(entrances, exits, size, alpha, survivor, pitfall, fire):
    """
    Create a maze with the given entrances and exits

    Parameters
    ----------
    entrances : list of tuples
        List of the coordinates of the entrances
    exits : list of tuples
        List of the coordinates of the exits
    size : tuple
        n x m size of the maze
    alpha : float
        Probability of a cell to be a corridor
    survivor : float
        Number of survivors in the maze
    pitfall : float
        Number of pitfalls in the maze
    fire : float
        Number of fires in the maze


    Return
    -------
    maze : numpy array
        The maze

    """

    ## Maze initialisation
    maze = np.ones(size) * -1

    queue = Queue()
    for (y, x) in entrances + exits:
        maze[y, x] = m_free
        add_adjacent(y, x, maze, queue)

    while not queue.empty():
        k, l = queue.get()
        if np.random.random() <= alpha:
            maze[k, l] = m_free
            add_adjacent(k, l, maze, queue)
        else:
            maze[k, l] = m_wall

    free = np.argwhere(maze == 0)
    r_survivor = free[np.random.choice(free.shape[0], survivor, replace=False), :]
    random = np.array(
        (
            np.random.randint(size[0], size=pitfall + fire),
            np.random.randint(size[1], size=pitfall + fire),
        )
    ).T

    for x, y in r_survivor:
        if (x, y) in entrances + exits:
            return creation_lab(entrances, exits, size, alpha, survivor, pitfall, fire)

    for x, y in r_survivor:
        maze[x, y] = m_survivor
    for x, y in random[:pitfall, :]:
        maze[x, y] = m_pitfall
    for x, y in random[pitfall:, :]:
        maze[x, y] = m_fire

    maze[maze == -1] = m_wall

    return maze


def canEscape(maze, entrances, exits):
    """
    Test if the maze can be escaped from the entrance to the exit

    Parameters
    ----------
    maze : numpy array
        The maze
    entrances : list of tuples
        List of the coordinates of the entrances
    exits : list of tuples
        List of the coordinates of the exits

    Return
    -------
    bool
        True if the maze can be escaped, False otherwise

    """

    maze = np.copy(maze)

    #  extent fire to adjacent cells
    for (y, x) in np.argwhere(maze == m_fire):
        for h in range(-2, 3):
            for w in range(-2, 3):
                # why is there an excpetion of index out of range here?
                if (
                    0 <= y + h < len(maze)
                    and 0 <= x + w < len(maze[0])
                    and h**2 + w**2 <= 4
                ):
                    if maze[y + h, x + w] == m_survivor:
                        return False
                    else:
                        maze[y + h, x + w] = m_wall

    queue = Queue()
    for (y_entrance, x_entrance) in entrances:
        maze[y_entrance, x_entrance] = -1
        queue.put((y_entrance, x_entrance))

    while not queue.empty():
        k, l = queue.get()
        maze[k, l] = -1
        add_adjacent(k, l, maze, queue, lookForCorridor=True)

    canExit = True
    for (y, x) in exits:
        canExit = maze[y, x] == -1 and canExit

    canSave = True
    for (x, y) in np.argwhere(maze == 2):
        canSave = maze[y, x] == -1 and canSave

    return canExit and canSave


def add_adjacent(i, j, maze, queue, lookForCorridor=False):
    """
    Add adjacent cells to the queue

    Parameters
    ----------
    i : int
        Height of the cell
    j : int
        Width of the cell
    maze : numpy array
        The maze to explore
    queue : Queue
        The queue to add the adjacent cells
    lookForCorridor : bool
        If True, the adjacent cells must be corridors

    """

    height = len(maze)
    width = len(maze[0])

    for x in [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]:
        if width > x[1] > -1 and height > x[0] > -1:
            if lookForCorridor:
                if maze[x[0], x[1]] == m_free:
                    queue.put((x[0], x[1]))
            else:
                if maze[x[0], x[1]] == -1:
                    queue.put((x[0], x[1]))
    return None
