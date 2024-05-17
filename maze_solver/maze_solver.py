"""
Module for defining and solving mazes using Dijkstra's algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.sparse as sp
from typing import Union, List, Tuple
import heapq


class Maze:
    """
    Class for defining and solving mazes using Dijkstra's algorithm.

    Can be initialized from a CSV file or a 2D NumPy array.

    Attributes:
        map (np.ndarray): 2D array representing the maze.
        start (int): Index of the start position in the maze.
        goal (int): Index of the goal position in the maze.

    Methods:
        from_csv: Reads the maze from a CSV file.
        save_csv: Saves the maze to a CSV file.
        incidence_matrix: Computes the incidence matrix of the maze.
        dijkstra: Performs Dijkstra's algorithm to find the shortest path.
        solve: Solves the maze and visualizes the solution.
        plot: Plots the maze.
    """

    def __init__(self, input_data: Union[str, np.ndarray, None] = None):
        """
        Initializes the maze from a CSV file or a 2D NumPy array by setting the map attribute with start and goal positions to the corners using boolean values.

        Args:
            input_data (Union[str, np.ndarray]): Path to a CSV file or a 2D NumPy array.
        """
        self.map = None
        self.start = None
        self.goal = None
        if input_data is not None:
            if isinstance(input_data, str):
                self.map = self.from_csv(input_data)
            elif isinstance(input_data, np.ndarray):
                self.map = input_data.astype(bool)
            if self.map.shape[0] != self.map.shape[1]:
                raise ValueError("Maze must be rectangular.")
            self.start = 0
            self.goal = self.map.size - 1
            self.map[0, 0] = False
            self.map[-1, -1] = False

    def from_csv(self, filename: str) -> np.ndarray:
        """
        Reads the maze from a CSV file as a 2D NumPy array with boolean values and returns it.

        Args:
            filename (str): Path to the CSV file.

        Returns:
            np.ndarray: 2D NumPy array representing the maze.
        """
        return np.genfromtxt(filename, delimiter=",", dtype=int).astype(bool)

    def save_csv(self, filename: str):
        """
        Saves the maze to a CSV file.

        Args:
            filename (str): Path to the CSV file.
        """
        if self.map is not None:
            np.savetxt(filename, self.map.astype(int), fmt="%d", delimiter=",")
        else:
            raise ValueError("No maze to save.")

    def incidence_matrix(self) -> sp.lil_matrix:
        """
        Computes the incidence matrix of the maze.

        Returns:
            sp.lil_matrix: Sparse matrix representing the incidence matrix.
        """

        if self.map is None:
            raise ValueError("No maze to compute the incidence matrix.")

        def index(r: int, c: int) -> int:
            """
            Returns the index of the cell in the maze.

            Args:
                r (int): Row index.
                c (int): Column index.

            Returns:
                int: Index of the cell in the maze.
            """
            return r * cols + c

        rows, cols = self.map.shape
        n = rows * cols
        incidence = sp.lil_matrix((n, n), dtype=int)

        for r in range(rows):
            for c in range(cols):
                if not self.map[r, c]:
                    current_index = index(r, c)
                    if c + 1 < cols and not self.map[r, c + 1]:  # Right
                        right_index = index(r, c + 1)
                        incidence[current_index, right_index] = 1  # Right
                        incidence[right_index, current_index] = 1  # Left
                    if r + 1 < rows and not self.map[r + 1, c]:  # Down
                        down_index = index(r + 1, c)
                        incidence[current_index, down_index] = 1  # Down
                        incidence[down_index, current_index] = 1  # Up

        return incidence

    def dijkstra(self, incidence_matrix: sp.lil_matrix) -> np.ndarray:
        """
        Performs Dijkstra's algorithm to find the shortest path in the maze.
        If the path does not exist, an empty array is returned.

        Args:
            incidence_matrix (sp.lil_matrix): Sparse matrix representing the incidence matrix.

        Returns:
            np.ndarray: Array of indices representing the shortest path.
        """
        if self.map is None:
            raise ValueError("No maze to solve.")

        num_vertices = incidence_matrix.shape[0]
        distances = np.full(num_vertices, np.inf)
        previous = np.full(num_vertices, -1, dtype=int)
        distances[self.start] = 0
        priority_queue: List[Tuple[float, int | None]] = [(0, self.start)]

        while priority_queue:
            current_distance, u = heapq.heappop(priority_queue)
            if u == self.goal:
                break

            for v in range(num_vertices):
                if incidence_matrix[u, v] != 0:
                    alt = current_distance + 1
                    if alt < distances[v]:
                        distances[v] = alt
                        previous[v] = u
                        heapq.heappush(priority_queue, (alt, v))

        path = []
        step = self.goal
        if distances[self.goal] == np.inf:
            return np.array([])
        while step != -1:
            path.append(step)
            step = previous[step]
        return np.array(path[::-1])

    def solve(self) -> None:
        """
        Solves the maze and visualizes the solution by plotting the original maze and the solution path.

        Walls are represented by black color, the path by red color, and the empty space by white color.

        If the path does not exist, a message is printed and only the original maze is plotted.
        """
        if self.map is None:
            raise ValueError("No maze to solve.")

        incidence_matrix = self.incidence_matrix()
        path = self.dijkstra(incidence_matrix)
        if len(path) == 0:
            print("Path does not exist.")
            self.plot()
            return

        maze_visual = np.zeros(self.map.shape, dtype=int)
        path_visual = np.zeros_like(maze_visual)
        rows, cols = self.map.shape
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            if self.map[r, c]:
                maze_visual[r, c] = 1  # Walls are 1
                path_visual[r, c] = 1  # Walls are 1

        for idx in path:
            r, c = divmod(idx, cols)
            path_visual[r, c] = 2  # Path is 2

        cmap_maze = ListedColormap(["white", "black"])  # 0: passage, 1: wall
        cmap_path = ListedColormap(
            ["white", "black", "red"]
        )  # 0: passage, 1: wall, 2: path

        # Plot the maze and the solution
        _, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(maze_visual, cmap=cmap_maze, interpolation="nearest")
        ax[0].set_title("Maze")
        ax[0].axis("off")

        ax[1].imshow(path_visual, cmap=cmap_path, interpolation="nearest")
        ax[1].set_title("Solution")
        ax[1].axis("off")

        plt.show()

    def plot(self) -> None:
        """
        Plots the maze.

        Walls are represented by black color and the empty space by white color.
        """
        if self.map is None:
            raise ValueError("No maze to plot.")

        cmap_maze = ListedColormap(["white", "black"])  # 0: passage, 1: wall
        plt.imshow(self.map, cmap=cmap_maze)
        plt.title("Maze - No Solution")
        plt.axis("off")
        plt.show()
