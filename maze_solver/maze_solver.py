import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import scipy.sparse as sp
from typing import Union
import heapq

class Maze:
    
    def __init__(self, input_data: Union[str, np.ndarray] = None):
        self.map = None
        self.start = None
        self.goal = None
        if input_data is not None:
            if isinstance(input_data, str):
                self.map = self.from_csv(input_data)
            elif isinstance(input_data, np.ndarray):
                self.map = input_data
            self.start = 0
            self.goal = self.map.size - 1
            self.map[0, 0] = False
            self.map[-1, -1] = False

    def from_csv(self, filename: str) -> np.ndarray:
        return np.genfromtxt(filename, delimiter=',', dtype=int).astype(bool)
    
    def save(self, filename: str):
        np.savetxt(filename, self.map.astype(int), fmt='%d', delimiter=',')
    
    def incidence_matrix(self) -> sp.lil_matrix:
        def index(r: int, c: int) -> int:
            return r * cols + c

        rows, cols = self.map.shape
        n = rows * cols
        incidence = sp.lil_matrix((n, n), dtype=int)
        
        for r in range(rows):
            for c in range(cols):
                if not self.map[r, c]:
                    current_index = index(r, c)
                    if c + 1 < cols and not self.map[r, c + 1]:
                        right_index = index(r, c + 1)
                        incidence[current_index, right_index] = 1
                        incidence[right_index, current_index] = 1
                    if r + 1 < rows and not self.map[r + 1, c]:
                        down_index = index(r + 1, c)
                        incidence[current_index, down_index] = 1
                        incidence[down_index, current_index] = 1

        return incidence

    def dijkstra(self, incidence_matrix: sp.lil_matrix) -> np.ndarray:
        """
        Performs the Dijkstra's algorithm to find the shortest path
        in the maze using an incidence matrix.
        """
        if incidence_matrix is None:
            raise ValueError('Incidence matrix is not defined.')
    
        num_vertices = incidence_matrix.shape[0]
        distances = np.full(num_vertices, np.inf)
        previous = np.full(num_vertices, -1, dtype=int)
        distances[self.start] = 0
        priority_queue = [(0, self.start)]

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

    def solve(self):
        """
        Solves the maze and visualizes the solution.
        """

        incidence_matrix = self.incidence_matrix()
        path = self.dijkstra(incidence_matrix)
        if len(path) == 0:
            print('Path does not exist.')
            self.plot()
            return
        
        maze_visual = np.zeros(self.map.shape, dtype=int)
        path_visual = np.zeros_like(maze_visual)
        rows, cols = self.map.shape
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            if self.map[r, c]:
                maze_visual[r, c] = 1                           # Stěny jsou 1
                path_visual[r, c] = 1                           # Stěny jsou 1

        for idx in path:
            r, c = divmod(idx, cols)
            path_visual[r, c] = 2                               # Cesta je 2

        cmap_maze = ListedColormap(['white', 'black'])          # 0: průchod, 1: stěna
        cmap_path = ListedColormap(['white', 'black', 'red'])   # 0: průchod, 1: stěna, 2: cesta

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(maze_visual, cmap=cmap_maze, interpolation='nearest')
        ax[0].set_title('Maze')
        ax[0].axis('off')
        
        ax[1].imshow(path_visual, cmap=cmap_path, interpolation='nearest')
        ax[1].set_title('Solution')
        ax[1].axis('off')

        plt.show()

    def plot(self):
        cmap_maze = ListedColormap(['white', 'black'])          # 0: průchod, 1: stěna
        plt.imshow(self.map, cmap=cmap_maze)
        plt.show()