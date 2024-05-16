from .maze_solver import Maze
import numpy as np
from enum import Enum

MazeType = Enum('MazeType', 'EMPTY LINE SLALOM ANGRY_BIRDS RANDOM')

class MazeGenerator(Maze):
     
    def __init__(self, n: int):
        super().__init__(np.zeros((n, n), dtype=bool))
        self.type = MazeType.EMPTY
    
    def lines(self):
        self.type = MazeType.LINE
        r, c = self.map.shape
        line_spacing = np.random.randint(2, r // 3)
        for i in range(1, r - 1, line_spacing):
            for j in range(c):
                if not self.map[i, j]:
                    self.map[i, j] = True
            rand = np.random.randint(0, c)
            self.map[i, rand] = False

    def slalom(self):
        self.type = MazeType.SLALOM
        r, c = self.map.shape
        self.map[:, :] = False
        w = c // 10
        i = c // 4
        self.map[i-w:i+w, 0:-i] = True
        self.map[-i-w:-i+w, i:] = True

        self.map[0, 0] = False
        self.map[-1,-1] = False

    def angry_birds(self, num_columns: int):
        self.type = MazeType.ANGRY_BIRDS
        r, c = self.map.shape
        self.map[:, :] = False

        # maximum number of columns that can fit within the maze dimensions
        max_columns = (c - 1) // 4 
        if num_columns > max_columns:
            num_columns = max_columns
            print(f"Number of columns adjusted to fit the maze: {num_columns}")

        spacing = c // (num_columns + 1)
        w = max(1, spacing // 4) 

        for k in range(1, num_columns + 1):
            col = k * spacing
            if col - w < 0 or col + w >= c:
                continue 
            if k % 2 == 1:
                self.map[0:r-2, col-w:col+w] = True
            else:
                self.map[2:r, col-w:col+w] = True

        self.map[0, 0] = False
        self.map[-1, -1] = False
        
    def randomize(self, p: float = 0.05):
        if self.type == MazeType.EMPTY:
            while True:
                temp_map = np.random.choice([True, False], size=self.map.shape, p=[p, 1 - p])
                temp_map[self.map] = True
                temp_map[0, 0] = False
                temp_map[-1, -1] = False
                self.map = temp_map
                path = self.dijkstra(self.incidence_matrix())
                if len(path) > 0:
                    break
            for i in path:
                r, c = divmod(i, self.map.shape[1])
                self.map[r, c] = False
        else:
            temp_map = np.random.choice([True, False], size=self.map.shape, p=[p, 1 - p])
            temp_map[self.map] = True
            temp_map[0, 0] = False
            temp_map[-1, -1] = False
            self.map = temp_map

        self.type = MazeType.RANDOM