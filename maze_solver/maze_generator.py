from .maze_solver import Maze
import numpy as np

class MazeGenerator(Maze):
     
    def __init__(self, rows: int, cols: int):
        super().__init__(np.zeros((rows, cols), dtype=bool))
    
    def lines(self):
        r, c = self.map.shape
        self.map[:, :] = False
        for i in range(r):
            if i == 0 or i == r - 1:
                continue
            if i % 3 == 0:
                self.map[i, :] = True
                rand = np.random.randint(0, c)
                self.map[i, rand] = False
                continue
        self.map[0, 0] = False
        self.map[r - 1, c - 1] = False

    
