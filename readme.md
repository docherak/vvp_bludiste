# Finding the shortest path in a maze (WIP)

## Description

This project deals with mazes - specifically generating and solving them. The input is a maze $n\times n$, where
the start of the maze is the upper left corner and the end
is the bottom right corner. It is only possible to get from one cell to another
through a common edge (not through a corner). The goal of the project is to implement
algorithms for retrieval, shortest path search and maze generation.

This Maze class constructor creates / reads a maze $n\times n$ and stores it in memory as a NumPy matrix with True/False
values. Dijkstra's algorithm is implemented for finding the shortest path. This algorithm is also used when generating
random solvable maze. There are also multiple predefined maze templates available, which can be further randomized.

The output is in the form of an image
- black = impassable part
- white = passable,
- red = shortest path

This is a final project for **Scientific Computing in Python** class at [VSB TUO](https://www.vsb.cz/en).

## Functionalities

- loading a maze from a CSV file or using a generated NumPy matrix
- finding the shortest path from start (top left corner) to end (bottom right corner) via:
  - creating incidence matrix
  - using Dijkstra's algorithm to find the shortest path
- drawing the maze as a black/white picture with a red path if solution is found (otherwise just plotting the original maze)
- generating a set of solvable templates (lines, slalom, "Angry Birds", randomized)
  - each of these templates can be randomized further (not necessarily solvable)
- saving maze as a CSV file
- functionalities are showcased in the `examples.ipynb` file

## Docs (TODO)
See 

## Formatting, linting
I used [ruff](https://github.com/astral-sh/ruff) and [pyright](https://github.com/microsoft/pyright).
