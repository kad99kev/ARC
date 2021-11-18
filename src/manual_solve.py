#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

from sklearn import neighbors

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
# def solve_c0f76784(x):
#     """
#     There are holes in the shape of a square.
#     The biggest square is filled with light blue.
#     The medium sized square is filled with orange.
#     The small square is filled with pink.
#     """

#     def get_size(x, row_start, column_start):
#         """
#         Helper function to get size of the hole.
#         This will help us to assign the respective colour to the hole.
#         """
#         rows, columns = x.shape
#         row_end, column_end = row_start, column_start
#         size = 0
#         while (
#             (row_end < rows - 1)
#             and (column_end < columns - 1)
#             and x[row_end + 1, column_end + 1] == 0
#         ):
#             row_end += 1
#             column_end += 1
#             size += 1
#         return (size, row_start, column_start)

#     def visited(row_start, column_start, history):
#         """
#         Helper function to check if we already visited.
#         This will help us to avoid searching within the same hole.
#         """
#         for hist in history:
#             if (
#                 row_start >= hist[1]
#                 and row_start <= hist[1] + (hist[0] + 2)
#                 and column_start >= hist[2]
#                 and column_start <= hist[2] + (hist[0] + 2)
#             ):
#                 return True
#         return False

#     def fill(x, history):
#         """
#         Helper function to fill holes based on size.
#         This will fill each identified hole based on its size.
#         """
#         colours = {3: 8, 2: 7, 1: 6}  # Map colours based on size.
#         for hist in history:
#             size, row_start, column_start = hist
#             for i in range(row_start, row_start + size):
#                 for j in range(column_start, column_start + size):
#                     # The +1 is to compensate for the shift
#                     # without this, the borders would be coloured.
#                     x[i + 1][j + 1] = colours[size]
#         return x

#     x = x.copy()  # Prevents overwriting input array.

#     rows, columns = x.shape
#     history = []
#     for i in range(rows):
#         # If the row is zero, don't search.
#         # This means that there are no hole boundaries.
#         if x[i, :].sum() == 0:
#             continue
#         for j in range(1, columns - 1):
#             if not visited(i, j, history):
#                 # Check if there is a change in value.
#                 if x[i][j] != x[i][j - 1]:
#                     # If there is a change in value
#                     # then check diagonally.
#                     history.append(get_size(x, i, j))

#     filled_x = fill(x, history)
#     return filled_x


# def solve_ce9e57f2(x):
#     """
#     The bottom half of the bars need to be changed to blue.
#     """
#     x = x.copy()  # Prevents overwriting input array.
#     rows, columns = x.shape
#     for i in range(columns):
#         non_zero_x = x > 0
#         col_sum = non_zero_x[:, i].sum()
#         # Continue if column is empty.
#         if col_sum == 0:
#             continue

#         # Get the number of bottom blocks to be coloured.
#         bottom_length = col_sum // 2
#         # Colour the bottom blocks.
#         for j in range(bottom_length):
#             x[rows - j - 1][i] = 8

#     return x


# def solve_1e32b0e9(x):
#     """
#     Identify the pattern and fix holes based on it.
#     The holes are of size 5x5, with a pattern (full or half full) of a particular colour.
#     They are separated by a lines of different colour.
#     The pattern in empty or half filled holes must be coloured by this colour.
#     """

#     def fill(x, rows, columns, iter_row, iter_col, pattern, ref_colour, fill_colour):
#         """
#         Helper function to fill pattern for a particular offset.
#         """
#         for pat in pattern:
#             # Get the offset co-ordinates of the row and column for current iteration.
#             off_row, off_col = iter_row * rows + iter_row, iter_col * columns + iter_col
#             # Using offset get pattern co-ordinates.
#             pat_row, pat_col = off_row + pat[0], off_col + pat[1]
#             # Check if reference colour is present for the pattern.
#             # If it is missing then add fill colour.
#             if x[pat_row][pat_col] != ref_colour:
#                 x[pat_row][pat_col] = fill_colour
#         return x

#     x = x.copy()  # Prevents overwriting input array.

#     rows, columns = 5, 5 # Assumption based on patterns seen in training images.
#     # Get original pattern.
#     pattern = []
#     ref_colour = None
#     for i in range(rows):
#         for j in range(columns):
#             if x[i][j] != 0:
#                 pattern.append((i, j))
#                 if ref_colour is None:
#                     ref_colour = x[i][j]  # Store reference colour.

#     # Fill in pattern at missing places
#     fill_colour = x[0][columns]  # Get fill colour from border.
#     for i in range(3):
#         for j in range(3):
#             # First hole is reference pattern.
#             # needn't check
#             if i == 0 and j == 0:
#                 continue
#             # For other holes, fill if required.
#             x = fill(x, rows, columns, i, j, pattern, ref_colour, fill_colour)

#     return x


# def solve_5ad4f10b(x):
#     """
#     Simplfy the pattern.
#     Go through the grid to find the cluster area.
#     Once the cluster area has been identified, \
#         find the simplified pattern.
#     """

#     def extend_cluster(x, curr_row, curr_column, cluster_size):
#         """
#         Helper function to extend a cluster.
#         This tries to extend a cluster to check if a larger cluster exists.
#         """
#         cluster_arr = x[
#             curr_row : curr_row + cluster_size, curr_column : curr_column + cluster_size
#         ]
#         # Run while loop to see if we can increase cluster size.
#         while (
#             np.max(cluster_arr) == np.min(cluster_arr)
#             and cluster_arr.sum() > 0
#             and len(cluster_arr) == cluster_size
#         ):
#             # Extend to see if the cluster_size can be increased.
#             cluster_size += 1
#             cluster_arr = x[
#                 curr_row : curr_row + cluster_size,
#                 curr_column : curr_column + cluster_size,
#             ]
#         cluster_size -= 1
#         return cluster_size

#     def search_cluster(x, curr_row, curr_column, cluster_size):
#         """
#         Helper function to find cluster.
#         This searches for clusters that contain colour in them based on the largest found cluster size.
#         """
#         cluster_arr = x[
#             curr_row : curr_row + cluster_size, curr_column : curr_column + cluster_size
#         ]
#         # Check if current cluster has all elements of same colour.

#         if (
#             np.max(cluster_arr) == np.min(cluster_arr)
#             and cluster_arr.sum() > 0
#             and len(cluster_arr) == cluster_size
#         ):
#             return curr_row, curr_column, x[curr_row, curr_column], True

#         return curr_row, curr_column, 0, False

#     x = x.copy()  # Prevents overwriting input array.

#     rows, columns = x.shape
#     solution = np.zeros((3, 3), dtype=int)  # Simplified pattern placeholder.

#     # Go from left to right, up to down to find the first cluster.
#     # From there, find the left most and top most point
#     # at which a cluster begins.
#     cluster_colour = 0  # Placeholder colour.
#     cluster_size = 3  # Keeping cluster size as threshold initially.
#     left_most, top_most = float("inf"), float("inf")
#     for i in range(rows):
#         for j in range(columns):
#             curr_row, curr_column, curr_colour, found = search_cluster(
#                 x, i, j, cluster_size
#             )
#             # Perform checks to see
#             # if current data needs to be updated.
#             if found:
#                 if curr_row < left_most:
#                     left_most = curr_row
#                 if curr_column < top_most:
#                     top_most = curr_column
#                 # Assign cluster colour.
#                 cluster_colour = curr_colour
#                 # Try and see if cluster can be extended.
#                 new_size = extend_cluster(x, curr_row, curr_column, cluster_size)
#                 if new_size > cluster_size:
#                     cluster_size = new_size

#     # Filter out the cluster and background colours to obtain noise.
#     noise_colour = x[(x != cluster_colour) & (x > 0)][0]
#     # Once left and top most are found, start running cluster check
#     # on the isolated area.
#     row = 0  # To mark row of solution
#     for i in range(left_most, left_most + 3 * cluster_size, cluster_size):
#         col = 0  # To mark column of solution
#         for j in range(top_most, top_most + 3 * cluster_size, cluster_size):
#             # Go through the isolated area and
#             # mark the solution if cluster exists.
#             curr_row, curr_column, _, found = search_cluster(x, i, j, cluster_size)
#             if found:
#                 solution[row, col] = noise_colour
#             col += 1
#         row += 1
#     return solution


"""
Questions:
1. Is it okay to assume certain features or is it considered hardcoded?
2. Does it have to be optimised?
"""


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [
        np.array(data["train"][i]["input"]) for i in range(len(data["train"]))
    ]
    train_output = [
        np.array(data["train"][i]["output"]) for i in range(len(data["train"]))
    ]
    test_input = [np.array(data["test"][i]["input"]) for i in range(len(data["test"]))]
    test_output = [
        np.array(data["test"][i]["output"]) for i in range(len(data["test"]))
    ]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__":
    main()
