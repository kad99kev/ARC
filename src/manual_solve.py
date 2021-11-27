#!/usr/bin/python

"""
Written by: Kevlyn Kadamala
Student ID: 21236191
Class: MSc AI
GitHub Repository: https://github.com/kad99kev/ARC

Summary:
- Compared to other programming languages, writing code in Python is easy.
- The code looks "cleaner" since Python supports dynamic typing.
- The library that I used the most in the solve functions is NumPy \
    since I had a lot of array manipulations to carry out.
- Some of the main commonalities with regards to the solve functions are:
    - Problems involve searching for patterns.
    - Problems involve keeping track of data that we have seen before.
    - Problems have certain conditions that determine the output.
"""


import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_c0f76784(x):
    """
    Required Transformations:
        - There are holes in the shape of a square.
        - The biggest square is filled with light blue.
        - The medium sized square is filled with orange.
        - The small square is filled with pink.

    How solve works:
        - In each row we search for hole boundaries, if none exist we skip to the next row.
        - If we find a hole boundary, we check to see if we already visited that hole.
        - If we haven't visited the hole before, we try to find the size of the hole.
        - Once we have the size and the top left corner of the hole, we fill it based on its size.
        - The biggest hole is filled with blue, medium with orange and smallest with pink.

    Grids solved:
        - All grids have been solved.
    """

    def get_size(x, row_start, column_start):
        """
        Helper function to get size of the hole.
        This will help us to assign the respective colour to the hole.

        Arguments:
            x: The input grid.
            row_start: Starting index for the current hole.
            column_start: Starting index for the current hole.
        """
        rows, columns = x.shape
        row_end, column_end = row_start, column_start
        size = 0
        # Iterate diagonally to see how far the hole boundaries can be pushed.
        while (
            (row_end < rows - 1)
            and (column_end < columns - 1)
            and x[row_end + 1, column_end + 1] == 0
        ):
            row_end += 1
            column_end += 1
            size += 1
        return (size, row_start, column_start)

    def visited(row_start, column_start, history):
        """
        Helper function to check if we already visited.
        This will help us to avoid searching within the same hole.

        Arguments:
            row_start: Starting index for the current row.
            column_start: Starting index for the current column.
            history: Current visited hole history.
        """
        for hist in history:
            if (
                row_start >= hist[1]
                and row_start <= hist[1] + (hist[0] + 2)
                and column_start >= hist[2]
                and column_start <= hist[2] + (hist[0] + 2)
            ):
                return True
        return False

    def fill(x, history):
        """
        Helper function to fill holes based on size.
        This will fill each identified hole based on its size.

        Arugments:
            x: The input grid.
            history: Locations and sizes of holes in the grid.
        """
        colours = {3: 8, 2: 7, 1: 6}  # Map colours based on size.
        for hist in history:
            size, row_start, column_start = hist
            for i in range(row_start, row_start + size):
                for j in range(column_start, column_start + size):
                    # The +1 is to compensate for the shift
                    # without this, the borders would be coloured.
                    x[i + 1][j + 1] = colours[size]
        return x

    x = x.copy()  # Prevents overwriting input array.

    rows, columns = x.shape
    history = []
    for i in range(rows):
        # If the row is zero, don't search.
        # This means that there are no hole boundaries.
        if x[i, :].sum() == 0:
            continue
        for j in range(1, columns - 1):
            if not visited(i, j, history):
                # Check if there is a change in value.
                if x[i][j] != x[i][j - 1]:
                    # If there is a change in value
                    # then check diagonally.
                    history.append(get_size(x, i, j))

    filled_x = fill(x, history)
    return filled_x


def solve_ce9e57f2(x):
    """
    Required Transformations:
        - The bottom half of the bars need to be changed to blue.

    How solve works:
        - For each column we check to see if there are any bars.
        - If there are any bars, we colour the bottom half from red to blue.

    Grids solved:
        - All grids have been solved.
    """
    x = x.copy()  # Prevents overwriting input array.
    rows, columns = x.shape
    for i in range(columns):
        non_zero_x = x > 0
        col_sum = non_zero_x[:, i].sum()
        # Continue if column is empty.
        if col_sum == 0:
            continue

        # Get the number of bottom blocks to be coloured.
        bottom_length = col_sum // 2
        # Colour the bottom blocks.
        for j in range(bottom_length):
            x[rows - j - 1][i] = 8

    return x


def solve_1e32b0e9(x):
    """
    Required Transformations:
        - The entire grid has been divided into 9 cells with 3 cells in each row.
        - The top left cell is the reference cell containing the main pattern.
        - The remaining cells have patterns that are either fully filled, partial filled or completely empty.
        - We have to fill in the missing pattern in each cell with the colour of the cell borders.

    How solve works:
        - First we extract the reference pattern from the top leftmost cell.
        - While extracting the pattern we need to store the reference pattern colour.
        - We also need to store the cell border colour.
        - Once we have our reference pattern, pattern colour and border colour, \
            we iterate through each cell to fill in the missing pattern.

    Grids solved:
        - All grids have been solved.
    """

    def fill(x, rows, columns, iter_row, iter_col, pattern, ref_colour, fill_colour):
        """
        Helper function to fill pattern for a particular offset.
        Since, the reference pattern is stored with respect to the co-ordinates of the reference cell, \
            when searching for patterns in other cells we calculate the offset and add the pattern co-ordinates.

        Arguments:
            x: The input grid.
            rows: The numbers of rows in each cell.
            columns: The numbers of columns in each cell.
            iter_row: Current row iteration.
            iter_col: Current column iteration.
            pattern: The reference pattern.
            ref_colour: The colour with which the current cell pattern will be searched (reference pattern colour).
            fill_colour: The colour with which the missing pattern will be filled (cell border colour.)
        """
        for pat in pattern:
            # Get the offset co-ordinates of the row and column for current iteration.
            off_row, off_col = iter_row * rows + iter_row, iter_col * columns + iter_col
            # Using offset get pattern co-ordinates.
            pat_row, pat_col = off_row + pat[0], off_col + pat[1]
            # Check if reference colour is present for the pattern.
            # If it is missing then add fill colour.
            if x[pat_row][pat_col] != ref_colour:
                x[pat_row][pat_col] = fill_colour
        return x

    x = x.copy()  # Prevents overwriting input array.

    rows, columns = 5, 5  # Assumption based on patterns seen in training images.
    # Get original pattern.
    pattern = []
    ref_colour = None
    for i in range(rows):
        for j in range(columns):
            if x[i][j] != 0:
                pattern.append((i, j))
                if ref_colour is None:
                    ref_colour = x[i][j]  # Store reference colour.

    # Fill in pattern at missing places
    fill_colour = x[0][columns]  # Get fill colour from border.
    for i in range(3):
        for j in range(3):
            # First hole is reference pattern.
            # needn't check
            if i == 0 and j == 0:
                continue
            # For other holes, fill if required.
            x = fill(x, rows, columns, i, j, pattern, ref_colour, fill_colour)

    return x


def solve_5ad4f10b(x):
    """
    Required Transformations:
        - We have to first find and identify the pattern from the entire grid.
        - Each pattern is made out of clusters.
        - While the cluster size may vary from problem to problem, \
            the entire pattern will consist of clusters of the same size.
        - Once the pattern has been identified, we have to provide a simplied version of the pattern.
        - The simplified version of the pattern is filled with the colour of the noise from the input grid.

    How solve works:
        - First we iterate through the grid to find a cluster.
        - For each cluster we find, we try to identify the topmost and left most cluster co-ordinates.
        - We also identify the cluster colour.
        - For every cluster that we find we try to extend it in order to find its actual size.
        - Once the cluster size, the topmost and leftmost co-ordinates have been found, \
            we create the simplified pattern.

    Grids solved:
        - All grids have been solved.
    """

    def extend_cluster(x, curr_row, curr_column, cluster_size):
        """
        Helper function to extend a cluster.
        This tries to extend a cluster to check if a larger cluster exists.

        Arguments:
            x: The input grid.
            curr_row: The current row iteration.
            curr_column: The current column iteration.
            cluster_size: The current cluster size.
        """
        cluster_arr = x[
            curr_row : curr_row + cluster_size, curr_column : curr_column + cluster_size
        ]
        # Run while loop to see if we can increase cluster size.
        while (
            np.max(cluster_arr) == np.min(cluster_arr)
            and cluster_arr.sum() > 0
            and len(cluster_arr) == cluster_size
        ):
            # Extend to see if the cluster_size can be increased.
            cluster_size += 1
            cluster_arr = x[
                curr_row : curr_row + cluster_size,
                curr_column : curr_column + cluster_size,
            ]
        cluster_size -= 1
        return cluster_size

    def search_cluster(x, curr_row, curr_column, cluster_size):
        """
        Helper function to find cluster.
        This searches for clusters that contain colour in them based on the largest found cluster size.

        Arguments:
            x: The input grid.
            curr_row: The current row iteration.
            curr_column: The current column iteration.
            cluster_size: The current cluster size.
        """
        cluster_arr = x[
            curr_row : curr_row + cluster_size, curr_column : curr_column + cluster_size
        ]
        # Check if current cluster has all elements of same colour.

        if (
            np.max(cluster_arr) == np.min(cluster_arr)
            and cluster_arr.sum() > 0
            and len(cluster_arr) == cluster_size
        ):
            return x[curr_row, curr_column], True

        return 0, False

    x = x.copy()  # Prevents overwriting input array.

    rows, columns = x.shape
    solution = np.zeros((3, 3), dtype=int)  # Simplified pattern placeholder.

    # Go from left to right, up to down to find the first cluster.
    # From there, find the left most and top most point
    # at which a cluster begins.
    cluster_colour = 0  # Placeholder colour.
    cluster_size = 3  # Keeping cluster size as threshold initially.
    left_most, top_most = float("inf"), float("inf")
    for i in range(rows):
        for j in range(columns):
            curr_colour, found = search_cluster(x, i, j, cluster_size)
            # Perform checks to see
            # if current data needs to be updated.
            if found:
                if i < left_most:
                    left_most = i
                if j < top_most:
                    top_most = j
                # Assign cluster colour.
                cluster_colour = curr_colour
                # Try and see if cluster can be extended.
                new_size = extend_cluster(x, i, j, cluster_size)
                if new_size > cluster_size:
                    cluster_size = new_size

    # Filter out the cluster and background colours to obtain noise.
    noise_colour = x[(x != cluster_colour) & (x > 0)][0]
    # Once left and top most are found, start running cluster check
    # on the isolated area.
    row = 0  # To mark row of solution
    for i in range(left_most, left_most + 3 * cluster_size, cluster_size):
        col = 0  # To mark column of solution
        for j in range(top_most, top_most + 3 * cluster_size, cluster_size):
            # Go through the isolated area and
            # mark the solution if cluster exists.
            _, found = search_cluster(x, i, j, cluster_size)
            if found:
                solution[row, col] = noise_colour
            col += 1
        row += 1
    return solution


def solve_d43fd935(x):
    """
    Required Transformations:
        - Search for the larger green cluster and extend any block that \
            shares the same row or column with the cluster.
        - The extension colour is the same as the colour of the block.

    How solve works:
        - First we try to find a green cluster from the entire grid.
        - Once we have the indices of the green cluster, \
            we find the top leftmost and bottom rightmost co-ordinates.
        - We then extend any block that share the same co-ordinate values as the green cluster.
        - We obtain the color of the block and fill the extension with its colour.

    Grids solved:
        - All grids have been solved.
    """

    def extend_rows(x, curr_row, max_c, min_c, background_colour, block_colour):
        """
        Extend rows based on current row and the minimum and maximum column values.

        Arguments:
            x: The input grid.
            curr_row: The current row iteration.
            max_c: The maximum column co-ordinate.
            min_c: The minimum column co-ordinate.
            background_colour: The colour of the background.
            block_colour: The colour of the block.
        """
        row_data = x[curr_row, :]
        r_idx = np.where((row_data != background_colour) & (row_data != block_colour))[
            0
        ]
        for idx in r_idx:
            colour = x[curr_row, idx]
            if idx < min_c:
                x[curr_row, idx:min_c] = colour
            else:
                x[curr_row, max_c + 1 : idx] = colour
        return x

    def extend_cols(x, curr_col, max_r, min_r, background_colour, block_colour):
        """
        Extend columns based on current column and the minimum and maximum row values.

        Arguments:
            x: The input grid.
            curr_col: The current column iteration.
            max_r: The maximum row co-ordinate.
            min_r: The minimum row co-ordinate.
            background_colour: The colour of the background.
            block_colour: The colour of the block.
        """
        col_data = x[:, curr_col]
        c_idx = np.where((col_data != background_colour) & (col_data != block_colour))[
            0
        ]
        for idx in c_idx:
            colour = x[idx, curr_col]
            if idx < min_r:
                x[idx:min_r, curr_col] = colour
            else:
                x[max_r + 1 : idx, curr_col] = colour
        return x

    x = x.copy()  # Prevents overwriting input array.

    green = 3  # Colour for green
    black = 0  # Colour for black

    rows, columns = np.where(x == green)  # Find indices where green is present.
    max_r, max_c = max(rows), max(columns)  # Getting the bottom rightmost co-ordinate.
    min_r, min_c = max_r - 1, max_c - 1  # Getting the top leftmost co-ordinate.

    # For rows
    x = extend_rows(x, min_r, max_c, min_c, black, green)  # For top row.
    x = extend_rows(x, max_r, max_c, min_c, black, green)  # For bottom row.

    # For columns.
    x = extend_cols(x, min_c, max_r, min_r, black, green)  # For left column.
    x = extend_cols(x, max_c, max_r, min_r, black, green)  # For right column.

    return x


def solve_681b3aeb(x):
    """
    Required Transformations:
        - We have to join the two shapes from the grid to complete a 3x3 grid.

    How solve works:
        - First we try to find the different shapes in the grid.
        - If a row is empty, we skip that and check the next row.
        - Once we find a shape, we save it in a dictionary.
        - The dictionary has the colour of the shape as its key and \
            a list of the co-ordinates of the shape as its value.
        - Once both the shapes have been extracted, we combine them.

    Grids solved:
        - All grids have been solved.
    """

    def convert_cords(cords):
        """
        Convert the co-ordinates of the shape.
        Reduces the shape to a 3x3 grid.

        Arguments:
            cords: The co-ordinates of the shape.
        """
        cords = np.array(cords)
        # Get the minimum and maximum co-ordinate points.
        row_min, col_min = cords.min(axis=0)
        row_max, col_max = cords.max(axis=0)
        cords[:, 0] -= row_min
        cords[:, 1] -= col_min

        col_diff = col_max - col_min
        row_diff = row_max - row_min
        # Check if column difference is 1.
        # That means the rows of the largest shape is not fully filled.
        # Hence, this naturally means that the largest shape will need to be placed
        # on the rightmost corner.
        # That means we will need to shift it to the right
        if col_diff == 1 and row_diff == 2:
            cords[:, 1] += 1
        return cords

    def combine(shape1, shape2):
        """
        Combine the two shapes to form a single 3x3 grid.
        Once we decide the position of the largest shape, \
            we fill the remaining spaces with the colour of the smallest shape.

        Arguments:
            shape1: The first shape.
            shape2: The second shape.
        """
        large_shape = shape1 if len(shape1[1]) > len(shape2[1]) else shape2
        small_shape = shape1 if len(shape1[1]) < len(shape2[1]) else shape2
        solution = np.zeros((3, 3), dtype=int)

        large_cords = convert_cords(large_shape[1])
        for row, col in large_cords:
            # Assign colour to the co-ordinates of the large spaces.
            solution[row, col] = large_shape[0]

        solution[solution == 0] = small_shape[0]  # Assign colour to remaining places.

        return solution

    x = x.copy()  # Prevents overwriting input array.

    shapes = {}  # Stores shapes with colour as key and co-ordinates as values.

    # Find shapes in grid.
    for row_idx, row in enumerate(x):
        # Ignore if row empty.
        if row.sum() == 0:
            continue
        for col_idx in np.where(row != 0)[0]:
            colour = x[row_idx, col_idx]
            if colour not in shapes:
                shapes[colour] = [(row_idx, col_idx)]
            else:
                shapes[colour].append((row_idx, col_idx))

    solution = combine(*shapes.items())

    return solution


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
