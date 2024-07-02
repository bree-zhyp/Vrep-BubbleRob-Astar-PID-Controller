from copy import deepcopy


def printpaths(path, newpath):
    for old, new in zip(path, newpath):
        print('[' + ', '.join('%.3f' % x for x in old) +
              '] -> [' + ', '.join('%.3f' % x for x in new) + ']')


# def smooth(path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
def smooth(path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
    """
    Creates a smooth path for a n-dimensional series of coordinates.
    Arguments:
        path: List containing coordinates of a path
        weight_data: Float, how much weight to update the data (alpha)
        weight_smooth: Float, how much weight to smooth the coordinates (beta).
        tolerance: Float, how much change per iteration is necessary to keep iterating.
    Output:
        new: List containing smoothed coordinates.
    """

    new = deepcopy(path)
    dims = len(path[0])
    change = tolerance

    while change >= tolerance:
        change = 0.0
        for i in range(1, len(new) - 1):
            for j in range(dims):

                x_i = path[i][j]
                y_i, y_prev, y_next = new[i][j], new[i - 1][j], new[i + 1][j]

                y_i_saved = y_i
                y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
                new[i][j] = y_i

                change += abs(y_i - y_i_saved)

    return new

if __name__ == "__main__":
    path = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4]]

    printpaths(path, smooth(path))