import numpy as np

class Path():
    '''The expected welding path.

    Args:
        num (int): Number of points along the path.

    Attributes
        num (int): Number of points along the path.
        x (np.array): NumPy array of linearly spaced x-coordinates.
        y (np.array): NumPy array of corresponding y-coordinates.
    '''
    def __init__(self, slope, num):
        self.num = num
        self.x = np.linspace(0, 100, num=self.num)
        self.y = slope * self.x