import numpy as np
import random
import time

class Movement():
    '''The actual movement of the robotic arm.

    Args:
        path (Path): Expected welding path.
        movement_error (float): Standard deviation of the arm's error (which is normally distributed).
        delay (float): Time in seconds between evaluations.

    Attributes:
        path (Path): Expected welding path.
        dataframe (pd.DataFrame): Time series database.
        movement_error (float): Standard deviation of the arm's error (which is normally distributed).
        delay (float): Time in seconds between evaluations.
        current_idx (int): Current index in the path array.
        control (float): The suggested adjustment by the control loop.
        halfway (bool): Whether the job is at least halfway completed.
        done (bool): Whether the job is completed.
        x (np.array): The x-coordinates of the arm.
        y (np.array): The y-coordinates of the arm.
    '''
    def __init__(self, path, dataframe, movement_error, delay):
        self.path = path
        self.movement_error = movement_error
        self.delay = delay
        self.current_idx = 0
        self.control = 0
        self.halfway = False
        self.done = False
        self.x = self.path.x
        self.y = np.zeros(path.num)
        self.y[0] = self.path.y[0]

    def next(self):
        '''Calculate the next coordinates to drop the arm'''
        self.current_idx += 1
        if self.current_idx >= self.path.num:
            self.done = True
            return
        if self.current_idx == self.path.num // 2:
            self.halfway = True
        self.diff_y = self.path.y[self.current_idx] - self.path.y[self.current_idx - 1]
        self.error_y = random.gauss(0, self.movement_error)
        self.y[self.current_idx] = self.y[self.current_idx - 1] + self.diff_y + self.error_y + self.control

    def start(self):
        '''Begin calculating the movement in a loop'''
        start_time = time.time()
        while not self.done:
            if time.time() - start_time > self.delay:
                self.next()
                start_time = time.time()
            time.sleep(.001)