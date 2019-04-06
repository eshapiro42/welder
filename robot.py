import random
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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


class Movement():
    '''The actual movement of the robotic arm.

    Args:
        path (Path): Expected welding path.
        standard_error (float): Standard deviation of the arm's error (which is normally distributed).
        delay (float): Time in seconds between evaluations.

    Attributes:
        path (Path): Expected welding path.
        standard_error (float): Standard deviation of the arm's error (which is normally distributed).
        delay (float): Time in seconds between evaluations.  
        current_idx (int): Current index in the path array.
        control (float): The suggested adjustment by the control loop.
        halfway (bool): Whether the job is at least halfway completed.
        done (bool): Whether the job is completed.     
        x (np.array): The x-coordinates of the arm.
        y (np.array): The y-coordinates of the arm.
    '''
    def __init__(self, path, standard_error, delay):
        self.path = path
        self.standard_error = standard_error
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
        self.error_y = random.gauss(0, self.standard_error)
        self.y[self.current_idx] = self.y[self.current_idx - 1] + self.diff_y + self.error_y + self.control

    def start(self):
        '''Begin calculating the movement in a loop'''
        start_time = time.time()
        while not self.done:
            if time.time() - start_time > self.delay:
                self.next()
                start_time = time.time()
            time.sleep(self.delay / 100)


class Sensor():
    '''The sensor which determines the current error.
    
    Args:
        path (Path): Expected welding path.
        movement (Movement): The actual movement of the robotic arm.
        delay (float): Time in seconds between evaluations.

    Attributes:
        path (Path): Expected welding path.
        movement (Movement): The actual movement of the robotic arm.
        delay (float): Time in seconds between evaluations.
        error (float): Current error.
    '''
    def __init__(self, path, movement, delay):
        self.path = path
        self.movement = movement
        self.delay = delay
        self.error = 0

    def sense(self):
        '''Calculate the current error of the arm'''
        current_y = self.movement.y[self.movement.current_idx]
        expected_y = self.path.y[self.movement.current_idx]
        self.error = current_y - expected_y

    def start(self):
        '''Begin sensing the error in a loop'''
        start_time = time.time()
        while not self.movement.done:
            if time.time() - start_time > self.delay:
                self.sense()
            time.sleep(self.delay / 100)


class Controller():
    '''The controller.

    Args:
        sensor (Sensor): The sensor which determines the current error.
        movement (Movement): Actual movement of the robotic arm.
        delay (float): Time in seconds between evaluations.

    Attributes:
        sensor (Sensor): The sensor which determines the current error.
        movement (Movement): Actual movement of the robotic arm.
        delay (float): Time in seconds between evaluations.
        on (bool): Whether the control loop is on.
    '''
    def __init__(self, sensor, movement, delay):
        self.sensor = sensor
        self.movement = movement
        self.delay = delay
        self.on = False

    def toggle(self):
        if self.on:
            self.on = False
        else:
            self.on = True

    def start(self):
        '''Start the control loop'''
        integral = 0
        last_error = 0
        Kp = .5
        Ki = .5
        start_time = time.time()
        while not self.movement.done:
            if time.time() - start_time > self.delay:
                if self.on:
                    error = self.sensor.error
                    integral += error * self.delay
                    output = (error * Kp) + (integral * Ki)
                    last_error = error
                    self.movement.control = -output
                else:
                    self.movement.control = 0
                start_time = time.time()
            time.sleep(self.delay / 100)


class Robot():
    '''The robot.

    Args:
        num (int): Number of points along the path.
        standard_error (float): Standard deviation of the arm's error (which is normally distributed).
        freq (float): The frequency with which the arm should move.

    Attributes:
        delay (float): Time in seconds between evaluations.
        path (Path): Expected welding path.
        movement (Movement): The actual movement of the robotic arm.
        sensor (Sensor): The sensor which determines the current error.
        controller (Controller): The controller.
    '''
    def __init__(self, num, standard_error, freq):
        self.delay = 1 / freq
        self.path = Path(num=num, slope=.1)
        self.movement = Movement(path=self.path, standard_error=standard_error, delay=self.delay)
        self.sensor = Sensor(path=self.path, movement=self.movement, delay=self.delay)
        self.controller = Controller(sensor=self.sensor, movement=self.movement, delay=self.delay)

    def plot(self):
        '''Plot the current state'''
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(self.path.x, self.path.y, c='black')
        ax2.set_xlim(0, 100)

        def controller_toggle(event):
            self.controller.toggle()
            controller_btn.label.set_text('Turn control loop {}'.format('off' if self.controller.on else 'on'))

        controller_btn = Button(ax3, 'Turn control loop on') 
        controller_btn.on_clicked(controller_toggle)

        while not self.movement.done:
            idx = self.movement.current_idx
            ax1.scatter(self.movement.x[idx], self.movement.y[idx], c='blue')
            ax2.scatter(self.movement.x[idx], abs(self.sensor.error), c='red')
            plt.pause(self.delay)
            time.sleep(self.delay / 100)
        plt.show()

    def start(self):
        '''Start all of the components'''
        movement_thread = threading.Thread(target=self.movement.start)
        sensor_thread = threading.Thread(target=self.sensor.start)
        controller_thread = threading.Thread(target=self.controller.start)

        movement_thread.start()
        sensor_thread.start()
        controller_thread.start()

        self.plot()

        movement_thread.join()
        sensor_thread.join()
        controller_thread.join()


if __name__ == '__main__':
    robot = Robot(num=1000, standard_error=.05, freq=50)
    robot.start()
