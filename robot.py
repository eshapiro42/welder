import argparse
import hashlib
import json
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
        movement_error (float): Standard deviation of the arm's error (which is normally distributed).
        delay (float): Time in seconds between evaluations.

    Attributes:
        path (Path): Expected welding path.
        movement_error (float): Standard deviation of the arm's error (which is normally distributed).
        delay (float): Time in seconds between evaluations.
        current_idx (int): Current index in the path array.
        control (float): The suggested adjustment by the control loop.
        halfway (bool): Whether the job is at least halfway completed.
        done (bool): Whether the job is completed.
        x (np.array): The x-coordinates of the arm.
        y (np.array): The y-coordinates of the arm.
    '''
    def __init__(self, path, movement_error, delay):
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
            time.sleep(self.delay / 100)


class Sensor():
    '''The sensor which determines the current error.

    Args:
        path (Path): Expected welding path.
        movement (Movement): The actual movement of the robotic arm.
        sensor_error (float): Standard deviation of the sensor's error (which is normally distributed).
        delay (float): Time in seconds between evaluations.

    Attributes:
        path (Path): Expected welding path.
        movement (Movement): The actual movement of the robotic arm.
        delay (float): Time in seconds between evaluations.
        difference (float): Current deviation from expected path.
    '''
    def __init__(self, path, movement, sensor_error, delay):
        self.path = path
        self.movement = movement
        self.sensor_error = sensor_error
        self.delay = delay
        self.difference = 0

    def sense(self):
        '''Calculate the current error of the arm'''
        current_y = self.movement.y[self.movement.current_idx]
        expected_y = self.path.y[self.movement.current_idx]
        error = random.gauss(0, self.sensor_error)
        self.difference = current_y - expected_y + error

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
        Kp (float): Multiplier for the proportional component of the PID.
        Ki (float): Multiplier for the integral component of the PID.

    Attributes:
        sensor (Sensor): The sensor which determines the current error.
        movement (Movement): Actual movement of the robotic arm.
        delay (float): Time in seconds between evaluations.
        Kp (float): Multiplier for the proportional component of the PID.
        Ki (float): Multiplier for the integral component of the PID.
        on (bool): Whether the control loop is on.
    '''
    def __init__(self, sensor, movement, delay, Kp, Ki):
        self.sensor = sensor
        self.movement = movement
        self.delay = delay
        self.Kp = Kp
        self.Ki = Ki
        self.on = False
        self.total_iterations = 0
        self.sum_errors_squared = 0
        self.rms_error = 0

    def toggle(self):
        if self.on:
            self.on = False
        else:
            self.on = True

    def start(self):
        '''Start the control loop'''
        integral = 0
        last_error = 0
        start_time = time.time()
        while not self.movement.done:
            if time.time() - start_time > self.delay:
                if self.on:
                    error = self.sensor.difference
                    integral += error * self.delay
                    output = (error * self.Kp) + (integral * self.Ki)
                    last_error = error
                    self.movement.control = -output
                    self.total_iterations += 1
                    self.sum_errors_squared += error**2
                else:
                    self.movement.control = 0
                start_time = time.time()
            time.sleep(self.delay / 100)
        if self.total_iterations > 0:
            self.rms_error = np.sqrt(self.sum_errors_squared / self.total_iterations)


class Calibration():
    def __init__(self):
    # def __init__(self, num, movement_error, sensor_error):
        # Read the config file
        with open('config.json', 'rb') as config_file:
            config = json.load(config_file)
        # Compute the config file checksum
        self.config_checksum = hashlib.sha256(open('config.json', 'rb').read()).hexdigest()
        self.num = config['num']
        self.movement_error = config['movement_error']
        self.sensor_error = config['sensor_error']
        self.gradient_h = .1
        self.learning_rate = .1
        self.precision = .01
        self.max_it = 50

        # self.Kp_range = np.linspace(0, 1, num=10)
        # self.Ki_range = np.linspace(0, 1, num=10)
        # self.lowest_rms_error = None
        # self.best_Kp = None
        # self.best_Ki = None

    def cost(self, Kp, Ki):
        robot = Robot(self.num, self.movement_error, self.sensor_error, self.num, Kp, Ki)
        robot.controller.toggle()
        robot.start(interactive=False)
        return robot.controller.rms_error

    def gradient(self, Kp, Ki):
        cost1_Kp = self.cost(Kp - self.gradient_h, Ki)
        cost2_Kp = self.cost(Kp + self.gradient_h, Ki)
        gradient_Kp = cost2_Kp - cost1_Kp / (2 * self.gradient_h)
        cost1_Ki = self.cost(Kp, Ki - self.gradient_h)
        cost2_Ki = self.cost(Kp, Ki + self.gradient_h)
        gradient_Ki = cost2_Ki - cost1_Ki / (2 * self.gradient_h)
        return gradient_Kp, gradient_Ki

    def gradient_descent(self):
        Kp = .5
        Ki = .5
        step_size_Kp = 1
        step_size_Ki = 1
        it = 0
        while (step_size_Kp > self.precision or step_size_Ki > self.precision) and it < self.max_it:
            Kp_prev = Kp
            Ki_prev = Ki
            gradient_Kp, gradient_Ki = self.gradient(Kp, Ki)
            Kp = Kp - self.learning_rate * gradient_Kp
            Ki = Ki - self.learning_rate * gradient_Ki
            step_size_Kp = abs(Kp - Kp_prev)
            step_size_Ki = abs(Ki - Ki_prev)
            print('it={}, Kp={}, Ki={}, step_size_Kp={}, step_size_Ki={}'.format(it, Kp, Ki, step_size_Kp, step_size_Ki))
            it += 1
        return Kp, Ki

    def calibrate(self):
        # Tune Kp and Ki
        Kp, Ki = self.gradient_descent()
        # Create the calibration dictionary
        cal = {
            'config_checksum': self.config_checksum,
            'Kp': Kp,
            'Ki': Ki,
        }
        # Save the calibration data
        with open('cal.json', 'w') as cal_file:
            json.dump(cal, cal_file)

    # def calibrate(self):
    #     # Tune Kp and Ki
    #     # TODO This currently sweeps out a huge grid. Switch to gradient descent if there's time.
    #     it = 0
    #     for Kp in self.Kp_range:
    #         for Ki in self.Ki_range:
    #             print('{}%'.format(100 * it / (len(self.Kp_range) * len(self.Ki_range))), end='')
    #             robot = Robot(self.num, self.movement_error, self.sensor_error, self.num, Kp, Ki)
    #             robot.controller.toggle()
    #             robot.start(interactive=False)
    #             if self.lowest_rms_error is None or robot.controller.rms_error < self.lowest_rms_error:
    #                 self.lowest_rms_error = robot.controller.rms_error
    #                 self.best_Kp = Kp
    #                 self.best_Ki = Ki
    #             print(' --- Kp={}, Ki={}'.format(self.best_Kp, self.best_Ki))
    #             it += 1
    #     # Create the calibration dictionary
    #     cal = {
    #         'config_checksum': self.config_checksum,
    #         'Kp': self.best_Kp,
    #         'Ki': self.best_Ki,
    #     }
    #     # Save the calibration data
    #     with open('cal.json', 'w') as cal_file:
    #         json.dump(cal, cal_file)


class Robot():
    '''The robot.

    Args:
        num (int): Number of points along the path.
        movement_error (float): Standard deviation of the arm's error (which is normally distributed).
        sensor_error (float): Standard deviation of the sensor's error (which is normally distributed).
        freq (float): The frequency with which the arm should move.
        Kp (float): Multiplier for the proportional component of the PID.
        Ki (float): Multiplier for the integral component of the PID.

    Attributes:
        delay (float): Time in seconds between evaluations.
        path (Path): Expected welding path.
        movement (Movement): The actual movement of the robotic arm.
        sensor (Sensor): The sensor which determines the current error.
        controller (Controller): The controller.
    '''
    def __init__(self, num, movement_error, sensor_error, freq, Kp, Ki):
        self.delay = 1 / freq
        self.path = Path(num=num, slope=.1)
        self.movement = Movement(path=self.path, movement_error=movement_error, delay=self.delay)
        self.sensor = Sensor(path=self.path, movement=self.movement, sensor_error=sensor_error, delay=self.delay)
        self.controller = Controller(sensor=self.sensor, movement=self.movement, delay=self.delay, Kp=Kp, Ki=Ki)

    def plot(self):
        '''Plot the current state'''
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(self.path.x, self.path.y, c='black')
        ax2.set_xlim(0, 100)
        ax3.set_xlim(0, 100)

        def controller_toggle(event):
            idx = self.movement.current_idx
            line_x = self.movement.x[idx]
            self.controller.toggle()
            ax3.axvline(x=line_x, c=('green' if self.controller.on else 'red'))
            controller_btn.label.set_text('Turn control loop {}'.format('off' if self.controller.on else 'on'))

        controller_btn = Button(ax3, 'Turn control loop on')
        controller_btn.on_clicked(controller_toggle)

        while not self.movement.done:
            # If the window is closed, stop everything
            if not plt.get_fignums():
                self.movement.done = True
                break
            idx = self.movement.current_idx
            ax1.scatter(self.movement.x[idx], self.movement.y[idx], c='blue')
            ax2.scatter(self.movement.x[idx], abs(self.sensor.difference), c='red')
            plt.pause(self.delay)
        plt.show()

    def start(self, interactive=True):
        '''Start all of the components'''
        movement_thread = threading.Thread(target=self.movement.start)
        sensor_thread = threading.Thread(target=self.sensor.start)
        controller_thread = threading.Thread(target=self.controller.start)

        movement_thread.start()
        sensor_thread.start()
        controller_thread.start()

        if interactive:
            self.plot()

        movement_thread.join()
        sensor_thread.join()
        controller_thread.join()


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser()
    run_type = parser.add_mutually_exclusive_group(required=True)
    run_type.add_argument('--run', action='store_true')
    run_type.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()
    if args.run:
        # Read the calibration file
        with open('cal.json', 'rb') as cal_file:
            cal = json.load(cal_file)
        # Check whether the config has changed since the last calibration
        config_checksum = hashlib.sha256(open('config.json', 'rb').read()).hexdigest()
        if config_checksum != cal['config_checksum']:
            proceed = input('Calibration may be out of date. Do you want to continue? [y/N]\n')
            if proceed.lower() != 'y':
                exit()
        # Read the config file
        with open('config.json', 'rb') as config_file:
            config = json.load(config_file)
        robot = Robot(
                    num=config['num'], 
                    movement_error=config['movement_error'],
                    sensor_error=config['sensor_error'], 
                    freq=config['freq'], 
                    Kp=cal['Kp'], 
                    Ki=cal['Ki']
                    )
        robot.start()
    if args.calibrate:
        calibration = Calibration()
        calibration.calibrate()
