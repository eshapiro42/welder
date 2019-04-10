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
        mean_square_error (float): The total mean square error over the entire path.
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
        self.mean_square_error = 0

    def toggle(self):
        '''Change the state of the controller'''
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
                    self.total_iterations += 1
                    self.sum_errors_squared += error**2
                    self.movement.control = -output
                else:
                    self.movement.control = 0
                start_time = time.time()
            time.sleep(self.delay / 100)
        if self.total_iterations > 0:
            self.mean_square_error = self.sum_errors_squared / self.total_iterations


class Calibration():
    def __init__(self):
        # Read the config file
        with open('config.json', 'rb') as config_file:
            config = json.load(config_file)
        # Compute the config file checksum
        self.config_checksum = hashlib.sha256(open('config.json', 'rb').read()).hexdigest()
        # Get values from config
        self.num = config['num']
        self.movement_error = config['movement_error']
        self.sensor_error = config['sensor_error']
        self.freq = config['freq']
        # Gradient descent hyperparameters
        self.cost_samples = 10
        self.delta_h = .05
        self.learning_rate = .1
        self.precision = .05
        self.max_it = 20
        self.max_step = 1

    def cost(self, Kp, Ki):
        '''Estimate the cost function for given values of Kp and Ki using <cost_samples> samples'''
        cost_sum = 0
        for sample in range(self.cost_samples):
            robot = Robot(self.num, self.movement_error, self.sensor_error, self.freq, Kp, Ki)
            robot.controller.toggle()
            robot.start(interactive=False)
            cost = robot.controller.mean_square_error
            cost_sum += cost
        avg_cost = cost_sum / self.cost_samples
        return avg_cost

    def gradient_Kp(self, Kp, Ki):
        '''Estimate Kp component of cost function's gradient at the point (Kp, Ki)'''
        cost1_Kp = self.cost(Kp - self.delta_h, Ki)
        cost2_Kp = self.cost(Kp + self.delta_h, Ki)
        gradient_Kp = cost2_Kp - cost1_Kp / (2 * self.delta_h)
        return gradient_Kp

    def gradient_Ki(self, Kp, Ki):
        '''Estimate Ki component of cost function's gradient at the point (Kp, Ki)'''
        cost1_Ki = self.cost(Kp, Ki - self.delta_h)
        cost2_Ki = self.cost(Kp, Ki + self.delta_h)
        gradient_Ki = cost2_Ki - cost1_Ki / (2 * self.delta_h)
        return gradient_Ki

    def gradient_descent_Kp(self):
        '''Tune Kp using gradient descent and Ki=0'''
        print('Tuning Kp...')
        Kp = 1
        Kp_step = 1
        it = 0
        while abs(Kp_step) > self.precision and it < self.max_it:
            Kp_prev = Kp
            gradient_Kp = self.gradient_Kp(Kp, Ki=0)
            Kp_step = self.learning_rate * gradient_Kp
            # step magnitudes should not exceed max_step
            if abs(Kp_step) > self.max_step:
                Kp_step = Kp_step / abs(Kp_step) * self.max_step
            # recalculate Kp
            Kp = Kp - Kp_step
            # Kp cannot be less than zero
            if Kp < 0:
                Kp = 0
            print('it={}, Kp={}, delta_Kp={}'.format(it, Kp, -Kp_step))
            it += 1
        return Kp
        
    def gradient_descent_Ki(self, Kp):
        '''Tune Ki using gradient descent and fixed Kp'''
        print('Tuning Ki...')
        Ki = 1
        Ki_step = 1
        it = 0
        while abs(Ki_step) > self.precision and it < self.max_it:
            Ko_prev = Ki
            gradient_Ki = self.gradient_Ki(Kp, Ki)
            Ki_step = self.learning_rate * gradient_Ki
            # step magnitudes should not exceed max_step
            if abs(Ki_step) > self.max_step:
                Ki_step = Ki_step / abs(Ki_step) * self.max_step
            # recalculate Ki
            Ki = Ki - Ki_step
            # Ki cannot be less than zero
            if Ki < 0:
                Ki = 0            
            print('it={}, Ki={}, delta_Ki={}'.format(it, Ki, -Ki_step))
            it += 1
        return Ki

    def tune(self):
        '''Tune Kp first and then Ki using gradient descent'''
        Kp = self.gradient_descent_Kp()
        Ki = self.gradient_descent_Ki(Kp)
        return Kp, Ki

    def calibrate(self):
        '''Perform the calibration'''
        # Tune Kp and Ki
        Kp, Ki = self.tune()
        # Create the calibration dictionary
        cal = {
            'config_checksum': self.config_checksum,
            'Kp': Kp,
            'Ki': Ki,
        }
        # Save the calibration data
        with open('cal.json', 'w') as cal_file:
            json.dump(cal, cal_file)


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
