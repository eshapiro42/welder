import argparse
import hashlib
import json
import threading
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from components import (
    Path,
    Controller,
    Sensor,
    Movement,
)


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
        ax1.set_title('Actual Path vs Expected Path')
        ax2.set_xlim(0, 100)
        ax2.set_title('Sensor Data')
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
