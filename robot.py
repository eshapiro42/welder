import argparse
import hashlib
import json
import random
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from components import (
    Path,
    Calibration,
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
        self.dataframe = pd.DataFrame(columns=['timestamp', 'x', 'y'])
        self.delay = 1 / freq
        self.path = Path(num=num, slope=.1)
        self.movement = Movement(path=self.path, dataframe=self.dataframe, movement_error=movement_error, delay=self.delay)
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
