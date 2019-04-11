import random
import time

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
            time.sleep(.001)