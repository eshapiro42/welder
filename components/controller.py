import time

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
            time.sleep(.001)
        if self.total_iterations > 0:
            self.mean_square_error = self.sum_errors_squared / self.total_iterations
