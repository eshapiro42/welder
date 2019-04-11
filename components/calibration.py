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