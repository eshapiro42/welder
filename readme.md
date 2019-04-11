# Assumptions

1. The path is flat with respect to the z-axis.
2. Errors occur along the y-axis only.
3. Errors will compound unless the control loop is turned on to compensate.

# Setup

This has been tested only on Ubuntu with Python 3.6.

1. Clone this repository and navigate into it.
2. Run ```pip install -r requirements.txt```.
3. Make sure ```config.json``` and ```cal.json``` files are present and update the values as desired.
4. Make sure you have a working X Server to view the output.

# Running

    usage: robot.py [-h] (--run | --calibrate)

The ```--run``` flag will run the machine using the current configuration and calibration data. The controller is off by default, but the giant button at the bottom of the screen will toggle it on/off.

![screenshot](/screenshot.PNG)

The ```--calibrate``` flag will calibrate the controller parameters according to the current configuration data. **This will take a long time, so run it overnight if possible.**
