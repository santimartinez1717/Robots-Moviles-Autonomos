import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, dt, sigma_v, sigma_w, sigma_z):
        """Initializes the EKF for robot localization.

        Args:
            dt: Time step [s].
            sigma_v: Standard deviation of linear velocity noise.
            sigma_w: Standard deviation of angular velocity noise.
            sigma_z: Standard deviation of sensor measurements.
        """
        self.dt = dt
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.sigma_z = sigma_z
        
        # State vector [x, y, theta]
        self.x = np.zeros((3, 1))
        
        # Covariance matrix
        self.P = np.eye(3) * 0.1
        
        # Process noise covariance
        self.Q = np.diag([sigma_v**2, sigma_v**2, sigma_w**2])
        
        # Measurement noise covariance
        self.R = np.diag([sigma_z**2, sigma_z**2])
    

    @property
    def pose(self):
        """Returns the current estimated pose (x, y, theta)."""
        return self.x[0, 0], self.x[1, 0], self.x[2, 0]
    
    def initialize(self, x0, y0, theta0):   
        """Initializes the state with the given pose.

        Args:
            x0: Initial x position.
            y0: Initial y position.
            theta0: Initial orientation.
        """
        self.x[0, 0] = x0
        self.x[1, 0] = y0
        self.x[2, 0] = theta0



    def predict(self, v, w):
        """Predicts the next state using the motion model.
        PREDICTION STEP
        Args: (robot commands)
            v: Linear velocity.
            w: Angular velocity.
        """
        theta = self.x[2, 0]
        
        if np.abs(w) < 1e-5: # Straight line motion
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
            dtheta = 0
        else: # Circular motion
            dx = (v / w) * (np.sin(theta + w * self.dt) - np.sin(theta))
            dy = (v / w) * (-np.cos(theta + w * self.dt) + np.cos(theta))
            dtheta = w * self.dt
        
        self.x += np.array([[dx], [dy], [dtheta]])
        self.x[2, 0] = (self.x[2, 0] + np.pi) % (2 * np.pi) - np.pi  # Normalize theta
        
        # Compute Jacobian of the motion model
        F = np.array([
            [1, 0, -dy],
            [0, 1, dx],
            [0, 0, 1]
        ])
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, landmark):
        """Updates the state using the sensor measurement.
        MEASUREMENT (CORRECTION) STEP
        
        Args:
            z: Sensor measurement [range, bearing].
            landmark: Landmark position [x, y].
        """
        lx, ly = landmark
        
        # Expected measurement
        dx = lx - self.x[0, 0] # (lx-x)
        dy = ly - self.x[1, 0] # (ly-y)
        r_hat = np.sqrt(dx**2 + dy**2)
        phi_hat = np.arctan2(dy, dx) - self.x[2, 0]
        
        h = np.array([[r_hat], [phi_hat]])
        
        # Measurement Jacobian
        H = np.array([
            [-dx / r_hat, -dy / r_hat, 0],
            [dy / (r_hat**2), -dx / (r_hat**2), -1]
        ])
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        y = z.reshape(2, 1) - h
        y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
        self.x += K @ y
        self.x[2, 0] = (self.x[2, 0] + np.pi) % (2 * np.pi) - np.pi  # Normalize theta
        
        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P
