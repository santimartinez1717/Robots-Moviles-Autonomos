import numpy as np
import math
 
# own modules
from amr_localization.maps import Map
from amr_localization.particle_filter import ParticleFilter as PF

class ExtendedKalmanFilter:
    def __init__(self, dt, sigma_v, sigma_w, sigma_z, map_path):
        """Initializes the EKF for robot localization.

        Args:
            dt: Time step [s].
            sigma_v: Standard deviation of linear velocity noise.
            sigma_w: Standard deviation of angular velocity noise.
            sigma_z: Standard deviation of sensor measurements.
            map_path: Path to the map file.
            sensor_range_max: Maximum sensor range [m].
        """
        sensor_range_max: float = 8.0
        sensor_range_min: float = 0.16
        
        self.dt = dt
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.sigma_z = sigma_z
        self._map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        self.sensor_range_max = sensor_range_max
        self.sensor_range_min = sensor_range_min
        
        # State vector [x, y, theta]
        self.x = np.zeros((3, 1))
        
        # Covariance matrix
        self.P = np.eye(3) * 0.1
        
        # Process noise covariance
        self.Q = np.diag([sigma_v**2, sigma_v**2, sigma_w**2])
        
        # Measurement noise covariance
        self.R = np.array([[self.sigma_z**2]])
    

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

    def update(self, measurements):
        
        """EKF update using multiple LIDAR range measurements and map-based expected distances."""

        # Angles corresponding to LIDAR beams: every 30° from 0 to 210
        indices = range(0, 240, 30)
        rays = PF.lidar_rays(self.pose, indices, self.sensor_range_max)

        x_r, y_r, theta_r = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        
        for ray in rays:
            intersection,  distance = self._map.check_collision(ray, True)

            if intersection:
                # Compute expected distance
                z = np.sqrt((intersection[0] - x_r)**2 + (intersection[1] - y_r)**2)
                
                # Compute Jacobian of the measurement model
                H = np.array([
                    [(x_r - intersection[0]) / z, (y_r - intersection[1]) / z, 0]
                ])
                
                # Compute Kalman gain
                S = H @ self.P @ H.T + self.R
                K = self.P @ H.T @ np.linalg.inv(S)
                
                # Update state and covariance
                z_hat = np.array([[distance]])
                y = z_hat - z
                
                self.x += K @ y
                self.P = (np.eye(3) - K @ H) @ self.P
        
