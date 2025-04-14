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


    def update(self, measurements):
        """Updates the state using LIDAR measurements as lines (p, alpha)."""
        rays = PF.lidar_rays(self.pose, range(0, 240, 1), self.sensor_range_max)
        points = []

        # Step 1: Collect valid LIDAR points in global coordinates
        for z, ray in zip(measurements, rays):
            if math.isnan(z) or z < self.sensor_range_min or z > self.sensor_range_max:
                continue

            origin = np.array(ray[0]).flatten()
            if origin.shape[0] < 2:
                continue  # malformed point

            x0, y0 = float(origin[0]), float(origin[1])
            angle = ray[1]

            # Endpoint of the ray
            x1 = x0 + z * np.cos(angle)
            y1 = y0 + z * np.sin(angle)

            points.append((x1, y1))

        # Step 2: If there are enough points, try to fit a line with 3 of them
        if len(points) < 3:
            return

        try:
            # Take the first 3 valid points
            def flatten_point(p):
                x = float(np.array(p[0]).flatten()[0])
                y = float(np.array(p[1]).flatten()[0])
                return (x, y)

            raw_p1, raw_p2, raw_p3 = points[0], points[1], points[2]
            p1 = flatten_point(raw_p1)
            p2 = flatten_point(raw_p2)
            p3 = flatten_point(raw_p3)

            # Build matrix A for SVD (ax + by + c = 0)
            A = np.array([
                [p1[0], p1[1], 1.0],
                [p2[0], p2[1], 1.0],
                [p3[0], p3[1], 1.0]
            ])

            _, _, vh = np.linalg.svd(A)
            line = vh[-1, :]
            a, b, c = line / np.linalg.norm(line[:2])  # normalize

        except Exception as e:
            print(f"[update] Error estimating line: {e}")
            return

        # Step 3: Build the expected measurement prediction z_hat = h(x)
        x_r, y_r, theta_r = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        p = a * x_r + b * y_r + c
        alpha = np.arctan2(b, a) - theta_r
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # normalize angle

        z_hat = np.array([[p], [alpha]])

        # Step 4: Real observation (from detected landmark)
        z = np.array([[p], [alpha]])  # use the line as the "real" observation

        # Step 5: Jacobian of the observation function h
        H = np.array([
            [a, b, 0],
            [0, 0, -1]
        ])

        # Step 6: EKF update
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - z_hat
        y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi  # normalize angle

        self.x += K @ y
        self.x[2, 0] = (self.x[2, 0] + np.pi) % (2 * np.pi) - np.pi  # normalize orientation
        self.P = (np.eye(3) - K @ H) @ self.P
