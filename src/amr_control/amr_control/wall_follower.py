import math
import numpy as np

class WallFollower:
    def __init__(self, dt: float) -> None:
        """WallFollower class initializer.

        Args:
            dt: Sampling period [s].
        """
        self.dt: float = dt
        self.pred_dist = 0.2  # Desired distance to the wall [m]
        self.Kp = 2           # Proportional gain for trajectory corrections
        self.Kd = 1           # Derivative gain to avoid oscillations
        self.Ki = 0.005       # Integral gain for cumulative corrections

        self.integral_error = 0.0
        self.last_error = 0.0
        self.safety_dist = 0.22  # Minimum safety distance to the wall
        self.PRED_VEL = 0.5      # Desired linear velocity [m/s]
        self.PRED_ANG = 0.0      # Desired angular velocity [rad/s]

        # Turning and dead-end states
        self.turn_left_mode = False
        self.turn_right_mode = False
        self.dead_end_mode = False
        self.rotation_completed = 0.0

    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        """Wall-following algorithm.

        Args:
            z_scan: List of LiDAR distances to obstacles [m].
            z_v: Estimated linear velocity of the robot [m/s].
            z_w: Estimated angular velocity of the robot [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].
        """
        # Convert z_scan to numpy array and replace NaN values
        z_scan = np.array(z_scan)
        z_scan = np.where(np.isnan(z_scan), 8.0, z_scan)  # Replace NaN with 8.0 m

        # Get relevant distances for navigation
        front_distance = z_scan[0]     # Frontal measurements
        l_dist = z_scan[60]            # Left lateral measurements
        r_dist = z_scan[-60]           # Right lateral measurements

        # Initial velocity values
        v = self.PRED_VEL
        w = self.PRED_ANG

        # **1. DEAD-END DETECTION**
        if (front_distance <= self.safety_dist and l_dist <= 0.2 and r_dist <= 0.2):
            print("Dead-end detected")
            self.dead_end_mode = True

        # **2. FRONTAL OBSTACLE DETECTION AND TURN SELECTION**
        if front_distance <= self.safety_dist and not self.dead_end_mode and not self.turn_left_mode and not self.turn_right_mode:
            if r_dist >= l_dist:
                self.turn_right_mode = True
            else:
                self.turn_left_mode = True

        # **3. RIGHT TURN**
        if self.turn_right_mode:
            v = 0.0
            w = -1.0
            self.rotation_completed += abs(w) * self.dt
            if self.rotation_completed >= math.pi / 2:
                self.turn_right_mode = False
                self.reset()

            return v, w

        # **4. LEFT TURN**
        elif self.turn_left_mode:
            v = 0.0
            w = 1.0
            self.rotation_completed += abs(w) * self.dt
            if self.rotation_completed >= math.pi / 2:
                self.turn_left_mode = False
                self.reset()

            return v, w

        # **5. FULL TURN IN DEAD-END**
        elif self.dead_end_mode:
            v = 0.0
            w = 2.0
            self.rotation_completed += abs(w) * self.dt
            if self.rotation_completed >= math.pi:
                self.dead_end_mode = False
                self.reset()

            return v, w

        # **6. PID CONTROL FOR SMOOTH TRAJECTORY CORRECTIONS**
        if abs(l_dist - r_dist) < 0.2:
            error = l_dist - self.pred_dist
            derivative = (error - self.last_error) / self.dt
            self.integral_error += error * self.dt
            w = self.Kp * error + self.Kd * derivative + self.Ki * self.integral_error
            self.last_error = error

        return v, w

    def reset(self):
        """Resets internal variables after completing a turn."""
        self.last_error = 0.0
        self.integral_error = 0.0
        self.rotation_completed = 0.0