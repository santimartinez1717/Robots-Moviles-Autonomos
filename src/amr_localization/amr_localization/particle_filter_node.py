import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

import message_filters
from amr_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import math
import os
import time
import traceback
from transforms3d.euler import euler2quat

from amr_localization.particle_filter import ParticleFilter
from amr_localization.kalman_filter import ExtendedKalmanFilter

class ParticleFilterNode(LifecycleNode):
    def __init__(self):
        """Particle filter node initializer."""
        super().__init__("particle_filter")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_plot", False)
        self.declare_parameter("global_localization", True)
        self.declare_parameter("initial_pose", (0.0, 0.0, math.radians(0)))
        self.declare_parameter("initial_pose_sigma", (0.05, 0.05, math.radians(5)))
        self.declare_parameter("particles", 1000)
        self.declare_parameter("sigma_v", 0.1)
        self.declare_parameter("sigma_w", 0.1)
        self.declare_parameter("sigma_z", 0.1)
        self.declare_parameter("steps_btw_sense_updates", 10)
        self.declare_parameter("world", "lab03")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            dt = self.get_parameter("dt").get_parameter_value().double_value
            self._enable_plot = self.get_parameter("enable_plot").get_parameter_value().bool_value
            global_localization = (
                self.get_parameter("global_localization").get_parameter_value().bool_value
            )
            initial_pose = tuple(
                self.get_parameter("initial_pose").get_parameter_value().double_array_value.tolist()
            )
            initial_pose_sigma = tuple(
                self.get_parameter("initial_pose_sigma")
                .get_parameter_value()
                .double_array_value.tolist()
            )
            particles = self.get_parameter("particles").get_parameter_value().integer_value
            sigma_v = self.get_parameter("sigma_v").get_parameter_value().double_value
            sigma_w = self.get_parameter("sigma_w").get_parameter_value().double_value
            sigma_z = self.get_parameter("sigma_z").get_parameter_value().double_value
            self._steps_btw_sense_updates = (
                self.get_parameter("steps_btw_sense_updates").get_parameter_value().integer_value
            )
            world = self.get_parameter("world").get_parameter_value().string_value

            # Attribute and object initializations
            self._localized = False
            self._converged = False
            self._steps = 0
            map_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
            )
            self._particle_filter = ParticleFilter(
                dt,
                map_path,
                particle_count=particles,
                sigma_v=sigma_v,
                sigma_w=sigma_w,
                sigma_z=sigma_z,
                global_localization=global_localization,
                initial_pose=initial_pose,
                initial_pose_sigma=initial_pose_sigma,
            )

            if self._enable_plot:
                self._particle_filter.show("Initialization", save_figure=True)

            self._ekf = ExtendedKalmanFilter(
                dt = dt,
                sigma_v = sigma_v,
                sigma_w = sigma_w,
                sigma_z = sigma_z,
                map_path = map_path,
            )

            # Publishers
            # TODO: 3.1. Create the /pose publisher (PoseStamped message).
            
            self._pose_publisher = self.create_publisher(PoseStamped, "pose", 10)
            self._localized_publisher = self.create_publisher(PoseStamped, "localized_pose", 10)
            
            
            # Subscribers
            scan_qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )

            self._subscribers: list[message_filters.Subscriber] = []
            self._subscribers.append(message_filters.Subscriber(self, Odometry, "odometry"))
            self._subscribers.append(
                message_filters.Subscriber(self, LaserScan, "scan", qos_profile=scan_qos_profile)
            )

            ts = message_filters.ApproximateTimeSynchronizer(
                self._subscribers, queue_size=10, slop=9
            )
            ts.registerCallback(self._compute_pose_callback)

        except Exception:
            self.get_logger().error(f"{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR

        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles an activating transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'active' state.")

        return super().on_activate(state)

    def _compute_pose_callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        """Subscriber callback. Executes a particle filter and publishes (x, y, theta) estimates.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR sensor readings.

        """
        # Parse measurements
        z_v: float = odom_msg.twist.twist.linear.x
        z_w: float = odom_msg.twist.twist.angular.z
        z_scan: list[float] = scan_msg.ranges

        # Execute particle filter
        if not self._localized: 
            self._execute_motion_step(z_v, z_w) # Move particles in direction of odometry
            x_h, y_h, theta_h = self._execute_measurement_step(z_scan) # Resample and cluster particles
            self._steps += 1

        else: # Delegate to the EKF when localized
            if self._converged:
                # Initialize the EKF with the pose estimate
                x_h, y_h, theta_h = self._execute_measurement_step(z_scan)
                self.get_logger().info(f"EKF initialized with pose estimate: {x_h:.3f}, {y_h:.3f}, {math.degrees(theta_h):.3f} deg")
                self._ekf.initialize(x_h, y_h, theta_h)
                self._converged = False

            # Predict the EKF with the odometry measurements
            self._ekf.predict(z_v, z_w)
            # Update the EKF with the LiDAR measurements
            self._ekf.update(z_scan)

            # Get the pose estimate from the EKF
            x_h, y_h, theta_h = self._ekf.pose

        # Publish
        self._publish_pose_estimate(x_h, y_h, theta_h) # publish pose estimation based on clustering

    def _execute_measurement_step(self, z_us: list[float]) -> tuple[float, float, float]:
        """Executes and monitors the measurement step (sense) of the particle filter.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].

        Returns:
            Pose estimate (x_h, y_h, theta_h) [m, m, rad]; inf if cannot be computed.
        """
        pose = (float("inf"), float("inf"), float("inf"))

        if self._localized or not self._steps % self._steps_btw_sense_updates:
            start_time = time.perf_counter()

            # 1. RESAMPLING:
            self._particle_filter.resample(z_us)
            sense_time = time.perf_counter() - start_time
            self.get_logger().info(f"Sense step time: {sense_time:6.3f} s")

            # 2. Check for loss:
            if self._particle_filter.check_for_loss(z_us):
                self.get_logger().warn("El robot parece estar perdido. Reinicializando partículas...")
                self._particle_filter.reset_particles() # Reinitialize particle filter; start over
                self._localized = False  
                return pose  

            # 3. Visualize particles:
            if self._enable_plot:
                self._particle_filter.show("Sense", save_figure=True)

            # 4. CLUSTERING:
            start_time = time.perf_counter()
            self._localized, pose = self._particle_filter.compute_pose()
            self._converged = self._localized # if the clustering converged for the first time
            clustering_time = time.perf_counter() - start_time
            self.get_logger().info(f"Clustering time: {clustering_time:6.3f} s")

        return pose

    def _execute_motion_step(self, z_v: float, z_w: float):
        """Executes and monitors the motion step (move) of the particle filter.

        Args:
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].
        """
        start_time = time.perf_counter()
        self._particle_filter.move(z_v, z_w)
        move_time = time.perf_counter() - start_time

        self.get_logger().info(f"Move step time: {move_time:7.3f} s")

        if self._enable_plot:
            self._particle_filter.show("Move", save_figure=True)

    def _publish_pose_estimate(self, x_h: float, y_h: float, theta_h: float) -> None:
        """Publishes the robot's pose estimate in a custom amr_msgs.msg.PoseStamped message.

        Args:
            x_h: x coordinate estimate [m].
            y_h: y coordinate estimate [m].
            theta_h: Heading estimate [rad].

        """
        # TODO: 3.2. Complete the function body with your code (i.e., replace the pass statement).
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.localized = self._localized
        
        if self._localized:
            pose_msg.pose.position.x = x_h
            pose_msg.pose.position.y = y_h
            pose_msg.pose.position.z = 0.0
            
            self.get_logger().info(f"Pose: x = {x_h:.3f} m, y = {y_h:.3f} m, theta = {math.degrees(theta_h):.3f} deg")

            qw, qx, qy, qz = euler2quat(0, 0, theta_h)
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            
        self._pose_publisher.publish(pose_msg)
        

def main(args=None):
    rclpy.init(args=args)
    particle_filter_node = ParticleFilterNode()

    try:
        rclpy.spin(particle_filter_node)
    except KeyboardInterrupt:
        pass

    particle_filter_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()