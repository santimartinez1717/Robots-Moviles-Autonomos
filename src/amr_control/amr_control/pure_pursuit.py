import numpy as np

class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(self, dt: float, lookahead_distance: float = 0.5):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].

        """
        self._dt: float = dt
        self._lookahead_distance: float = lookahead_distance
        self._path: list[tuple[float, float]] = []

    def compute_commands(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Pure pursuit controller implementation.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # 4.11. Complete the function body with your code (i.e., compute v and w).
        closest_xy, closest_idx = self._find_closest_point(x, y)
        target_xy = self._find_target_point(closest_xy, closest_idx)

        # Compute the angle between the robot and the target point
        alpha = np.arctan2(target_xy[1] - y, target_xy[0] - x) - theta
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # If the robot is not reasonably aligned with the path, rotate in place
        if abs(alpha) > np.pi / 6:  # Threshold angle (30 degrees)
            v = 0.0  # No linear velocity
            w = 2.0 * alpha  # Proportional angular velocity
        else:
            v = self._lookahead_distance * np.cos(alpha)
            w = (2 * v * np.sin(alpha)) / self._lookahead_distance
        
        return v, w

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value

    def _find_closest_point(self, x: float, y: float) -> tuple[tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # 4.9. Complete the function body (i.e., find closest_xy and closest_idx).
        closest_node = min(self._path, key=lambda node: np.linealg.norm(np.array(node) - np.array((x, y))))
        closest_idx = self._path.index(closest_node)
    
        return closest_node, closest_idx
        
    def _find_target_point(
        self, origin_xy: tuple[float, float], origin_idx: int
    ) -> tuple[float, float]:
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # 4.10. Complete the function body with your code (i.e., determine target_xy).

        target_xy = self._path[origin_idx]
        for i in range(origin_idx, len(self._path)):
            if np.linalg.norm(np.array(self._path[i]) - np.array(origin_xy)) > self._lookahead_distance:
                target_xy = self._path[i]
                break
        
        # En caso de que no haya un punto exactamente a esa distancia, se toma el punto más cercano
        return target_xy
        