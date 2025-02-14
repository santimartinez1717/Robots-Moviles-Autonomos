import numpy as np


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""
    
    def __init__(self, dt: float) -> None:
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].

        """
        self._dt: float = dt
        self.target_distance = 0.5  # Distancia deseada a la pared [m]
        self.kp = 0.8  # Ganancia proporcional (ajustable)
        self.kd = 0.04  # Ganancia derivativa (ajustable)
        self.prev_error = 0.0

    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_scan: Distance from every LiDAR ray to the closest obstacle [m].
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 2.14. Complete the function body with your code (i.e., compute v and w).
        
        # Filtrar valores NaN del LiDAR
        z_scan = np.array(z_scan)
        z_scan = np.where(np.isnan(z_scan), 8.0, z_scan)  # Rango máximo de 8m para valores NaN
            
        # Obtener distancias relevantes
        front_distance = np.min(z_scan[0:10])  # Medidas frontales
        left_distance = np.min(z_scan[80:100])  # Medidas laterales izquierda
        right_distance = np.min(z_scan[180:240])  # Medidas laterales derecha

        print(z_scan)

        # Máquina de estados para evitar colisiones
        if front_distance < 0.3:  # Obstáculo muy cerca
            v = 0.0
            w = 0.5  # Gira sobre sí mismo
        else:
            v = 0.15  # Velocidad base

            # Control PD para seguir la pared
            error = self.target_distance - right_distance  # Error respecto a la pared derecha
            derivative = (error - self.prev_error) / self._dt
            w = self.kp * error + self.kd * derivative
            self.prev_error = error

        return v,w