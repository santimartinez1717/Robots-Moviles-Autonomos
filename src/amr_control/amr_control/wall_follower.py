import math
import numpy as np

class WallFollower:
    def __init__(self, dt: float) -> None:
        """Inicializador de la clase WallFollower.

        Args:
            dt: Periodo de muestreo [s].
        """
        self._dt: float = dt
        self._desired_distance = 0.2  # Distancia deseada a la pared [m]
        self.Kp = 2  # Ganancia proporcional para correcciones de trayectoria
        self.Kd = 1  # Ganancia derivativa para evitar oscilaciones
        self.Ki = 0.005  # Ganancia integral para correcciones acumulativas
        self.integral_error = 0.0
        self.last_error = 0.0
        self._safety_distance = 0.22  # Distancia mínima de seguridad a la pared

        # Estados de giro y callejón sin salida
        self._turn_left_mode = False
        self._turn_right_mode = False
        self._dead_end_mode = False
        self._rotation_completed = 0.0

    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        """Algoritmo de seguimiento de paredes.

        Args:
            z_scan: Lista de distancias LiDAR a los obstáculos [m].
            z_v: Estimación de velocidad lineal del robot [m/s].
            z_w: Estimación de velocidad angular del robot [rad/s].

        Returns:
            v: Velocidad lineal [m/s].
            w: Velocidad angular [rad/s].
        """
        # Convertir z_scan a numpy array y reemplazar valores NaN
        z_scan = np.array(z_scan)
        z_scan = np.where(np.isnan(z_scan), 8.0, z_scan)  # Reemplazar NaN con 8.0 m

        # Obtener distancias relevantes para la navegación
        front_distance = z_scan[0]  # Medidas frontales
        left_distance = z_scan[59]  # Medidas laterales izquierda
        right_distance = z_scan[-59]  # Medidas laterales derecha

        # Valores de velocidad iniciales
        v = 0.15  # Para el robot real, usar 0.1
        w = 0.0

        # **1. DETECCIÓN DE CALLEJÓN SIN SALIDA**
        if (front_distance <= self._safety_distance
            and left_distance <= 0.23
            and right_distance <= 0.23
        ):
            self._dead_end_mode = True


        #GIRO DE 180 GRADOS PARA SALIR DE CALLEJÓN SIN SALIDA
        if self._dead_end_mode:
            v = 0.0
            w = 1.0  
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi:  # Giro de 180 grados
                self._dead_end_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0
            return v, w

        # **2. DETECCIÓN DE OBSTÁCULO FRONTAL Y SELECCIÓN DE GIRO**
        if front_distance <= self._safety_distance:
            if right_distance >= left_distance:
                self._turn_right_mode = True
            else:
                self._turn_left_mode = True

        # **3. GIRO A LA DERECHA**
        if self._turn_right_mode:
            v = 0.0
            w = -1.0
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi / 2:  # Giro de 90 grados
                self._turn_right_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0
            return v, w

        # **4. GIRO A LA IZQUIERDA**
        elif self._turn_left_mode:
            v = 0.0
            w = 1.0
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi / 2:  # Giro de 90 grados
                self._turn_left_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0
            return v, w

        # **5. CONTROL PID PARA CORRECCIONES SUAVES DE TRAYECTORIA**
        # Si el robot se acerca demasiado a la pared, el PID ajusta la trayectoria.
        if abs(left_distance - right_distance) < 0.2:
            error = left_distance - self._desired_distance  # Error respecto a la pared
            derivative = (error - self.last_error) / self._dt
            self.integral_error += error * self._dt

            # **Control PID para corrección suave**
            w = self.Kp * error + self.Kd * derivative + self.Ki * self.integral_error
            self.last_error = error

        return v, w
