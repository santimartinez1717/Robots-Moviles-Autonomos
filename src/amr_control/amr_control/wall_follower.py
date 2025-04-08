import math
import numpy as np

class WallFollower:
    def __init__(self, dt: float) -> None:
        """Inicializador de la clase WallFollower.

        Args:
            dt: Periodo de muestreo [s].
        """
        self.dt: float = dt
        self.pred_dist = 0.2  # Distancia deseada a la pared [m]
        self.Kp = 2           # Ganancia proporcional para correcciones de trayectoria
        self.Kd = 1           # Ganancia derivativa para evitar oscilaciones
        self.Ki = 0.005       # Ganancia integral para correcciones acumulativas

        self.integral_error = 0.0
        self.last_error = 0.0
        self.safety_dist = 0.22  # Distancia mínima de seguridad a la pared
        self.PRED_VEL = 0.25      # Velocidad lineal deseada [m/s]
        self.PRED_ANG = 0.0      # Velocidad angular deseada [rad/s]

        # Estados de giro y callejón sin salida
        self.turn_left_mode = False
        self.turn_right_mode = False
        self.dead_end_mode = False


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
        front_distance = z_scan[0]     # Medidas frontales
        l_dist = z_scan[60]            # Medidas laterales izquierda
        r_dist = z_scan[-60]           # Medidas laterales derecha

        # Valores de velocidad iniciales
        v = self.PRED_VEL
        w = self.PRED_ANG

        # **1. DETECCIÓN DE CALLEJÓN SIN SALIDA (dead end) **
        if (front_distance <= self.safety_dist and l_dist <= 0.2 and r_dist <= 0.2):
            print("Callejón sin salida detectado")
            self.dead_end_mode = True

        # **2. DETECCIÓN DE OBSTÁCULO FRONTAL Y SELECCIÓN DE GIRO**
        if front_distance <= self.safety_dist and not self.dead_end_mode and not self.turn_left_mode and not self.turn_right_mode:
            if r_dist >= l_dist:
                self.turn_right_mode = True
            else:
                self.turn_left_mode = True

        # **3. GIRO A LA DERECHA**
        if self.turn_right_mode:
            v = 0.0
            w = -1.0
            self.rotation_completed += abs(w) * self.dt
            if self.rotation_completed >= math.pi / 2:
                self.turn_right_mode = False
                self.reset()

            return v, w

        # **4. GIRO A LA IZQUIERDA**
        elif self.turn_left_mode:
            v = 0.0
            w = 1.0
            self.rotation_completed += abs(w) * self.dt
            if self.rotation_completed >= math.pi / 2:
                self.turn_left_mode = False
                self.reset()

            return v, w

        # **5. GIRO COMPLETO EN CALLEJÓN SIN SALIDA**
        elif self.dead_end_mode:
            v = 0.0
            w = 1.0
            self.rotation_completed += abs(w) * self.dt
            if self.rotation_completed >= math.pi:
                self.dead_end_mode = False
                self.reset()

            return v, w

        # **6. CONTROL PID PARA CORRECCIONES SUAVES DE TRAYECTORIA**
        if abs(l_dist - r_dist) < 0.2:
            error = l_dist - self.pred_dist
            derivative = (error - self.last_error) / self.dt
            self.integral_error += error * self.dt
            w = self.Kp * error + self.Kd * derivative + self.Ki * self.integral_error
            self.last_error = error

        return v, w

    def reset(self):
        """Reinicia variables internas tras completar un giro."""
        self.last_error = 0.0
        self.integral_error = 0.0
        self.rotation_completed = 0.0
