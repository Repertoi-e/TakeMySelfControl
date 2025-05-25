from PySide6.QtGui import (QImage)
from PySide6.QtCore import Qt

from enum import Enum
import math
import random

import numpy as np

from physics.body import Body, ensure_transformed_shape, apply_impulse, apply_force
from physics.shape import Circle, ConvexPolygon, make_rect

from physics.vec import clamp_magnitude, magnitude, sqr_magnitude, normalized, dot, dot_mat, orthogonal
from physics.collision import collide
from physics.hit import ray_vs_segment

import numba
from numba.core import types
from numba import njit
from numba.np.extensions import cross2d


import pygame as pg

class Direction(Enum):
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

OPPOSITE = {
    Direction.TOP: Direction.BOTTOM,
    Direction.RIGHT: Direction.LEFT,
    Direction.BOTTOM: Direction.TOP,
    Direction.LEFT: Direction.RIGHT
}

OFFSETS = {
    Direction.TOP: np.array([0, -1]),
    Direction.RIGHT: np.array([1, 0]),
    Direction.BOTTOM: np.array([0, 1]),
    Direction.LEFT: np.array([-1, 0])
}

maze = """
o---o---o---o---o---o---o---o---o---o---o---o---o---o---o---o---o
|                               |                               |
o   o---o---o---o---o---o---o   o   o---o---o---o   o   o---o   o
|   |                                               |       |   |
o   o   o   o---o---o---o---o---o---o---o---o---o   o---o   o---o
|   |   |                                       |       |       |
o   o   o   o---o---o---o---o---o---o---o---o   o---o   o---o   o
|   |   |   |                               |       |       |   |
o   o   o   o   o---o---o---o---o---o---o   o---o   o---o   o   o
|   |   |   |   |                                               |
o   o   o   o   o   o---o---o---o---o   o---o   o---o---o---o   o
|   |   |   |       |               |       |       |           |
o   o   o   o   o   o   o---o---o   o   o   o---o   o   o   o   o
|   |   |   |   |       |       |       |       |       |   |   |
o   o   o   o   o   o   o   o   o---o   o---o   o   o   o   o   o
|       |   |   |   |   |   | G   G |       |   |   |   |       |
o---o   o   o   o   o   o   o   o   o   o   o   o   o   o   o---o
|       |   |   |   |       | G   G |   |   |   |   |   |       |
o   o   o   o   o   o---o   o---o---o   o   o   o   o   o   o   o
|   |   |       |       |               |   |   |   |   |   |   |
o   o   o   o   o---o   o---o   o---o---o   o   o   o   o   o   o
|   |   |   |       |       |               |   |   |   |   |   |
o   o   o   o---o   o   o   o---o---o   o---o   o   o   o   o   o
|   |   |       |       |                       |   |   |   |   |
o   o   o---o   o---o   o---o---o---o---o---o---o   o   o   o   o
|   |       |       |                               |   |   |   |
o   o---o   o---o   o   o---o---o---o---o---o   o---o   o   o   o
|       |       |   |                                   |   |   |
o---o   o---o   o   o---o---o---o---o---o---o---o   o---o   o   o
|       |   |                                   |           |   |
o   o   o   o---o---o---o---o   o   o---o---o---o---o   o---o   o
| S |                           |                               |
o---o---o---o---o---o---o---o---o---o---o---o---o---o---o---o---o
"""

maze_lines = maze.strip().split("\n")

def get_walls_from_maze_for(cell_x, cell_y):
    walls = {
        Direction.TOP: False,
        Direction.RIGHT: False,
        Direction.BOTTOM: False,
        Direction.LEFT: False
    }
    
    # Check bounds
    max_y = (len(maze_lines) - 1) // 2
    max_x = (len(maze_lines[0]) - 1) // 4
    if cell_x < 0 or cell_x > max_x or cell_y < 0 or cell_y > max_y:
        return walls
    
    # Calculate positions
    top_line = cell_y * 2
    bottom_line = cell_y * 2 + 2
    left_col = cell_x * 4
    right_col = cell_x * 4 + 4
    
    # Check top wall (horizontal line above the cell)
    if top_line >= 0:
        for x in range(left_col + 1, right_col):
            if x < len(maze_lines[top_line]) and maze_lines[top_line][x] == '-':
                walls[Direction.TOP] = True
                break
    
    # Check bottom wall (horizontal line below the cell)
    if bottom_line < len(maze_lines):
        for x in range(left_col + 1, right_col):
            if x < len(maze_lines[bottom_line]) and maze_lines[bottom_line][x] == '-':
                walls[Direction.BOTTOM] = True
                break
    
    # Check left wall (vertical line to the left of the cell)
    middle_line = cell_y * 2 + 1
    if left_col >= 0 and middle_line < len(maze_lines):
        if maze_lines[middle_line][left_col] == '|':
            walls[Direction.LEFT] = True
    
    # Check right wall (vertical line to the right of the cell)
    if right_col < len(maze_lines[0]) and middle_line < len(maze_lines):
        if maze_lines[middle_line][right_col] == '|':
            walls[Direction.RIGHT] = True
    
    return walls


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        walls = get_walls_from_maze_for(x, y)
        self.wall_top: bool = walls[Direction.TOP]
        self.wall_right: bool = walls[Direction.RIGHT]
        self.wall_bottom: bool = walls[Direction.BOTTOM]
        self.wall_left: bool = walls[Direction.LEFT]

        self.last_drag_ids: dict[Direction, int] = {
            Direction.TOP: None,
            Direction.RIGHT: None,
            Direction.BOTTOM: None,
            Direction.LEFT: None
        }
        
CELL_COUNT = 16
CELL_SIZE = 64
WALL_THICKNESS = 4

MAZE_SIZE = CELL_COUNT * CELL_SIZE + WALL_THICKNESS * 3

gravity = 9.81 * 100 # centi meters per second squared

def cross_scalar(s, v):
	return [-s * v[1], s * v[0]]

def angle_diff(a, b):
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi

class KalmanFilter:
    def __init__(self, initial_pos, initial_theta):
        # State vector: [x, y, theta, v, omega]
        self.X = np.zeros((5, 1))
        self.X[0:3, 0] = [initial_pos[0], initial_pos[1], initial_theta]
        
        # Covariance matrix (now 5x5)
        self.P = np.eye(5) * 0.1
        
        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])  # Small noise for bias

        # Measurement noise - separate for position and orientation
        self.R_pos = np.diag([0.05, 0.05])  # x, y
        self.R_gyro = np.array([[0.02]]) 

    def predict(self, v, omega_meas, dt):
        """Predict step using control inputs (v, omega)"""
        x, y, theta, _, _ = self.X.flatten()

        omega_corrected = omega_meas

        # Predict next state (no angle wrapping!)
        self.X[0] = x + v * np.cos(theta) * dt
        self.X[1] = y + v * np.sin(theta) * dt
        self.X[2] = theta + omega_corrected * dt  # unwrapped
        self.X[3] = v
        self.X[4] = omega_corrected

        # Jacobian
        F = np.eye(5)
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        F[2, 4] = dt

        self.P = F @ self.P @ F.T + self.Q

    def update_position(self, x_meas, y_meas):
        """2D position update"""
        H = np.zeros((2, 5))
        H[0, 0] = 1  # ∂x_meas/∂x
        H[1, 1] = 1  # ∂y_meas/∂y

        z = np.array([[x_meas], [y_meas]])
        y = z - self.X[:2]  # Innovation

        S = H @ self.P @ H.T + self.R_pos
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X += K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    def update_gyro(self, omega_meas):
        """Update step for angular velocity measurement (gyro reading)."""
        H = np.zeros((1, 5))
        H[0, 4] = 1  # ∂z/∂omega

        z = np.array([[omega_meas]])
        y = np.array([[omega_meas - self.X[4, 0]]])  # residual

        S = H @ self.P @ H.T + self.R_gyro
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X += K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    def get_estimated_pose(self):
        return self.X[0, 0], self.X[1, 0], self.X[2, 0]

STARTING_POS_OFFSET = np.array([0.5 * CELL_SIZE, 15.5 * CELL_SIZE])

class TOFDirection(Enum):
    FRONT = 0
    LEFT = 1
    RIGHT = 2
    FRONT_LEFT_45 = 3
    FRONT_RIGHT_45 = 4


@njit
def ray_segment_new(ray, ray_dir, segment):
    ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-6)
    point1, point2 = segment[0], segment[1]

    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    v1 = ray - point1
    v2 = point2 - point1
    v3 = np.array([-ray_dir[1], ray_dir[0]])
    t1 = cross2d(v2, v1) / (np.dot(v2, v3) + 1e-6)
    t2 = np.dot(v1, v3) / (np.dot(v2, v3) + 1e-6)
    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return ray + t1 * ray_dir
    return np.array([np.nan, np.nan])  # No intersection, return NaN

@njit
def update_tof_sensor_data(sensor_data, pos, rot, walls):
    closest_distance_per_sensor = [float('inf') for _ in range(5)]
    closest_point_per_sensor = [np.zeros(2) for _ in range(5)]

    ANGLE_OFFSETS = [0, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4]
    angle_noise = np.random.normal(0, 0.001, 5)  # Small noise for angles

    # Check intersection with all walls
    for aabb in walls:
        m = aabb[0] - aabb[1]
        min_pt = np.array([m[0], m[1]])
        max_pt = min_pt + np.array([aabb[1][0] * 2, aabb[1][1] * 2])
        
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        
        if width > height:  # Horizontal wall
            wall_center = (min_pt + max_pt) / 2
            wall_segment = (
                np.array([min_pt[0], wall_center[1]]),
                np.array([max_pt[0], wall_center[1]])
            )
        else:  # Vertical wall
            wall_center = (min_pt + max_pt) / 2
            wall_segment = (
                np.array([wall_center[0], min_pt[1]]),
                np.array([wall_center[0], max_pt[1]])
            )

        for dir in range(5):
            angle = -rot + ANGLE_OFFSETS[dir] + angle_noise[dir] 
            ray_dir = np.array([math.sin(angle), -math.cos(angle)])  
            ip = ray_segment_new(pos, ray_dir, wall_segment)

            if not np.isnan(ip[0]) and not np.isnan(ip[1]):
                distance = np.linalg.norm(ip - pos)
                if distance < closest_distance_per_sensor[dir]:
                    closest_distance_per_sensor[dir] = distance
                    closest_point_per_sensor[dir] = ip

    for dir, point in enumerate(closest_point_per_sensor):
        if point is not None:
            distance = np.linalg.norm(point - pos)
            noise_magnitude = 0.01 * distance
            noise = np.random.normal(0, noise_magnitude, 2)
            sensor_data[dir] = point + noise    

class Mouse(Body):
    image: pg.Surface

    def __init__(self, uber):
        super().__init__(shape=Circle(radius=CELL_SIZE // 3), density=0.001, static=False)
        self.u = uber.mission

        self.pos = STARTING_POS_OFFSET.copy() 

        self.true_v = 0
        self.true_omega = 0
        
        self.wheel_radius = 20  # cm
        self.wheel_base = 10    # cm (distance between wheels)
        
        self.kalman = KalmanFilter(initial_pos=np.zeros(2), initial_theta=np.pi/2)  # 60 FPS

        self.sensor_data = [np.zeros(2) for _ in range(5)]

        self.image = pg.image.load("sprite.png").convert_alpha()

    def handle_input(self, keys):
        if keys[pg.K_w] or keys[pg.K_s]:
            sign = 1 if keys[pg.K_w] else -1
            
            # Move forward in the current facing direction
            force = np.array([0, -self.u.motor_force * sign])

            angle = -self.rot
            rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]])
            force = np.dot(rotation_matrix, force)

            apply_force(self, force)

        if keys[pg.K_a]:
            self.torque += self.u.rotation_torque

        if keys[pg.K_d]:
            self.torque += -self.u.rotation_torque

    def update(self, dt, renderer):
        ensure_transformed_shape(self)

        self.noise_mag = 0

        acc = self.force * self.inv_mass
        self.vel += acc * dt

        ang_acc = self.torque * self.inv_rot_inertia
        self.ang_vel += ang_acc * dt

        # Apply air drag
        self.ang_vel *= self.u.drag

        # Apply friction
        if self.vel[0] != 0 or self.vel[1] != 0:
            self.vel -= self.vel * (1 - self.u.friction)

        a = self
        for b in renderer.walls:
            contacts = collide(a, b)

            # Precompute constant stuff for each iteration
            for c in contacts:
                c.r1 = c.pos - a.pos
                c.r2 = c.pos - b.pos

                rn1 = dot(c.r1, c.normal)
                rn2 = dot(c.r2, c.normal)
                k = a.inv_mass + b.inv_mass
                k += a.inv_rot_inertia * (sqr_magnitude(c.r1) - rn1 ** 2) + b.inv_rot_inertia * (sqr_magnitude(c.r2) - rn2 ** 2)
                c.inv_k = 1 / k
                
                allowed_penetration = 0.01
                position_bias_factor = 0.2
        
                c.bias = -position_bias_factor * 1 / dt * min(0, -c.collision_depth + allowed_penetration)
            
            # Perform iterations
            for _ in range(10):
                for c in contacts:
                    # Relative velocity at each contact
                    rel_vel = b.vel + cross_scalar(b.ang_vel, c.r2) - a.vel - cross_scalar(a.ang_vel, c.r1)
                    vel_along_normal = dot(rel_vel, c.normal)
                    
                    j = max(c.inv_k * (-vel_along_normal + c.bias), 0)
                    impulse = j * c.normal

                    apply_impulse(self, -impulse, c.r1)
                    apply_impulse(b, impulse, c.r2)

        self.pos += self.vel * dt
        self.rot += self.ang_vel * dt

        self.force = np.array([0.0, 0.0])
        self.torque = 0
        
        self.dirty_transform = True

        self.update_with_oracle(dt)
        update_tof_sensor_data(self.sensor_data, self.pos, self.rot, renderer.walls_aabbs)

    def update_with_oracle(self, dt):
        effective_vel = np.linalg.norm(self.vel)
        direction_vector = np.array([-math.sin(self.rot), -math.cos(self.rot)])
        
        if np.dot(self.vel, direction_vector) < 0:
            effective_vel = -effective_vel

        left_rpm = ((effective_vel - self.ang_vel * self.wheel_base/2) /  (2 * math.pi * self.wheel_radius)) * 60  
        right_rpm = ((effective_vel + self.ang_vel * self.wheel_base/2) /  (2 * math.pi * self.wheel_radius)) * 60

        left_rpm += random.gauss(0, 5) * self.noise_mag  
        right_rpm += random.gauss(0, 5) * self.noise_mag  

        print(f"Left 🛞 RPM: {left_rpm:.1f}, Right 🛞 RPM: {right_rpm:.1f}")
        # print(f"left_rpm = {left_rpm:.1f}, right_rpm = {right_rpm:.1f}")
        
        # Convert back to v and omega (this is what the robot would do)
        slip_factor = 0.95  # Empirical value (0.9-1.0)
        measured_v = slip_factor * (left_rpm + right_rpm) / 2 * (2 * math.pi * self.wheel_radius) / 60
        if abs(measured_v) < 0.001:  # Threshold for "stopped"
            measured_v = 0
        measured_omega = (right_rpm - left_rpm) / self.wheel_base * (2 * math.pi * self.wheel_radius) / 60
        
        # Simulate IMU measurement (angular velocity with noise)
        imu_omega = self.ang_vel + random.gauss(0, 0.1) * self.noise_mag   # rad/s noise

        #imu_theta = self.rot + random.gauss(0, 0.05) * self.noise_mag    # rad noise (if IMU has orientation)
        #imu_theta = (imu_theta + np.pi) % (2 * np.pi) - np.pi  # Convert to [-π, π]

        # print(f"IMU Omega: {imu_omega}, IMU Theta: {imu_theta}, True Theta: {self.rot}, True Omega: {self.ang_vel}", "measured_v:", measured_v, "measured_omega:", measured_omega, "true_v:", self.true_v)

        # Update Kalman filter with measurements

        self.kalman.predict(measured_v, imu_omega, dt)
        self.kalman.update_gyro(imu_omega)

        # Using TOF for position measurement, i.e. helping the Kalman filter
        measured_x = self.pos[0] - STARTING_POS_OFFSET[0] + random.gauss(0, 1) * self.noise_mag
        measured_y = -(self.pos[1] - STARTING_POS_OFFSET[1]) + random.gauss(0, 1) * self.noise_mag

        self.kalman.update_position(measured_x, measured_y)

    def draw(self, surface):
        rotated_image = pg.transform.rotate(self.image, math.degrees(self.rot))
        rotated_rect = rotated_image.get_rect(center=self.pos)
        surface.blit(rotated_image, rotated_rect)

        est_x, est_y, est_theta = self.kalman.get_estimated_pose()

        est_dir = np.array([math.cos(est_theta), -math.sin(est_theta)]) * 20
        draw_pos = STARTING_POS_OFFSET.copy() + np.array([est_x, -est_y])

        pg.draw.circle(surface, (0, 255, 0), (int(draw_pos[0]), int(draw_pos[1])), 8)
        pg.draw.line(surface, (0, 255, 0), (int(draw_pos[0]), int(draw_pos[1])),
                     (int((draw_pos[0] + est_dir[0])), int((draw_pos[1] + est_dir[1]))), 2)

        for dir, point in enumerate(self.sensor_data):
            color = (255, 0, 0) if dir == 0 else (0, 255, 0)  # Front sensor in red, others in green
            pg.draw.line(surface, color, self.pos, point, 1)
            
            # Draw distance text
            distance = np.linalg.norm(point - self.pos)
            font = pg.font.SysFont(None, 20)
            text = font.render(f"{distance:.1f}", True, (255, 255, 255))
            text_pos = (self.pos + point) / 2
            surface.blit(text, text_pos)

class PgRenderer:
    width: int = MAZE_SIZE
    height: int = MAZE_SIZE

    surface: pg.Surface = pg.Surface((width, height))
    maze: list[list[Cell]] = [[Cell(x, y) for x in range(CELL_COUNT)] for y in range(CELL_COUNT)]
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = None

    mouse: Mouse

    clock: pg.time.Clock = pg.time.Clock()
    pressed_keys: list[bool] = [False] * 323

    def __init__(self, uber):
        self.uber = uber

        pg.init()
        pg.display.set_mode((1, 1))

        self.mouse = Mouse(uber)

        for y in range(CELL_COUNT):
            for x in range(CELL_COUNT):
                cell = self.maze[y][x]
                if y == 0:
                    cell.wall_top = True
                if y == CELL_COUNT - 1:
                    cell.wall_bottom = True
                if x == 0:
                    cell.wall_left = True
                if x == CELL_COUNT - 1:
                    cell.wall_right = True
        self.refresh_walls()

    def toggle_wall(self, x, y, direction, drag_id=None):
        offset_x, offset_y = OFFSETS[direction]
        neighbor_x, neighbor_y = x + offset_x, y + offset_y

        cell = self.maze[y][x]

        if not (0 <= neighbor_x < CELL_COUNT and 0 <= neighbor_y < CELL_COUNT):
            neighbor = None
        else:
            neighbor = self.maze[neighbor_y][neighbor_x]

        if cell.last_drag_ids[direction] == drag_id or neighbor.last_drag_ids[direction] == drag_id:
            return  # Already handled

        wall_attrs = {
            Direction.TOP:    ("wall_top",    "wall_bottom"),
            Direction.RIGHT:  ("wall_right",  "wall_left"),
            Direction.BOTTOM: ("wall_bottom", "wall_top"),
            Direction.LEFT:   ("wall_left",   "wall_right"),
        }

        cell_attr, neighbor_attr = wall_attrs[direction]

        # Toggle the wall
        wall_state = getattr(cell, cell_attr) or getattr(neighbor, neighbor_attr) if neighbor is not None else False
        setattr(cell, cell_attr, not wall_state)
        if neighbor is not None:
            setattr(neighbor, neighbor_attr, False)

        cell.last_drag_ids[direction] = drag_id

        self.refresh_walls()

    def mouse_pressed_in_game(self, x, y, drag_id=None):
        # Convert to maze coordinates
        x, xr = divmod(x - WALL_THICKNESS, CELL_SIZE)
        y, yr = divmod(y - WALL_THICKNESS, CELL_SIZE)

        if 0 <= x < CELL_COUNT and 0 <= y < CELL_COUNT:
            if xr < yr:
                if xr > CELL_SIZE // 2:
                    self.toggle_wall(x, y, Direction.RIGHT, drag_id)
                else:
                    self.toggle_wall(x, y, Direction.LEFT, drag_id)
            else:
                if yr > CELL_SIZE // 2:
                    self.toggle_wall(x, y, Direction.BOTTOM, drag_id)
                else:
                    self.toggle_wall(x, y, Direction.TOP, drag_id)
            self.refresh_walls()

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_W:
            self.pressed_keys[pg.K_w] = True
        elif key == Qt.Key_S:
            self.pressed_keys[pg.K_s] = True
        elif key == Qt.Key_A:
            self.pressed_keys[pg.K_a] = True
        elif key == Qt.Key_D:
            self.pressed_keys[pg.K_d] = True

    def keyReleaseEvent(self, event):
        key = event.key()

        if key == Qt.Key_W:
            self.pressed_keys[pg.K_w] = False
        elif key == Qt.Key_S:
            self.pressed_keys[pg.K_s] = False
        elif key == Qt.Key_A:
            self.pressed_keys[pg.K_a] = False
        elif key == Qt.Key_D:
            self.pressed_keys[pg.K_d] = False

    def refresh_walls(self):
        def add_wall(rect):
            half_width = rect.width // 2
            half_height = rect.height // 2

            center = np.array([rect.left + half_width, rect.top + half_height])
            wall = Body(shape=make_rect(half_width, half_height), density=1000, static=True)
            wall.pos = np.array(center)
            wall.dirty_transform = True

            ensure_transformed_shape(wall)

            self.walls.append(wall)
            self.walls_aabbs.append(wall.transformed_shape.aabb)
    
        self.walls = []
        self.walls_aabbs = []
        
        for y in range(CELL_COUNT):
            for x in range(CELL_COUNT):
                cell = self.maze[y][x]
                cx = x * CELL_SIZE + WALL_THICKNESS + WALL_THICKNESS // 2
                cy = y * CELL_SIZE + WALL_THICKNESS + WALL_THICKNESS // 2

                if cell.wall_top:
                    add_wall(pg.Rect(cx, cy - WALL_THICKNESS // 2, CELL_SIZE, WALL_THICKNESS))
                if cell.wall_right:
                    add_wall(pg.Rect(cx + CELL_SIZE - WALL_THICKNESS // 2, cy, WALL_THICKNESS, CELL_SIZE))
                if cell.wall_bottom:
                    add_wall(pg.Rect(cx, cy + CELL_SIZE - WALL_THICKNESS // 2, CELL_SIZE, WALL_THICKNESS))
                if cell.wall_left:
                    add_wall(pg.Rect(cx - WALL_THICKNESS // 2, cy, WALL_THICKNESS, CELL_SIZE))
                
    
    def render(self):
        self.surface.fill((0, 0, 0))

        dt = self.clock.tick(60) / 1000.0

        self.mouse.handle_input(self.pressed_keys)
        self.mouse.update(dt, self)
        self.mouse.draw(self.surface)

        for wall in self.walls:
            aabb = wall.transformed_shape.aabb
            m = aabb[0] - aabb[1]
            pg.draw.rect(self.surface, (240, 0, 0), pg.Rect(m[0], m[1], aabb[1][0] * 2, aabb[1][1] * 2), WALL_THICKNESS)

        # Draw corners as filled circles 
        for y in range(CELL_COUNT + 1):
            for x in range(CELL_COUNT + 1):
                px = x * CELL_SIZE + WALL_THICKNESS + WALL_THICKNESS // 2
                py = y * CELL_SIZE + WALL_THICKNESS + WALL_THICKNESS // 2
                pg.draw.circle(self.surface, (240, 240, 240), (px, py), WALL_THICKNESS // 2 + 1)

        return self.surface


def pg_surface_to_qimage(surface):
    data = pg.image.tostring(surface, "RGB")
    return QImage(data, surface.get_width(), surface.get_height(), QImage.Format_RGB888)

