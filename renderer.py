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
from numba.experimental import jitclass
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

def get_walls_from_maze_for(cell_x: int, cell_y: int):
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
        return False, False, False, False
    
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
    
    return walls[Direction.TOP], walls[Direction.RIGHT], walls[Direction.BOTTOM], walls[Direction.LEFT]

cell_spec = [
    ('x', numba.int32),
    ('y', numba.int32),
    ('wall_top', numba.boolean),
    ('wall_right', numba.boolean),
    ('wall_bottom', numba.boolean),
    ('wall_left', numba.boolean),
]

@jitclass(cell_spec)
class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

        self.wall_top = False
        self.wall_right = False
        self.wall_bottom = False
        self.wall_left = False

SCALE = 64 / 18

CELL_COUNT = 16
CELL_SIZE = 18 * SCALE
WALL_THICKNESS = 4

MAZE_SIZE = CELL_COUNT * CELL_SIZE + WALL_THICKNESS * 3

def cross_scalar(s, v):
	return [-s * v[1], s * v[0]]

class TOFDirection(Enum):
    FRONT = 0
    LEFT = 1
    RIGHT = 2
    FRONT_LEFT_45 = 3
    FRONT_RIGHT_45 = 4

def fast_ray_march(
    ray_origin_world: np.ndarray,
    ray_dir_world: np.ndarray,
    maze: list,  # Assuming maze is a list of lists of Cell-like objects or a Numba-compatible structure
    MAX_SENSOR_RANGE: float
):
    norm_ray_dir = np.linalg.norm(ray_dir_world)
    if norm_ray_dir < 1e-9: # Ray direction is zero vector
        return np.array([np.nan, np.nan], dtype=np.float64)
    ray_dir_normalized = ray_dir_world / norm_ray_dir

    if not maze or not maze[0]: # Empty maze
        return np.array([np.nan, np.nan], dtype=np.float64)
    
    maze_grid_height = len(maze)
    maze_grid_width = len(maze[0])

    map_x = int(math.floor(ray_origin_world[0] / CELL_SIZE))
    map_y = int(math.floor(ray_origin_world[1] / CELL_SIZE))

    ray_step_x = 1 if ray_dir_normalized[0] > 0 else -1 if ray_dir_normalized[0] < 0 else 0
    ray_step_y = 1 if ray_dir_normalized[1] > 0 else -1 if ray_dir_normalized[1] < 0 else 0

    # Handle rays parallel to axes
    t_delta_x = abs(CELL_SIZE / ray_dir_normalized[0]) if ray_dir_normalized[0] != 0 else float('inf')
    t_delta_y = abs(CELL_SIZE / ray_dir_normalized[1]) if ray_dir_normalized[1] != 0 else float('inf')

    if ray_dir_normalized[0] > 0:
        t_max_x = ((map_x + 1) * CELL_SIZE - ray_origin_world[0]) / ray_dir_normalized[0]
    elif ray_dir_normalized[0] < 0:
        t_max_x = (map_x * CELL_SIZE - ray_origin_world[0]) / ray_dir_normalized[0]
    else:
        t_max_x = float('inf')

    if ray_dir_normalized[1] > 0:
        t_max_y = ((map_y + 1) * CELL_SIZE - ray_origin_world[1]) / ray_dir_normalized[1]
    elif ray_dir_normalized[1] < 0:
        t_max_y = (map_y * CELL_SIZE - ray_origin_world[1]) / ray_dir_normalized[1]
    else:
        t_max_y = float('inf')

    current_distance_to_boundary = 0.0

    while True:
        cell_being_exited_was_valid = (0 <= map_x < maze_grid_width) and \
                                      (0 <= map_y < maze_grid_height)
        
        advancing_in_x: bool
        if t_max_x < t_max_y:
            current_distance_to_boundary = t_max_x
            advancing_in_x = True
        else:
            current_distance_to_boundary = t_max_y
            advancing_in_x = False

        if current_distance_to_boundary > MAX_SENSOR_RANGE:
            # Ray exceeded max range before hitting a wall or exiting grid within range
            return ray_origin_world + ray_dir_normalized * MAX_SENSOR_RANGE

        # Check for wall hit if the cell we are currently in (and about to exit) is valid
        if cell_being_exited_was_valid:
            # Use MAZE_Y_IDX_FLIP_CONST for y-indexing transformation
            # This assumes map_y is 0-indexed from bottom, and maze array is 0-indexed from top.
            current_cell_y_idx = 15 - map_y
            
            # Ensure current_cell_y_idx is valid before accessing maze
            if not (0 <= current_cell_y_idx < maze_grid_height):
                 # This case should ideally not be hit if map_y was valid.
                 # Indicates an issue with MAZE_Y_IDX_FLIP_CONST or coordinate setup.
                 # If ray starts outside valid map_y range for indexing, treat as exiting grid.
                 return ray_origin_world + ray_dir_normalized * current_distance_to_boundary


            current_cell = maze[current_cell_y_idx][map_x]

            if advancing_in_x:
                if ray_step_x > 0: # Moving Right
                    if current_cell.wall_right:
                        return ray_origin_world + ray_dir_normalized * current_distance_to_boundary
                elif ray_step_x < 0: # Moving Left
                    if current_cell.wall_left:
                        return ray_origin_world + ray_dir_normalized * current_distance_to_boundary
            else: # Advancing in Y
                if ray_step_y > 0: # Moving "Up" in world coords (positive Y)
                                   # Corresponds to checking 'top' wall of cell in a typical grid depiction
                                   # or 'bottom' wall if map_y increases upwards.
                                   # Your code: ray_step_y > 0 means moving "Down" (screen coordinates)
                                   # maze[15-map_y] means lower map_y is higher on screen.
                                   # So ray_step_y > 0 means map_y increases, moving "down" in array, "up" in world
                                   # wall_top in your original code is correct for moving "down" (map_y increases) in world.
                                   # Your current code has: Moving Down (map_y increases) -> wall_top of (map_x, map_y)
                                   # Let's stick to your wall naming: wall_top of (map_x, map_y) means the wall at its +Y side (if Y is up).
                                   # If ray_dir_normalized[1] > 0 (ray moves towards +Y), map_y increases. Check wall_top.
                                   # If ray_dir_normalized[1] < 0 (ray moves towards -Y), map_y decreases. Check wall_bottom.
                                   # Original: ray_step_y > 0 => maze[...].wall_top
                                   # ray_step_y is based on ray_dir_normalized[1].
                                   # if ray_dir_normalized[1] > 0, step_y = 1 (map_y increases, moving "up" in world coords) -> check wall_top of cell (map_x, map_y)
                                   # if ray_dir_normalized[1] < 0, step_y = -1 (map_y decreases, moving "down" in world coords) -> check wall_bottom of cell (map_x, map_y)
                        if current_cell.wall_top: # Wall at the positive Y face of cell (map_x, map_y)
                             return ray_origin_world + ray_dir_normalized * current_distance_to_boundary
                elif ray_step_y < 0: # Moving "Down" in world coords (negative Y)
                        if current_cell.wall_bottom: # Wall at the negative Y face of cell (map_x, map_y)
                            return ray_origin_world + ray_dir_normalized * current_distance_to_boundary
        
        # Advance to next cell
        if advancing_in_x:
            map_x += ray_step_x
            t_max_x += t_delta_x
        else:
            map_y += ray_step_y
            t_max_y += t_delta_y

        # After stepping, if the new cell (map_x, map_y) is outside the grid,
        # the ray has exited. The current_distance_to_boundary was the distance to this exit line.
        # This distance is guaranteed to be <= MAX_SENSOR_RANGE due to the check at the loop start.
        if not ((0 <= map_x < maze_grid_width) and (0 <= map_y < maze_grid_height)):
            return ray_origin_world + ray_dir_normalized * current_distance_to_boundary

def update_tof_sensor_data_ray_marching(
    pos: np.ndarray,
    rot: float,
    maze: list[list[Cell]], # Again, Numba compatibility for Cell objects
    noise_factor: float = 0.005 # 0.5% of distance for noise std dev
):
    """
    Updates ToF sensor data using fast ray marching.
    Returns a list of 5 np.ndarrays, each representing the (noisy) intersection point [x, y]
    or [np.nan, np.nan] if no intersection or out of range.
    """
    noisy_sensor_points = [np.array([np.nan, np.nan]) for _ in range(5)]

    # Angle offsets for FRONT, LEFT, RIGHT, FRONT_LEFT_45, FRONT_RIGHT_45
    ANGLE_OFFSETS = np.array([0, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4]) + np.pi/2
    MAX_SENSOR_RANGES = np.array([255.0 / SCALE] * 5)  # Same range for all sensors
    MAX_SENSOR_RANGES[0] = 5 * 255.0 / SCALE  # Front sensor has a longer range

    for i in range(5): # For each of the 5 ToF sensors
        angle = rot + ANGLE_OFFSETS[i]
        
        # Ray direction based on your convention:
        # angle = 0 (robot front) => (0, -1) (Negative Y)
        # angle = pi/2 (robot left) => (1, 0) (Positive X)
        ray_dir = np.array([math.sin(angle), -math.cos(angle)])

        intersection_point = fast_ray_march(
            pos, ray_dir, maze, MAX_SENSOR_RANGES[i]
        )
        
        if not np.isnan(intersection_point[0]):
            distance = np.linalg.norm(intersection_point - pos)
            if distance > 1e-9 : # Add noise if there's a meaningful distance
                # Apply noise to the intersection point
                # Noise magnitude proportional to distance
                noise_std_dev = noise_factor * distance
                # Generate 2D Gaussian noise
                noise = np.random.normal(0, noise_std_dev, 2)
                noisy_sensor_points[i] = intersection_point + noise
            else:
                noisy_sensor_points[i] = intersection_point # No noise if distance is zero

    return noisy_sensor_points

NOISE_MAG = 0.05

class Action(Enum):
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2

class Mouse(Body):
    image: pg.Surface

    def __init__(self, uber):
        super().__init__(shape=Circle(radius=CELL_SIZE // 3), density=0.001, static=False)
        self.u = uber.mission

        self.pos = np.array([CELL_SIZE / 2, CELL_SIZE / 2])

        self.left_pwm = 0
        self.right_pwm = 0

        self.left_rpm = 0.0
        self.right_rpm = 0.0
        self.input_left_pwm = 0.0
        self.input_right_pwm = 0.0

        self.cell_x = 0
        self.cell_y = 0

        self.rot = np.pi / 2

        self.sensor_data = [np.zeros(2) for _ in range(5)]

        self.image = pg.image.load("sprite.png").convert_alpha()

    def handle_input(self, keys):
        self.input_left_pwm = 0.0
        self.input_right_pwm = 0.0

        if keys[Qt.Key_W]:
            self.input_left_pwm = self.u.wasd_move_pwm
            self.input_right_pwm = self.u.wasd_move_pwm
        if keys[Qt.Key_S]:
            self.input_left_pwm = -self.u.wasd_move_pwm
            self.input_right_pwm = -self.u.wasd_move_pwm
        if keys[Qt.Key_A]:
            self.input_left_pwm = -self.u.wasd_turn_pwm
            self.input_right_pwm = self.u.wasd_turn_pwm
        if keys[Qt.Key_D]:
            self.input_left_pwm = self.u.wasd_turn_pwm
            self.input_right_pwm = -self.u.wasd_turn_pwm

    # def is_wall_front(self): 
    #     front_tof = self.sensor_data[TOFDirection.FRONT.value]
    #     if front_tof is not None:
    #         distance = np.linalg.norm(front_tof - self.pos)
    #         return distance < 0.8 * CELL_SIZE
        
    # def is_wall_left(self):
    #     left_tof = self.sensor_data[TOFDirection.LEFT.value]
    #     if left_tof is not None:
    #         distance = np.linalg.norm(left_tof - self.pos)
    #         return distance < 0.8 * CELL_SIZE
        
    # def is_wall_right(self):
    #     right_tof = self.sensor_data[TOFDirection.RIGHT.value]
    #     if right_tof is not None:
    #         distance = np.linalg.norm(right_tof - self.pos)
    #         return distance < 0.8 * CELL_SIZE
        
    # def rot_to_direction(self):
    #     angle = self.rot % (2 * np.pi)
    #     if angle < np.pi / 4 or angle > 7 * np.pi / 4:
    #         return Direction.TOP
    #     elif angle > np.pi / 4 and angle < 3 * np.pi / 4:
    #         return Direction.LEFT
    #     elif angle > 3 * np.pi / 4 and angle < 5 * np.pi / 4:
    #         return Direction.BOTTOM
    #     else:
    #         return Direction.RIGHT

    # def move_forward_one_cell(self):
    #     if self.is_wall_front():
    #         print("Cannot move forward, wall detected!")
    #         return
        
    #     if self.target_cell is not None:
    #         return
        
    #     direction = self.rot_to_direction()
    #     if direction == Direction.TOP:
    #         self.target_cell = (self.cell_x, self.cell_y - 1)
    #     elif direction == Direction.RIGHT:
    #         self.target_cell = (self.cell_x + 1, self.cell_y)
    #     elif direction == Direction.BOTTOM:
    #         self.target_cell = (self.cell_x, self.cell_y + 1)
    #     else:
    #         self.target_cell = (self.cell_x - 1, self.cell_y)

    #     self.start_front_tof = np.linalg.norm(self.pos - self.sensor_data[TOFDirection.FRONT.value]) 

    # def turn_left(self):
    #     if self.target_cell is not None:
    #         return
        
    #     self.target_theta = (self.est_theta + np.pi / 2) % (2 * np.pi)

    # def turn_right(self):
    #     if self.target_cell is not None:
    #         return
        
    #     self.target_theta = (self.est_theta - np.pi / 2) % (2 * np.pi)

    def update(self, dt, renderer):
        ensure_transformed_shape(self)

        #print("Is wall front:", self.is_wall_front())
        #print("Is wall left:", self.is_wall_left())
        #print("Is wall right:", self.is_wall_right())

        self.cell_x = int((self.pos[0] - WALL_THICKNESS) // CELL_SIZE)
        self.cell_y = 15 - int((self.pos[1] - WALL_THICKNESS) // CELL_SIZE)

        #print(f"Cell: ({self.cell_x}, {self.cell_y})")

        max_rpm = self.u.rpm_at_100_pwm
        tau = 0.5  # seconds to reach ~100% rpm
        decay_rate = max_rpm / tau  # linear fall

        def _compute_rpm_single(effective_pwm, current_rpm):
            """
            Computes the new RPM for a single motor based on the effective PWM and current RPM.
            """
            decay_rate = max_rpm / tau  # Linear decay rate when PWM is 0

            if effective_pwm == 0:
                # No input: linear decay toward zero
                if current_rpm > 0:
                    return max(0.0, current_rpm - decay_rate * dt)
                elif current_rpm < 0:
                    return min(0.0, current_rpm + decay_rate * dt)
                else:
                    return 0.0
            else:
                # PWM is non-zero: Exponential rise/fall toward target_rpm
                pwm_frac = effective_pwm / 100.0 # Normalize PWM to -1.0 to 1.0
                
                # Target RPM calculation:
                # The magnitude of target RPM is proportional to the square of pwm_frac.
                # The sign of target RPM matches the sign of effective_pwm.
                # This gives finer control at lower PWM values.
                # e.g., pwm_frac = 0.5 (50% PWM) => (0.5)^2 = 0.25 => 25% of max_rpm
                # e.g., pwm_frac = -0.5 (-50% PWM) => (-0.5)^2 = 0.25 => -25% of max_rpm
                
                target_rpm_magnitude_factor = pwm_frac**2 # This is always positive

                if effective_pwm > 0:
                    target_rpm = max_rpm * target_rpm_magnitude_factor
                else: # effective_pwm < 0
                    target_rpm = -max_rpm * target_rpm_magnitude_factor
                
                # Exponential approach to target_rpm:
                # new_rpm = target_rpm + (current_rpm - target_rpm) * exp(-dt / tau)
                # This can be rewritten as:
                # new_rpm = target_rpm * (1 - exp(-dt/tau)) + current_rpm * exp(-dt/tau)
                # Your formulation is: target_rpm - (target_rpm - current_rpm) * math.exp(-dt / tau)
                # Let exp_val = math.exp(-dt / tau).
                # new_rpm = target_rpm - target_rpm * exp_val + current_rpm * exp_val
                # new_rpm = target_rpm * (1 - exp_val) + current_rpm * exp_val. This is correct.
                return target_rpm - (target_rpm - current_rpm) * math.exp(-dt / tau)

        combined_left_pwm = self.left_pwm + self.input_left_pwm
        combined_right_pwm = self.right_pwm + self.input_right_pwm

        clamped_left_pwm = max(-100.0, min(100.0, combined_left_pwm))
        clamped_right_pwm = max(-100.0, min(100.0, combined_right_pwm))

        self.left_rpm = _compute_rpm_single(clamped_left_pwm, self.left_rpm)
        self.right_rpm = _compute_rpm_single(clamped_right_pwm, self.right_rpm)

        print(f"Left 🛞 RPM: {self.left_rpm:.1f}, Right 🛞 RPM: {self.right_rpm:.1f}, left pwm: {clamped_left_pwm}, right pwm: {clamped_right_pwm}") 
        
        wr = self.u.wheel_radius * SCALE
        wb = self.u.wheel_base * SCALE

        # Convert to m/s and rad/s
        circ = 2 * math.pi * wr
        v = (self.left_rpm + self.right_rpm) / 2 * circ / 60
        omega = (self.right_rpm - self.left_rpm) * circ / (60 * wb)

        angle = self.rot
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])
        v_world = rotation_matrix @ np.array([v, 0])

        self.vel = v_world.copy()
        self.ang_vel = omega

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

                c.bias = -position_bias_factor * 1 / (dt + 1e-6) * min(0, -c.collision_depth + allowed_penetration)

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

        # self.update_with_oracle(dt)
        self.sensor_data = update_tof_sensor_data_ray_marching(self.pos, self.rot, renderer.maze)


    # def update_with_oracle(self, dt):
    #     left_rpm = self.left_rpm + random.gauss(0, 1) * NOISE_MAG
    #     right_rpm = self.right_rpm + random.gauss(0, 1) * NOISE_MAG

    #     #print(f"Left 🛞 RPM: {left_rpm:.1f}, Right 🛞 RPM: {right_rpm:.1f}")

    #     wr = self.u.wheel_radius * SCALE
    #     wb = self.u.wheel_base * SCALE
        
    #     # Reason about v and omega 
    #     slip_factor = 0.95  # Empirical value (0.9-1.0)
    #     measured_v = slip_factor * (left_rpm + right_rpm) / 2 * (2 * math.pi * wr) / 60
    #     if abs(measured_v) < 0.001:  # Threshold for "stopped"
    #         measured_v = 0
    #     measured_omega = (right_rpm - left_rpm) / wb * (2 * math.pi * wr) / 60
        
    #     # Simulate IMU measurement (angular velocity with noise)
    #     imu_omega = self.ang_vel + random.gauss(0, 0.1) * NOISE_MAG   # rad/s noise

        #imu_theta = self.rot + random.gauss(0, 0.05) * NOISE_MAG    # rad noise (if IMU has orientation)
        #imu_theta = (imu_theta + np.pi) % (2 * np.pi) - np.pi  # Convert to [-π, π]

        # print(f"IMU Omega: {imu_omega}, IMU Theta: {imu_theta}, True Theta: {self.rot}, True Omega: {self.ang_vel}", "measured_v:", measured_v, "measured_omega:", measured_omega, "true_v:", self.true_v)

        # Update Kalman filter with measurements

        # Using TOF for position measurement, i.e. helping the Kalman filter
        # measured_x = self.pos[0] - STARTING_POS_OFFSET[0] + random.gauss(0, 1) * NOISE_MAG
        # measured_y = -(self.pos[1] - STARTING_POS_OFFSET[1]) + random.gauss(0, 1) * NOISE_MAG

        # self.kalman.update_position(measured_x, measured_y)

    def draw(self, surface):
        rotated_image = pg.transform.rotate(self.image, math.degrees(-self.rot - np.pi / 2))
        rotated_rect = rotated_image.get_rect(center=self.pos)
        surface.blit(rotated_image, rotated_rect)
        
        #est_dir = np.array([math.cos(est_theta), math.sin(est_theta)]) * 20
        #draw_pos = np.array([est_x, est_y])

        #pg.draw.circle(surface, (0, 255, 0), (int(draw_pos[0]), int(draw_pos[1])), 8)
        #pg.draw.line(surface, (0, 255, 0), (int(draw_pos[0]), int(draw_pos[1])),
        #             (int((draw_pos[0] + est_dir[0])), int((draw_pos[1] + est_dir[1]))), 2)

        for dir, point in enumerate(self.sensor_data):
            color = (255, 0, 0) if dir == 0 else (0, 255, 0)  # Front sensor in red, others in green
            pg.draw.line(surface, color, self.pos, point, 1)

class PgRenderer:
    width: int = MAZE_SIZE
    height: int = MAZE_SIZE

    surface: pg.Surface = pg.Surface((width, height))
    maze: list[list[Cell]] = [[Cell(x, y) for x in range(CELL_COUNT)] for y in range(CELL_COUNT)]
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = None

    mouse: Mouse

    clock: pg.time.Clock = pg.time.Clock()
    pressed_keys: list[bool] = [False] * 512

    def __init__(self, uber):
        self.uber = uber

        pg.init()
        pg.display.set_mode((1, 1))

        self.mouse = Mouse(uber)

        for y in range(CELL_COUNT):
            for x in range(CELL_COUNT):
                cell = self.maze[y][x]

                wall_top, wall_right, wall_bottom, wall_left = get_walls_from_maze_for(x, y)
                cell.wall_top = wall_top
                cell.wall_right = wall_right
                cell.wall_bottom = wall_bottom
                cell.wall_left = wall_left


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

    # def toggle_wall(self, x, y, direction, drag_id=None):
    #     offset_x, offset_y = OFFSETS[direction]
    #     neighbor_x, neighbor_y = x + offset_x, y + offset_y

    #     cell = self.maze[y][x]

    #     if not (0 <= neighbor_x < CELL_COUNT and 0 <= neighbor_y < CELL_COUNT):
    #         neighbor = None
    #     else:
    #         neighbor = self.maze[neighbor_y][neighbor_x]

    #     wall_attrs = {
    #         Direction.TOP:    ("wall_top",    "wall_bottom"),
    #         Direction.RIGHT:  ("wall_right",  "wall_left"),
    #         Direction.BOTTOM: ("wall_bottom", "wall_top"),
    #         Direction.LEFT:   ("wall_left",   "wall_right"),
    #     }

    #     cell_attr, neighbor_attr = wall_attrs[direction]

    #     # Toggle the wall
    #     wall_state = getattr(cell, cell_attr) or getattr(neighbor, neighbor_attr) if neighbor is not None else False
    #     setattr(cell, cell_attr, not wall_state)
    #     if neighbor is not None:
    #         setattr(neighbor, neighbor_attr, False)

    #     self.refresh_walls()

    # def mouse_pressed_in_game(self, x, y, drag_id=None):
    #     # Convert to maze coordinates
    #     x, xr = divmod(x - WALL_THICKNESS, CELL_SIZE)
    #     y, yr = divmod(y - WALL_THICKNESS, CELL_SIZE)

    #     if 0 <= x < CELL_COUNT and 0 <= y < CELL_COUNT:
    #         if xr < yr:
    #             if xr > CELL_SIZE // 2:
    #                 self.toggle_wall(x, y, Direction.RIGHT, drag_id)
    #             else:
    #                 self.toggle_wall(x, y, Direction.LEFT, drag_id)
    #         else:
    #             if yr > CELL_SIZE // 2:
    #                 self.toggle_wall(x, y, Direction.BOTTOM, drag_id)
    #             else:
    #                 self.toggle_wall(x, y, Direction.TOP, drag_id)
    #         self.refresh_walls()

    def keyPressEvent(self, event):
        key = event.key()
        if key < len(self.pressed_keys):
            self.pressed_keys[key] = True

    def keyReleaseEvent(self, event):
        key = event.key()
        if key < len(self.pressed_keys):
            self.pressed_keys[key] = False

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
                y = 15 - y
                cx = x * CELL_SIZE + WALL_THICKNESS + WALL_THICKNESS // 2
                cy = y * CELL_SIZE + WALL_THICKNESS + WALL_THICKNESS // 2

                if cell.wall_top:
                    add_wall(pg.Rect(cx, cy + CELL_SIZE - WALL_THICKNESS // 2, CELL_SIZE, WALL_THICKNESS))
                if cell.wall_right:
                    add_wall(pg.Rect(cx + CELL_SIZE - WALL_THICKNESS // 2, cy, WALL_THICKNESS, CELL_SIZE))
                if cell.wall_bottom:
                    add_wall(pg.Rect(cx, cy - WALL_THICKNESS // 2, CELL_SIZE, WALL_THICKNESS))
                if cell.wall_left:
                    add_wall(pg.Rect(cx - WALL_THICKNESS // 2, cy, WALL_THICKNESS, CELL_SIZE))
                
    def update(self, dt):
        self.mouse.handle_input(self.pressed_keys)
        self.mouse.update(dt, self)

    def render(self):
        self.surface.fill((0, 0, 0))

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

        self.surface = pg.transform.flip(self.surface, False, True)

        return self.surface


def pg_surface_to_qimage(surface):
    data = pg.image.tostring(surface, "RGB")
    return QImage(data, surface.get_width(), surface.get_height(), QImage.Format_RGB888)

