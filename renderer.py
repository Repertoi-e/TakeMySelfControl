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

class Cell:
    wall_top: bool = False
    wall_right: bool = False
    wall_bottom: bool = False
    wall_left: bool = False
    last_drag_ids: set = {
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

class Mouse(Body):
    image: pg.Surface

    def __init__(self, uber):
        super().__init__(shape=Circle(radius=CELL_SIZE // 3), density=0.001, static=False)
        self.u = uber.mission

        self.pos = np.array([0.5 * CELL_SIZE, 15.5 * CELL_SIZE])
        
        self.image = pg.image.load("sprite.png").convert_alpha()

    def handle_input(self, keys):
        if keys[pg.K_w]:
            # Move forward in the current facing direction
            force = np.array([0, -self.u.motor_force])

            angle = -self.rot
            rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]])
            force = np.dot(rotation_matrix, force)

            apply_force(self, force)

        if keys[pg.K_a]:
            self.torque += self.u.rotation_torque

        if keys[pg.K_d]:
            self.torque += -self.u.rotation_torque

    def update(self, dt, walls):
        ensure_transformed_shape(self)

        acc = self.force * self.inv_mass
        self.vel += acc * dt

        ang_acc = self.torque * self.inv_rot_inertia
        self.ang_vel += ang_acc * dt

        # Apply air drag
        self.ang_vel *= self.u.drag

        # Apply friction
        if self.vel[0] != 0 or self.vel[1] != 0:
            self.vel *= self.u.friction

        a = self
        for b in walls:
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


    def draw(self, surface):
        rotated_image = pg.transform.rotate(self.image, math.degrees(self.rot))
        rotated_rect = rotated_image.get_rect(center=self.pos)
        surface.blit(rotated_image, rotated_rect)

class PgRenderer:
    width: int = MAZE_SIZE
    height: int = MAZE_SIZE

    surface: pg.Surface = pg.Surface((width, height))
    maze: list[list[Cell]] = [[Cell() for _ in range(CELL_COUNT)] for _ in range(CELL_COUNT)]
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
        print(x, xr, y, yr)

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
    
        self.walls = []
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
        self.mouse.update(dt, self.walls)
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

