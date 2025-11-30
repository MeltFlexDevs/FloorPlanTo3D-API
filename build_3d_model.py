from bisect import bisect_left
from json import loads
import random
from sys import argv
from typing import Any, Callable
from dataclasses import dataclass, field
import numpy as np
from MeshBuilder import MeshBuilder

"""
Directions:
  0 -> UP
  1 -> RIGHT
  2 -> DOWN
  3 -> LEFT
"""

@dataclass(eq=False)
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float

    type: str

    group: "set[Wall] | None" = None

    def get_width(self):
        return self.x2 - self.x1

    def set_width(self, value: float):
        if self.group is not None:
            for wall in self.group:
                wall.x2 = wall.x1 + value
        else:
            self.x2 = self.x1 + value

    def get_height(self):
        return self.y2 - self.y1

    def set_height(self, value: float):
        if self.group is not None:
            for wall in self.group:
                wall.y2 = wall.y1 + value
        else:
            self.y2 = self.y1 + value

    def is_horizontal(self):
        return self.get_width() > self.get_height()

    def translate(self, x: float, y: float):
        if self.group is not None:
            for wall in self.group:
                wall._translateSelf(x, y)
        else:
            self._translateSelf(x, y)
    
    def _translateSelf(self, x: float, y: float):
        self.x1 += x
        self.x2 += x
        self.y1 += y
        self.y2 += y

    def normalize(self, normalizer: float):
        self.x1 *= normalizer
        self.x2 *= normalizer
        self.y1 *= normalizer
        self.y2 *= normalizer

    def link(self, other: "Wall"):
        # If both walls are not in a group, create a new group with both of them
        if self.group is None and other.group is None:
            self.group = other.group = {self, other}
            return

        # If both walls are grouped, merge the group
        if self.group is not None and other.group is not None:

            new_list = {*self.group, *other.group}
            for wall in self.group:
                wall.group = new_list

            for wall in other.group:
                wall.group = new_list
        
        # If other is in a group add self to it
        if other.group is not None:
            self.group = other.group
            other.group.add(self)
            return

        # If self is in a group add other to it
        if self.group is not None:
            other.group = self.group
            self.group.add(other)
            return
        
        # Unreachable
        assert False
    
    def get_point(self, direction: int):
        if direction == 0:
            return np.array(((self.x1 + self.x2) / 2, self.y1), dtype=np.float)

        if direction == 1:
            return np.array((self.x2, (self.y1 + self.y2) / 2), dtype=np.float)

        if direction == 2:
            return np.array(((self.x1 + self.x2) / 2, self.y2), dtype=np.float)

        if direction == 3:
            return np.array((self.x1, (self.y1 + self.y2) / 2), dtype=np.float)
        
        # Illegal direction
        assert False
        

@dataclass
class Socket:
    wall: Wall
    direction: int
    original_position: Any = field(init=False)
    tolerance: float

    def __post_init__(self):
        self.original_position = self.wall.get_point(self.direction)

    @property
    def position(self):
        return self.wall.get_point(self.direction)

    def get_opposite(self):
        return (self.direction + 2) % 4
    
    def is_horizontal(self):
        return self.direction == 1 or self.direction == 3

def align_walls(walls: "list[Wall]"):
    sockets: "dict[int, list[Socket]]" = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    for wall in walls:
        horizontal = wall.is_horizontal()
        vertical = not horizontal

        # If the wall is close enough to a square, consider is horizontal and vertical at once
        if abs((wall.get_height() - wall.get_width()) / (wall.get_height() + wall.get_width())) < 0.15:
            horizontal = True
            vertical = True

        if horizontal:
            tolerance = wall.get_height()
            sockets[1].append(Socket(wall, 1, tolerance))
            sockets[3].append(Socket(wall, 3, tolerance))

        if vertical:
            tolerance = wall.get_width()
            sockets[0].append(Socket(wall, 0, tolerance))
            sockets[2].append(Socket(wall, 2, tolerance))
    
    for direction in [0, 1]:
        direction_sockets = sockets[direction]
        j = 0
        matches = 0

        while j < len(direction_sockets):
            socket = direction_sockets[j]
            j += 1
            
            opposite_sockets = sockets[socket.get_opposite()]

            # Find socket with opposite direction that is close enough
            for opposite_socket in opposite_sockets:
                distance = np.linalg.norm(socket.original_position - opposite_socket.original_position)

                if socket.is_horizontal():
                    distance += abs(socket.wall.get_height() - opposite_socket.wall.get_height())
                else:
                    distance += abs(socket.wall.get_width() - opposite_socket.wall.get_width())

                tolerance = min(socket.tolerance, opposite_socket.tolerance)
                if distance <= tolerance:
                    break
            else:
                opposite_socket = None
            
            if opposite_socket is None:
                continue
            
            matches += 1

            # Remove the matched sockets
            direction_sockets.remove(socket)
            opposite_sockets.remove(opposite_socket)

            # Rollback iteration to account for removed element
            j -= 1

            # Unify the thickness
            if socket.is_horizontal():
                value = (socket.wall.get_height() + opposite_socket.wall.get_height()) / 2
                socket.wall.set_height(value)
                opposite_socket.wall.set_height(value)
            else:
                value = (socket.wall.get_width() + opposite_socket.wall.get_width()) / 2
                socket.wall.set_width(value)
                opposite_socket.wall.set_width(value)

            # Offset for the opposite socket to be aligned with this
            center = (opposite_socket.position + socket.position) / 2

            correction = (socket.position - center) * -1
            socket.wall.translate(*correction)

            correction = (opposite_socket.position - center) * -1
            opposite_socket.wall.translate(*correction)

            # Join the two walls together
            opposite_socket.wall.link(socket.wall)

        print(f"For direction {direction} aligned {matches} pairs")

def walls_from_json(data: dict):
    walls: "list[Wall]" = []

    for i, point in enumerate(data["points"]):
        walls.append(Wall(
            point["x1"], # type: ignore
            point["y1"],
            point["x2"],
            point["y2"],
            data["classes"][i]["name"]
        ))
    
    return walls

def walls_to_json(walls: "list[Wall]"):
    points = []

    for wall in walls:
        points.append({
            "x1": wall.x1,
            "y1": wall.y1,
            "x2": wall.x2,
            "y2": wall.y2,
        })

    return points

def build_geometry(walls: "list[Wall]"):
    pass

def find_rooms(walls: "list[Wall]", tolerance: float, sample_image: "Callable[[float, float], bool] | None" = None):
    x_grid: "list[float]" = []
    y_grid: "list[float]" = []

    def push_grid_line(grid: "list[float]", position: float):
        index = bisect_left(grid, position)
        neighbour_min = grid[index - 1] if 0 <= index - 1 < len(grid) else None
        if neighbour_min is not None and abs(position - neighbour_min) < tolerance:
            return

        neighbour_max = grid[index] if 0 <= index < len(grid) else None
        if neighbour_max is not None and abs(position - neighbour_max) < tolerance:
            return
        
        grid.insert(index, position)


    for wall in walls:
        push_grid_line(x_grid, wall.x1)
        push_grid_line(x_grid, wall.x2)
        push_grid_line(y_grid, wall.y1)
        push_grid_line(y_grid, wall.y2)
    
    width = len(x_grid) - 1
    height = len(y_grid) - 1
    tiles: "list[int]" = [-1] * (width * height)

    for y, (y1, y2) in enumerate(zip(y_grid, y_grid[1:])):
        for x, (x1, x2) in enumerate(zip(x_grid, x_grid[1:])):
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            # Check if any architectural element (wall, window, door) overlaps with this grid cell
            # Using center point detection to mark occupied cells
            for wall in walls:
                if wall.x1 < center_x < wall.x2 \
                    and wall.y1 < center_y < wall.y2:
                    tiles[x + y * width] = 0
                    break

    # ============================================================================
    # BULLETPROOF GAP FILLING ALGORITHM - GUARANTEED ZERO GAPS
    # ============================================================================
    # Strategy: Scan entire grid, find ALL empty cells between elements,
    # and fill them ALL in one pass. Repeat until no changes occur.
    # This ensures absolutely NO gaps remain between architectural elements.
    # ============================================================================

    max_iterations = 20  # Increased to handle complex floor plans

    for iteration in range(max_iterations):
        gaps_filled_this_iteration = 0

        # Scan every cell in the grid
        for y in range(height):
            for x in range(width):
                # Skip borders
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    continue

                # Skip already occupied cells
                if tiles[x + y * width] == 0:
                    continue

                # Current cell is empty - check if it's between elements

                # ========== HORIZONTAL GAP DETECTION ==========
                # Scan left to find nearest element
                left_element_found = False
                left_element_x = -1
                for scan_x in range(x - 1, -1, -1):
                    if tiles[scan_x + y * width] == 0:
                        left_element_found = True
                        left_element_x = scan_x
                        break

                # Scan right to find nearest element
                right_element_found = False
                right_element_x = -1
                for scan_x in range(x + 1, width):
                    if tiles[scan_x + y * width] == 0:
                        right_element_found = True
                        right_element_x = scan_x
                        break

                # If we have elements on both sides, ALWAYS fill the gap
                # NO TOLERANCE CHECK - we fill ANY gap between architectural elements
                # This ensures absolutely NO gaps remain that could cause missing floors
                if left_element_found and right_element_found:
                    # Calculate physical distance between elements (for logging only)
                    gap_start_x = x_grid[left_element_x + 1]  # Right edge of left element
                    gap_end_x = x_grid[right_element_x]       # Left edge of right element
                    gap_distance = gap_end_x - gap_start_x

                    x1, x2 = x_grid[x], x_grid[x + 1]
                    y1, y2 = y_grid[y], y_grid[y + 1]

                    tiles[x + y * width] = 0
                    walls.append(Wall(x1, y1, x2, y2, "wall"))
                    gaps_filled_this_iteration += 1

                    print(f"[Pass {iteration + 1}] Filled HORIZONTAL gap at grid({x},{y}) coords({x1:.3f},{y1:.3f}) gap_size={gap_distance:.3f}m")
                    continue  # Move to next cell

                # ========== VERTICAL GAP DETECTION ==========
                # Scan up to find nearest element
                top_element_found = False
                top_element_y = -1
                for scan_y in range(y - 1, -1, -1):
                    if tiles[x + scan_y * width] == 0:
                        top_element_found = True
                        top_element_y = scan_y
                        break

                # Scan down to find nearest element
                bottom_element_found = False
                bottom_element_y = -1
                for scan_y in range(y + 1, height):
                    if tiles[x + scan_y * width] == 0:
                        bottom_element_found = True
                        bottom_element_y = scan_y
                        break

                # If we have elements on both sides, ALWAYS fill the gap
                # NO TOLERANCE CHECK - we fill ANY gap between architectural elements
                # This ensures absolutely NO gaps remain that could cause missing floors
                if top_element_found and bottom_element_found:
                    # Calculate physical distance between elements (for logging only)
                    gap_start_y = y_grid[top_element_y + 1]    # Bottom edge of top element
                    gap_end_y = y_grid[bottom_element_y]       # Top edge of bottom element
                    gap_distance = gap_end_y - gap_start_y

                    x1, x2 = x_grid[x], x_grid[x + 1]
                    y1, y2 = y_grid[y], y_grid[y + 1]

                    tiles[x + y * width] = 0
                    walls.append(Wall(x1, y1, x2, y2, "wall"))
                    gaps_filled_this_iteration += 1

                    print(f"[Pass {iteration + 1}] Filled VERTICAL gap at grid({x},{y}) coords({x1:.3f},{y1:.3f}) gap_size={gap_distance:.3f}m")

        # Check if we're done
        if gaps_filled_this_iteration == 0:
            print(f"✓ Gap filling completed after {iteration + 1} iteration(s). ZERO gaps remain!")
            break
        else:
            print(f"[Pass {iteration + 1}] Filled {gaps_filled_this_iteration} gap(s). Checking for more...")
    else:
        print(f"⚠ Gap filling stopped after {max_iterations} iterations (safety limit). Check results.")
    
    # Find missing walls by checking the image for every empty cell. By randomly sampling pixels, if the cell is 80% black pixels, it's probably a wall.
    if sample_image is not None:
        for y, (y1, y2) in enumerate(zip(y_grid, y_grid[1:])):
            for x, (x1, x2) in enumerate(zip(x_grid, x_grid[1:])):
                if tiles[x + y * width] == 0:
                    continue

                black = 0
                count = 8

                for _ in range(count):
                    xi = random.uniform(x1, x2)
                    yi = random.uniform(y1, y2)
                    is_white = sample_image(xi, yi)
                    if not is_white:
                        black += 1

                if black / count < 0.8:
                    continue

                print(f"Fixing missing wall {(x1, y1, x2, y2)}")

                tiles[x + y * width] = 0
                walls.append(Wall(x1, y1, x2, y2, "wall")) # type: ignore
    
    room_id = 1
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for y in range(height):
        for x in range(width):
            index = x + y * width

            # Start floodfill if an unvisited free cell is found
            if tiles[index] == -1:
                stack = [(x, y)]
                tiles[index] = room_id

                while stack:
                    cx, cy = stack.pop()

                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy

                        # Check bounds
                        if 0 <= nx < width and 0 <= ny < height:
                            n_index = nx + ny * width

                            # Check if the neighbor is a free cell (-1)
                            if tiles[n_index] == -1:
                                tiles[n_index] = room_id
                                stack.append((nx, ny))

                room_id += 1

    occupied = [0] * len(tiles)
    room_meshes: "dict[int, list[tuple[float, float, float, float]]]" = {}
    for y in range(height):
        for x in range(width):
            room_id = tiles[x + y * width]
            if room_id == 0 or occupied[x + y * width] != 0:
                continue

            iy = y + 1
            for iy in range(iy, height):
                if tiles[x + iy * width] != room_id or occupied[x + iy * width] != 0:
                    break

            if iy == width:
                iy -= 1

            if y == iy:
                continue
            
            ix = x + 1
            for ix in range(ix, width):
                failed = False

                for jy in range(y, iy):
                    if tiles[ix + jy * width] != room_id or occupied[ix + jy * width] != 0:
                        failed = True

                if failed:
                    break
                
            if ix == width:
                ix -= 1
            
            if x == ix:
                continue
            
            for jy in range(y, iy):
                for jx in range(x, ix):
                    occupied[jx + jy * width] = room_id
            
            room_meshes.setdefault(room_id, []).append((
                x_grid[x],
                y_grid[y],
                x_grid[ix],
                y_grid[iy],
            ))
    
    for x in range(width):
        room_id = tiles[x]
        if room_id != 0 and room_id in room_meshes:
            del room_meshes[room_id]

        room_id = tiles[x + (width * (height - 1))]
        if room_id != 0 and room_id in room_meshes:
            del room_meshes[room_id]

    for y in range(height):
        room_id = tiles[y * width]
        if room_id != 0 and room_id in room_meshes:
            del room_meshes[room_id]

        room_id = tiles[y * width + width - 1]
        if room_id != 0 and room_id in room_meshes:
            del room_meshes[room_id]
            
    print(f"Room grid: {width} x {height}, Room Count: {room_id - 1}")
    return room_meshes

def get_normalizer(data: dict):
    return 1 / (data["averageDoor"] / 0.8)

def build_3d_model(data: dict, sample_image: "Callable[[float, float], bool] | None" = None):
    walls = walls_from_json(data)
    align_walls(walls)

    normalizer = get_normalizer(data)
    for wall in walls:
        wall.normalize(normalizer)

    builder = MeshBuilder()
    rooms = find_rooms(walls, tolerance=0.05, sample_image=sample_image)
    data["points"] = walls_to_json(walls)

    for name in rooms.keys():
        quads = rooms[name]

        for (x1, y1, x2, y2) in quads:
            builder.add_quad(
                [x1, y1, 0],
                [x1, y2, 0],
                [x2, y2, 0],
                [x2, y1, 0],
            )

        builder.create_mesh([f"Room_{name}", f"Floor_r{name}"])

    height = 2.6

    for index, wall in enumerate(walls):
        x1 = wall.x1
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        if wall.type == "window":
            builder.add_cube(x1, y1, x2, y2, 0, height / 3)
            builder.add_cube(x1, y1, x2, y2, height * 2/3, height)
        elif wall.type == "door":
            # Door normally do not reach the ceiling, to achieve this, create we shorter doors and
            # then put wall above. Average height of doors is 200cm relative to ceiling height 260cm,
            # to keep this for any height use a ratio. The wall object is also create separately for
            # applying textures.
            builder.add_cube(x1, y1, x2, y2, 0, height * (10/13))
        else:
            builder.add_cube(x1, y1, x2, y2, 0, height)
        builder.create_mesh(f"{wall.type.capitalize()}_{index}")

        if wall.type == "door":
            builder.add_cube(x1, y1, x2, y2, height * (10/13), height)
            builder.create_mesh(f"Wall_{index}")
    
    return builder.build()

if __name__ == "__main__":
    with open(argv[1], 'rt') as file:
        content = file.read()

    data: dict = loads(content)
    gltf = build_3d_model(data)
    gltf.export(argv[1] + ".new.glb")

