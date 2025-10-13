import numpy as np
import heapq
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.ndimage import binary_dilation

class PathPlanner:
    def __init__(self, occupancy_grid: np.ndarray, robot_radius: float = 0):
        """
        Initialize path planner with occupancy grid.
        
        Args:
            occupancy_grid: 2D numpy array where:
                           0 = free space
                           1 = occupied/obstacle
            robot_radius: Robot radius in grid cells for safety margin
        """
        self.original_grid = occupancy_grid.copy()
        self.robot_radius = robot_radius
        self.grid = self._inflate_obstacles(occupancy_grid, robot_radius)
        self.rows, self.cols = self.grid.shape
        self.current_path = None
        self.current_path_index = 0
        
    def _inflate_obstacles(self, grid: np.ndarray, radius: float) -> np.ndarray:
        """
        Inflate obstacles by robot radius for safety margin.
        
        Args:
            grid: Original occupancy grid
            radius: Inflation radius in grid cells
            
        Returns:
            Inflated occupancy grid
        """
        if radius <= 0:
            return grid.copy()
        
        # Create circular structuring element
        size = int(2 * radius + 1)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        structure = x**2 + y**2 <= radius**2
        
        # Dilate obstacles
        inflated = binary_dilation(grid, structure=structure)
        return inflated.astype(np.uint8)
    
    def update_grid(self, new_occupancy_grid: np.ndarray):
        """
        Update occupancy grid with new sensor data.
        
        Args:
            new_occupancy_grid: Updated occupancy grid from sensors
        """
        self.original_grid = new_occupancy_grid.copy()
        self.grid = self._inflate_obstacles(new_occupancy_grid, self.robot_radius)
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 8-connected neighbors"""
        row, col = pos
        neighbors = []
        
        # 8-connected grid (includes diagonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                # Check if cell is free
                if self.grid[new_row, new_col] == 0:
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find collision-free path using A* algorithm.
        
        Args:
            start: (row, col) starting position
            goal: (row, col) goal position
            
        Returns:
            List of (row, col) positions from start to goal, or None if no path exists
        """
        # Validate start and goal
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position out of bounds")
        if not (0 <= goal[0] < self.rows and 0 <= goal[1] < self.cols):
            raise ValueError("Goal position out of bounds")
        if self.grid[start[0], start[1]] == 1:
            raise ValueError("Start position is occupied")
        if self.grid[goal[0], goal[1]] == 1:
            raise ValueError("Goal position is occupied")
        
        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        
        # Track visited nodes
        came_from = {}
        
        # Cost from start to each node
        g_score = {start: 0}
        
        # Estimated total cost (g + h)
        f_score = {start: self.heuristic(start, goal)}
        
        # Set for quick lookup
        open_set_hash = {start}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Goal reached
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                # Calculate cost to neighbor
                # Diagonal movement costs sqrt(2), cardinal movement costs 1
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = np.sqrt(dx**2 + dy**2)
                
                tentative_g_score = g_score[current] + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return None
    
    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to goal"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def line_of_sight(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Check if there's a clear line of sight between two points using Bresenham's algorithm.
        
        Args:
            p1: First point (row, col)
            p2: Second point (row, col)
            
        Returns:
            True if line of sight is clear, False otherwise
        """
        x0, y0 = p1[1], p1[0]  # Convert to (col, row) for Bresenham
        x1, y1 = p2[1], p2[0]
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # Check if current position is occupied
            if self.grid[y0, x0] == 1:
                return False
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return True
    
    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary waypoints using line-of-sight checks.
        
        Args:
            path: Original path from A*
            
        Returns:
            Smoothed path with fewer waypoints
        """
        if not path or len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to find the farthest point with line of sight
            farthest_idx = current_idx + 1
            
            for i in range(len(path) - 1, current_idx, -1):
                if self.line_of_sight(path[current_idx], path[i]):
                    farthest_idx = i
                    break
            
            smoothed.append(path[farthest_idx])
            current_idx = farthest_idx
        
        return smoothed
    
    def is_path_valid(self, path: List[Tuple[int, int]]) -> bool:
        """
        Check if current path is still collision-free.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid, False if it intersects obstacles
        """
        if not path:
            return False
        
        for i in range(len(path) - 1):
            if not self.line_of_sight(path[i], path[i + 1]):
                return False
        
        return True
    
    def dynamic_replan(self, current_pos: Tuple[int, int], goal: Tuple[int, int], 
                       new_occupancy_grid: Optional[np.ndarray] = None,
                       replan_threshold: int = 5) -> Optional[List[Tuple[int, int]]]:
        """
        Dynamic replanning: check if current path is valid, replan if necessary.
        
        Args:
            current_pos: Current robot position
            goal: Goal position
            new_occupancy_grid: Updated occupancy grid (if available)
            replan_threshold: Replan if obstacle detected within this many waypoints ahead
            
        Returns:
            Updated path, or None if no path exists
        """
        # Update grid if new data available
        if new_occupancy_grid is not None:
            self.update_grid(new_occupancy_grid)
        
        # Check if we need to replan
        need_replan = False
        
        if self.current_path is None:
            need_replan = True
        elif not self.is_path_valid(self.current_path):
            need_replan = True
            print("Path blocked! Replanning...")
        else:
            # Check upcoming waypoints
            check_until = min(self.current_path_index + replan_threshold, len(self.current_path))
            upcoming_path = self.current_path[self.current_path_index:check_until]
            
            for waypoint in upcoming_path:
                if self.grid[waypoint[0], waypoint[1]] == 1:
                    need_replan = True
                    print("Obstacle detected ahead! Replanning...")
                    break
        
        # Replan if necessary
        if need_replan:
            raw_path = self.find_path(current_pos, goal)
            if raw_path:
                self.current_path = self.smooth_path(raw_path)
                self.current_path_index = 0
                print(f"New path planned with {len(self.current_path)} waypoints")
            else:
                self.current_path = None
                print("No valid path found!")
            return self.current_path
        
        return self.current_path
    
    def get_next_waypoint(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Get next waypoint from current path.
        
        Args:
            current_pos: Current robot position
            
        Returns:
            Next waypoint or None if at goal
        """
        if self.current_path is None or self.current_path_index >= len(self.current_path):
            return None
        
        # Update index if we've reached current waypoint
        current_waypoint = self.current_path[self.current_path_index]
        if current_pos == current_waypoint and self.current_path_index < len(self.current_path) - 1:
            self.current_path_index += 1
        
        return self.current_path[self.current_path_index]
    
    def visualize(self, path: Optional[List[Tuple[int, int]]] = None, 
                  smoothed_path: Optional[List[Tuple[int, int]]] = None,
                  start: Optional[Tuple[int, int]] = None,
                  goal: Optional[Tuple[int, int]] = None,
                  current_pos: Optional[Tuple[int, int]] = None,
                  show_inflation: bool = False):
        """Visualize the occupancy grid and paths"""
        plt.figure(figsize=(12, 10))
        
        # Display grid
        grid_to_show = self.grid if show_inflation else self.original_grid
        plt.imshow(grid_to_show, cmap='Greys', origin='upper', alpha=0.7)
        
        # Plot original path
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 1], path_array[:, 0], 'c--', 
                    linewidth=1, alpha=0.5, label='Original A* Path')
        
        # Plot smoothed path
        if smoothed_path:
            smoothed_array = np.array(smoothed_path)
            plt.plot(smoothed_array[:, 1], smoothed_array[:, 0], 'b-', 
                    linewidth=2.5, label='Smoothed Path')
            plt.plot(smoothed_array[:, 1], smoothed_array[:, 0], 'bo', 
                    markersize=6)
        
        # Plot current position
        if current_pos:
            plt.plot(current_pos[1], current_pos[0], 'mo', 
                    markersize=12, label='Current Position')
        
        # Plot start and goal
        if start:
            plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
        if goal:
            plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
        
        plt.xlabel('Column')
        plt.ylabel('Row')
        title = 'Robot Path Planning'
        if show_inflation:
            title += f' (with {self.robot_radius} cell safety margin)'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage with all features
if __name__ == "__main__":
    # Create sample occupancy grid (60x60)
    grid_size = 60
    occupancy_grid = np.zeros((grid_size, grid_size))
    
    # Add some obstacles
    occupancy_grid[10:45, 25] = 1  # Vertical wall
    occupancy_grid[30, 25:50] = 1  # Horizontal wall
    occupancy_grid[15:25, 40] = 1  # Another obstacle
    occupancy_grid[35:45, 10:15] = 1  # Block obstacle
    
    # Define start and goal
    start = (5, 5)
    goal = (55, 55)
    
    # Create planner with safety margin (robot radius = 2 cells)
    print("=" * 50)
    print("INITIAL PLANNING WITH SAFETY MARGIN")
    print("=" * 50)
    planner = PathPlanner(occupancy_grid, robot_radius=2)
    
    # Find initial path
    raw_path = planner.find_path(start, goal)
    
    if raw_path:
        print(f"Raw path found with {len(raw_path)} waypoints")
        
        # Smooth the path
        smoothed_path = planner.smooth_path(raw_path)
        print(f"Smoothed path has {len(smoothed_path)} waypoints")
        print(f"Reduction: {len(raw_path) - len(smoothed_path)} waypoints removed")
        
        # Visualize
        planner.visualize(raw_path, smoothed_path, start, goal, show_inflation=True)
        
        # Store current path for dynamic replanning
        planner.current_path = smoothed_path
        
        # Simulate robot movement and dynamic replanning
        print("\n" + "=" * 50)
        print("DYNAMIC REPLANNING SIMULATION")
        print("=" * 50)
        
        # Simulate: robot is at waypoint 5, new obstacle appears
        current_position = smoothed_path[min(5, len(smoothed_path)-1)]
        print(f"\nRobot at position: {current_position}")
        
        # Add new obstacle in the path
        new_grid = occupancy_grid.copy()
        new_grid[40:45, 45:50] = 1  # New obstacle blocking path
        print("New obstacle detected!")
        
        # Dynamic replan
        updated_path = planner.dynamic_replan(current_position, goal, new_grid)
        
        if updated_path:
            print(f"Replanned path has {len(updated_path)} waypoints")
            planner.visualize(None, updated_path, start, goal, 
                            current_position, show_inflation=True)
        
    else:
        print("No path found!")
        planner.visualize(start=start, goal=goal, show_inflation=True)