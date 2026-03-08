import heapq
import math
from typing import List, Tuple, Dict, Set

# ============================================================================
# A* ALGORITHM IMPLEMENTATION
# ============================================================================

class Node:
    """Represents a node in the search space."""
    def __init__(self, position: Tuple[int, int], parent=None, g_cost=0, h_cost=0):
        self.position = position
        self.parent = parent
        self.g_cost = g_cost  # Cost from start node
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


class AStarPathfinder:
    """Generic A* pathfinding algorithm."""
    
    def __init__(self, heuristic_func, get_neighbors_func, movement_cost_func=None):
        self.heuristic = heuristic_func
        self.get_neighbors = get_neighbors_func
        self.movement_cost = movement_cost_func or (lambda x, y: 1)
    
    def find_path(self, start: Tuple, goal: Tuple):
        """
        Find path from start to goal using A* algorithm.
        Returns: list of positions from start to goal, or empty list if no path exists
        """
        open_list = []
        closed_set = set()
        
        start_node = Node(start, None, 0, self.heuristic(start, goal))
        heapq.heappush(open_list, start_node)
        
        visited_order = []
        
        while open_list:
            current_node = heapq.heappop(open_list)
            
            if current_node.position in closed_set:
                continue
            
            closed_set.add(current_node.position)
            visited_order.append(current_node.position)
            
            # Check if goal reached
            if current_node.position == goal:
                return self._reconstruct_path(current_node), visited_order
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_node.position):
                if neighbor_pos in closed_set:
                    continue
                
                new_g_cost = current_node.g_cost + self.movement_cost(current_node.position, neighbor_pos)
                new_h_cost = self.heuristic(neighbor_pos, goal)
                new_node = Node(neighbor_pos, current_node, new_g_cost, new_h_cost)
                
                heapq.heappush(open_list, new_node)
        
        return [], visited_order  # No path found
    
    def _reconstruct_path(self, node):
        """Reconstruct path from goal node to start."""
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]


# ============================================================================
# EXAMPLE 1: MAZE NAVIGATION
# ============================================================================

class MazeNavigator:
    """Solves maze navigation using A* algorithm with enhanced visualization."""
    
    WALL = 1
    PATH = 0
    
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if maze else 0
    
    def heuristic(self, pos, goal):
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_neighbors(self, pos):
        """Get valid adjacent cells (4-directional movement)."""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.maze[nx][ny] == self.PATH:
                neighbors.append((nx, ny))
        return neighbors
    
    def solve(self, start, goal):
        """Solve maze using A* algorithm."""
        pathfinder = AStarPathfinder(self.heuristic, self.get_neighbors)
        path, visited = pathfinder.find_path(start, goal)
        return path, visited
    
    def display_maze_enhanced(self, path=None, visited=None):
        """Enhanced maze visualization with ASCII art."""
        display = [['█' if cell == self.WALL else ' ' for cell in row] for row in self.maze]
        
        # Mark visited cells
        if visited:
            for pos in visited:
                if path is None or pos not in path:
                    if display[pos[0]][pos[1]] == ' ':
                        display[pos[0]][pos[1]] = '·'
        
        # Mark path
        if path and len(path) > 0:
            for pos in path[1:-1]:
                display[pos[0]][pos[1]] = '•'
            
            # Mark start and goal
            display[path[0][0]][path[0][1]] = 'S'
            display[path[-1][0]][path[-1][1]] = 'G'
        
        # Add borders and display
        lines = []
        lines.append("┌" + "─" * (len(display[0]) * 2) + "┐")
        for row in display:
            lines.append("│ " + " ".join(row) + " │")
        lines.append("└" + "─" * (len(display[0]) * 2) + "┘")
        
        return "\n".join(lines)
    
    def get_maze_stats(self, path, visited):
        """Calculate detailed statistics about the maze solution."""
        stats = {
            'path_length': len(path) if path else 0,
            'cells_visited': len(visited),
            'efficiency': (len(path) / len(visited) * 100) if visited else 0,
        }
        return stats
    
    def display_maze(self, path=None, visited=None):
        """Display maze with path and visited cells."""
        display = [row[:] for row in self.maze]
        
        if visited:
            for pos in visited:
                if pos not in [path[0]] if path else []:
                    display[pos[0]][pos[1]] = '.'
        
        if path:
            for pos in path[1:-1]:
                display[pos[0]][pos[1]] = '*'
            display[path[0][0]][path[0][1]] = 'S'
            display[path[-1][0]][path[-1][1]] = 'G'
        
        result = "\n".join([''.join(str(cell) for cell in row) for row in display])
        return result


# ============================================================================
# EXAMPLE 2: DELIVERY ROUTE OPTIMIZATION
# ============================================================================

class DeliveryOptimizer:
    """Optimizes delivery routes using A* algorithm."""
    
    def __init__(self, graph: Dict[str, Dict[str, float]]):
        """
        graph: Dictionary of locations with distances to neighbors
        Example: {'A': {'B': 5, 'C': 10}, 'B': {'A': 5, 'C': 3}, ...}
        """
        self.graph = graph
        self.locations = set(graph.keys())
    
    def heuristic(self, pos, goal):
        """
        Straight-line distance heuristic (assuming positions are tuples).
        For delivery, we use a simplified heuristic.
        """
        if pos == goal:
            return 0
        return 1  # Simplified heuristic
    
    def get_neighbors(self, pos):
        """Get connected delivery locations."""
        return list(self.graph.get(pos, {}).keys())
    
    def movement_cost(self, from_pos, to_pos):
        """Get distance between locations."""
        return self.graph.get(from_pos, {}).get(to_pos, float('inf'))
    
    def find_route(self, start: str, goal: str):
        """Find optimal delivery route using A*."""
        pathfinder = AStarPathfinder(self.heuristic, self.get_neighbors, self.movement_cost)
        path, visited = pathfinder.find_path(start, goal)
        return path, visited
    
    def calculate_total_distance(self, path):
        """Calculate total distance for a path."""
        if len(path) < 2:
            return 0
        total = 0
        for i in range(len(path) - 1):
            total += self.graph[path[i]][path[i + 1]]
        return total


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def example_1_maze():
    """Example 1: Maze Navigation with A*."""
    print("\n" + "="*80)
    print(" "*20 + "EXAMPLE 1: MAZE NAVIGATION WITH A*")
    print("="*80)
    
    # Define simple maze
    maze = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (4, 4)
    
    # Solve maze
    navigator = MazeNavigator(maze)
    path, visited = navigator.solve(start, goal)
    stats = navigator.get_maze_stats(path, visited)
    
    # Display results
    print(f"\n📍 Maze Details:")
    print(f"   • Size: {navigator.rows}x{navigator.cols}")
    print(f"   • Start Position: {start}")
    print(f"   • Goal Position: {goal}")
    print(f"   • Heuristic: Manhattan Distance")
    print(f"   • Movement: 4-directional (Up, Down, Left, Right)")
    
    if path and len(path) > 0:
        print(f"\n✓ PATH FOUND!")
        print(f"\n📊 Algorithm Statistics:")
        print(f"   • Steps to reach goal: {stats['path_length']}")
        print(f"   • Total cells explored: {stats['cells_visited']}")
        print(f"   • Algorithm efficiency: {stats['efficiency']:.1f}%")
        
        print(f"\n📍 Optimal Path (Step by Step):")
        for i, pos in enumerate(path):
            if i == 0:
                print(f"   Step {i}: {pos} [START]")
            elif i == len(path) - 1:
                print(f"   Step {i}: {pos} [GOAL]")
            else:
                print(f"   Step {i}: {pos}")
        
        print(f"\n🗺️  MAZE VISUALIZATION:")
        print(f"   Legend: S = Start, G = Goal, • = Path, · = Explored, █ = Wall")
        print()
        print(navigator.display_maze_enhanced(path, visited))
    else:
        print("\n✗ NO PATH FOUND!")
        print("The maze has no solution with the given start and goal positions.")



def example_2_delivery():
    """Example 2: Delivery Route Optimization."""
    print("\n" + "="*60)
    print("EXAMPLE 2: DELIVERY ROUTE OPTIMIZATION")
    print("="*60)
    
    # Create delivery network with distances
    delivery_graph = {
        'Warehouse': {'CityA': 50, 'CityB': 80},
        'CityA': {'Warehouse': 50, 'CityB': 30, 'CityC': 60},
        'CityB': {'Warehouse': 80, 'CityA': 30, 'CityD': 40},
        'CityC': {'CityA': 60, 'CityD': 25},
        'CityD': {'CityB': 40, 'CityC': 25, 'CustomerX': 35},
        'CustomerX': {'CityD': 35}
    }
    
    start = 'Warehouse'
    goal = 'CustomerX'
    
    optimizer = DeliveryOptimizer(delivery_graph)
    route, visited = optimizer.find_route(start, goal)
    total_distance = optimizer.calculate_total_distance(route)
    
    print(f"\nStart: {start}, Destination: {goal}")
    print(f"Route found: {route is not None and len(route) > 0}")
    print(f"Route length: {len(route) if route else 0}")
    print(f"Locations visited: {len(visited)}")
    
    if route:
        print(f"\nOptimal Delivery Route:")
        print(" → ".join(route))
        print(f"\nTotal Distance: {total_distance} km")
        
        print("\nBreakdown:")
        for i in range(len(route) - 1):
            dist = delivery_graph[route[i]][route[i + 1]]
            print(f"  {route[i]} → {route[i + 1]}: {dist} km")


def main():
    """Main interface with user options."""
    while True:
        print("\n" + "="*60)
        print("A* ALGORITHM - PATHFINDING EXAMPLES")
        print("="*60)
        print("\nChoose an example to view:")
        print("1. Example 1: Maze Navigation")
        print("2. Example 2: Delivery Route Optimization")
        print("3. Run Both Examples")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            example_1_maze()
        elif choice == '2':
            example_2_delivery()
        elif choice == '3':
            example_1_maze()
            example_2_delivery()
        elif choice == '4':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
