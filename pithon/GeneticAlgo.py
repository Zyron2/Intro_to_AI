import random
import math
from typing import List, Tuple, Callable, Any
from copy import deepcopy
import time

# ============================================================================
# GENETIC ALGORITHM IMPLEMENTATION
# ============================================================================

class GeneticAlgorithm:
    """
    Generic Genetic Algorithm for optimization problems.
    Inspired by natural evolution and Darwin's theory of natural selection.
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def evolve(self, create_individual: Callable, fitness_func: Callable,
               mutate_func: Callable, crossover_func: Callable):
        """
        Execute the genetic algorithm.
        
        Args:
            create_individual: Function to create random individual
            fitness_func: Function to evaluate fitness
            mutate_func: Function to mutate individual
            crossover_func: Function to perform crossover
        """
        # Create initial population
        self.population = [create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Evaluate fitness
            self.fitness_scores = [fitness_func(ind) for ind in self.population]
            
            # Track best individual
            best_idx = self.fitness_scores.index(max(self.fitness_scores))
            current_best_fitness = self.fitness_scores[best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = deepcopy(self.population[best_idx])
            
            self.fitness_history.append(self.best_fitness)
            
            # Selection & Reproduction
            new_population = []
            
            # Elitism: Keep best individual
            new_population.append(deepcopy(self.best_individual))
            
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = crossover_func(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = mutate_func(child1)
                if random.random() < self.mutation_rate:
                    child2 = mutate_func(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3):
        """Select individual using tournament selection."""
        tournament_idx = random.sample(range(self.population_size), tournament_size)
        best_tournament_idx = max(tournament_idx, key=lambda i: self.fitness_scores[i])
        return deepcopy(self.population[best_tournament_idx])


# ============================================================================
# EXAMPLE 1: TRAVELING SALESMAN PROBLEM (TSP)
# ============================================================================

class TravelingSalesmanProblem:
    """Solves TSP using Genetic Algorithm with distance optimization."""
    
    @staticmethod
    def calculate_distance(city1: Tuple, city2: Tuple) -> float:
        """Calculate Euclidean distance between two cities."""
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    
    @staticmethod
    def calculate_route_distance(route: List[int], cities: List[Tuple]) -> float:
        """Calculate total distance of a route."""
        distance = 0
        for i in range(len(route)):
            city1 = cities[route[i]]
            city2 = cities[route[(i + 1) % len(route)]]
            distance += TravelingSalesmanProblem.calculate_distance(city1, city2)
        return distance
    
    @staticmethod
    def create_individual(num_cities: int) -> List[int]:
        """Create random route."""
        route = list(range(num_cities))
        random.shuffle(route)
        return route
    
    @staticmethod
    def fitness_func(route: List[int], cities: List[Tuple]) -> float:
        """Fitness = 1/distance (higher is better)."""
        distance = TravelingSalesmanProblem.calculate_route_distance(route, cities)
        return 1 / (distance + 1e-10)  # Avoid division by zero
    
    @staticmethod
    def mutate_swap(route: List[int]) -> List[int]:
        """Swap mutation: swap two random cities."""
        mutated = route.copy()
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    @staticmethod
    def crossover_order(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order Crossover (OX): preserves city order."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy segment from parent
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Fill remaining cities
        def fill_child(child, parent, other_parent):
            idx = end
            parent_idx = end
            while -1 in child:
                if parent[parent_idx] not in child:
                    child[idx % size] = parent[parent_idx]
                    idx += 1
                parent_idx += 1
                if parent_idx >= size:
                    parent_idx = 0
        
        fill_child(child1, parent2, parent1)
        fill_child(child2, parent1, parent2)
        
        return child1, child2
    
    @staticmethod
    def run_example():
        """Run TSP example with visualization."""
        print("\n" + "="*90)
        print(" "*20 + "EXAMPLE 1: TRAVELING SALESMAN PROBLEM (TSP)")
        print("="*90)
        
        # Generate random cities
        num_cities = 10
        random.seed(42)
        cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_cities)]
        
        print(f"\n📍 PROBLEM SETUP:")
        print(f"   • Number of cities: {num_cities}")
        print(f"   • Map area: 100 x 100 units")
        print(f"   • Objective: Find shortest route visiting all cities exactly once")
        print(f"\n   City Coordinates:")
        for i, city in enumerate(cities):
            print(f"      City {i}: {city}")
        
        # Create fitness function for this problem
        def fitness_tsp(route):
            return TravelingSalesmanProblem.fitness_func(route, cities)
        
        # Create crossover function for this problem
        def crossover_tsp(p1, p2):
            return TravelingSalesmanProblem.crossover_order(p1, p2)
        
        # Run genetic algorithm
        print(f"\n🧬 GENETIC ALGORITHM EVOLUTION:")
        print(f"   • Population size: 50")
        print(f"   • Generations: 100")
        print(f"   • Mutation rate: 10%")
        print(f"   • Crossover rate: 70%")
        
        ga = GeneticAlgorithm(population_size=50, generations=100, 
                             mutation_rate=0.1, crossover_rate=0.7)
        
        ga.evolve(
            create_individual=lambda: TravelingSalesmanProblem.create_individual(num_cities),
            fitness_func=fitness_tsp,
            mutate_func=TravelingSalesmanProblem.mutate_swap,
            crossover_func=crossover_tsp
        )
        
        # Display results
        best_distance = TravelingSalesmanProblem.calculate_route_distance(ga.best_individual, cities)
        
        print(f"\n✓ OPTIMIZATION RESULTS:")
        print(f"   • Best route found: {ga.best_individual}")
        print(f"   • Total distance: {best_distance:.2f} units")
        print(f"   • Best fitness: {ga.best_fitness:.6f}")
        
        print(f"\n🗺️  ROUTE VISUALIZATION:")
        print(f"   Start → {' → '.join(map(str, ga.best_individual))} → Start")
        
        print(f"\n📊 EVOLUTION STATISTICS:")
        print(f"   • Initial best distance: {1/ga.fitness_history[0]:.2f} units")
        print(f"   • Final best distance: {best_distance:.2f} units")
        improvement = ((1/ga.fitness_history[0] - best_distance) / (1/ga.fitness_history[0])) * 100
        print(f"   • Improvement: {improvement:.1f}%")


# ============================================================================
# EXAMPLE 2: JOB SCHEDULING PROBLEM
# ============================================================================

class JobSchedulingProblem:
    """Solves job scheduling using Genetic Algorithm with completion time optimization."""
    
    @staticmethod
    def create_individual(jobs):
        """Create random job schedule."""
        schedule = list(range(len(jobs)))
        random.shuffle(schedule)
        return schedule
    
    @staticmethod
    def calculate_makespan(schedule: List[int], jobs: List[Tuple[int, int]], num_machines: int) -> int:
        """Calculate makespan (total completion time) for schedule."""
        machine_times = [0] * num_machines
        
        for job_idx in schedule:
            job_time, machine_affinity = jobs[job_idx]
            # Assign to machine with least work + consider affinity
            best_machine = 0
            min_time = machine_times[0]
            
            for m in range(num_machines):
                # Add affinity bonus
                adjusted_time = machine_times[m]
                if m % num_machines == machine_affinity:
                    adjusted_time *= 0.9  # 10% time reduction for preferred machine
                
                if adjusted_time < min_time:
                    min_time = adjusted_time
                    best_machine = m
            
            machine_times[best_machine] += job_time
        
        return max(machine_times)
    
    @staticmethod
    def fitness_func(schedule: List[int], jobs: List[Tuple[int, int]], num_machines: int) -> float:
        """Fitness = 1/makespan (higher is better)."""
        makespan = JobSchedulingProblem.calculate_makespan(schedule, jobs, num_machines)
        return 1 / (makespan + 1e-10)
    
    @staticmethod
    def mutate_swap(schedule: List[int]) -> List[int]:
        """Swap mutation: swap two random jobs."""
        mutated = schedule.copy()
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    @staticmethod
    def mutate_insert(schedule: List[int]) -> List[int]:
        """Insert mutation: remove job and insert at random position."""
        mutated = schedule.copy()
        idx = random.randint(0, len(mutated) - 1)
        job = mutated.pop(idx)
        new_idx = random.randint(0, len(mutated))
        mutated.insert(new_idx, job)
        return mutated
    
    @staticmethod
    def crossover_pmx(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Partially Mapped Crossover (PMX) for scheduling."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = parent1[start:end] + [x for x in parent2 if x not in parent1[start:end]]
        child2 = parent2[start:end] + [x for x in parent1 if x not in parent2[start:end]]
        
        return child1, child2
    
    @staticmethod
    def run_example():
        """Run job scheduling example."""
        print("\n" + "="*90)
        print(" "*15 + "EXAMPLE 2: JOB SCHEDULING OPTIMIZATION")
        print("="*90)
        
        # Define jobs: (processing_time, machine_preference)
        jobs = [
            (5, 0),   # Job 0: 5 units, prefers machine 0
            (8, 1),   # Job 1: 8 units, prefers machine 1
            (3, 0),   # Job 2: 3 units, prefers machine 0
            (6, 2),   # Job 3: 6 units, prefers machine 2
            (7, 1),   # Job 4: 7 units, prefers machine 1
            (4, 2),   # Job 5: 4 units, prefers machine 2
            (5, 0),   # Job 6: 5 units, prefers machine 0
            (9, 1),   # Job 7: 9 units, prefers machine 1
            (6, 2),   # Job 8: 6 units, prefers machine 2
            (4, 0),   # Job 9: 4 units, prefers machine 0
        ]
        num_machines = 3
        
        print(f"\n🏭 PROBLEM DESCRIPTION:")
        print(f"   A manufacturing company has {num_machines} production machines and {len(jobs)} jobs to complete.")
        print(f"   Each job requires a specific processing time and works best on certain machines.")
        print(f"   Goal: Find the optimal job assignment order to minimize the total completion time")
        print(f"   (makespan) - when the last job finishes across all machines.")
        
        print(f"\n⚙️  PROBLEM SETUP:")
        print(f"   • Number of jobs: {len(jobs)}")
        print(f"   • Number of machines: {num_machines}")
        print(f"   • Objective: Minimize total completion time (makespan)")
        print(f"\n   Job Details:")
        print(f"   {'Job':<5} {'Time':<8} {'Preferred Machine':<20}")
        print(f"   {'-'*33}")
        for i, (time, pref) in enumerate(jobs):
            print(f"   {i:<5} {time:<8} {pref:<20}")
        
        # Create fitness function for this problem
        def fitness_scheduling(schedule):
            return JobSchedulingProblem.fitness_func(schedule, jobs, num_machines)
        
        def crossover_scheduling(p1, p2):
            return JobSchedulingProblem.crossover_pmx(p1, p2)
        
        # Run genetic algorithm
        print(f"\n🧬 GENETIC ALGORITHM EVOLUTION:")
        print(f"   • Population size: 50")
        print(f"   • Generations: 150")
        print(f"   • Mutation rate: 15%")
        print(f"   • Crossover rate: 70%")
        
        ga = GeneticAlgorithm(population_size=50, generations=150, 
                             mutation_rate=0.15, crossover_rate=0.7)
        
        ga.evolve(
            create_individual=lambda: JobSchedulingProblem.create_individual(jobs),
            fitness_func=fitness_scheduling,
            mutate_func=lambda s: (JobSchedulingProblem.mutate_swap(s) if random.random() < 0.5 
                                  else JobSchedulingProblem.mutate_insert(s)),
            crossover_func=crossover_scheduling
        )
        
        # Display results
        best_makespan = JobSchedulingProblem.calculate_makespan(ga.best_individual, jobs, num_machines)
        
        print(f"\n✓ OPTIMIZATION RESULTS:")
        print(f"   • Best schedule: {ga.best_individual}")
        print(f"   • Makespan (total time): {best_makespan} time units")
        print(f"   • Best fitness: {ga.best_fitness:.6f}")
        
        # Show machine load with color coding
        print(f"\n🤖 MACHINE LOAD ANALYSIS:")
        machine_times = [0] * num_machines
        for job_idx in ga.best_individual:
            job_time, machine_affinity = jobs[job_idx]
            best_machine = 0
            min_time = machine_times[0]
            
            for m in range(num_machines):
                adjusted_time = machine_times[m]
                if m % num_machines == machine_affinity:
                    adjusted_time *= 0.9
                
                if adjusted_time < min_time:
                    min_time = adjusted_time
                    best_machine = m
            
            machine_times[best_machine] += job_time
        
        # Calculate average load for balance analysis
        avg_load = sum(machine_times) / num_machines
        max_load = max(machine_times)
        
        # ANSI color codes
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        RESET = '\033[0m'
        
        for m, time in enumerate(machine_times):
            bar = "█" * (time // 2)
            
            # Determine color based on utilization
            deviation = abs(time - avg_load)
            if deviation <= 2:
                color = GREEN
                status = "✓ BALANCED"
            elif deviation <= 4:
                color = YELLOW
                status = "⚠ SLIGHT IMBALANCE"
            else:
                if time > avg_load:
                    color = RED
                    status = "✗ OVERLOADED"
                else:
                    color = BLUE
                    status = "↓ UNDER-UTILIZED"
            
            print(f"   Machine {m}: {color}{bar}{RESET} {time:2d} units  [{status}]")
        
        print(f"\n📊 EVOLUTION STATISTICS:")
        initial_makespan = 1 / ga.fitness_history[0]
        print(f"   • Initial best makespan: {initial_makespan:.1f} time units")
        print(f"   • Final best makespan: {best_makespan} time units")
        improvement = ((initial_makespan - best_makespan) / initial_makespan) * 100
        print(f"   • Improvement: {improvement:.1f}%")


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main interface with user options."""
    while True:
        print("\n" + "="*90)
        print("GENETIC ALGORITHM - EVOLUTION-INSPIRED OPTIMIZATION")
        print("="*90)
        print("\nChoose an example to view:")
        print("1. Example 1: Traveling Salesman Problem (TSP)")
        print("2. Example 2: Job Scheduling Optimization")
        print("3. Run Both Examples")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            TravelingSalesmanProblem.run_example()
        elif choice == '2':
            JobSchedulingProblem.run_example()
        elif choice == '3':
            TravelingSalesmanProblem.run_example()
            print("\n" + "="*90)
            input("Press Enter to continue to Example 2...")
            JobSchedulingProblem.run_example()
        elif choice == '4':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1-4.")
        
        if choice in ['1', '2', '3']:
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
