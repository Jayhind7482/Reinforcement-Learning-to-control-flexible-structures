import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pickle
from scipy.spatial import cKDTree

random.seed(42) 
np.random.seed(42)

# Define the parameters
l = 1.0  # Length
h_min, h_max = -32,   32 # Range for h  ( h =32 , v =17)
v_min, v_max = -17 , 17
num_points = 80  # Increased number of grid points in each direction

# Create meshgrid for h and v
h_values = np.linspace(h_min, h_max, num_points)
v_values = np.linspace(v_min, v_max, num_points)
H, V = np.meshgrid(h_values, v_values)
grid_points = np.column_stack((H.ravel(), V.ravel()))

# Function to define the ODE
def ode_system(s, y, h, v):
    theta, theta_prime = y  # Unpack the solution vector
    dtheta_ds = theta_prime
    dtheta_prime_ds = h * np.sin(theta) - v * np.cos(theta)
    return np.array([dtheta_ds, dtheta_prime_ds])

# Boundary conditions function
def boundary_conditions(ya, yb):
    return np.array([ya[0], yb[1]])  # theta(0) = 0 and theta'(l) = 0

# Function to solve ODE for given h and v
def solve_bvp_for_hv(h, v, initial_guess):
    s = np.linspace(0, l, 500)  # Increased discretization points
    y_guess = initial_guess  # Use provided initial guess

    # Pass h and v directly to the ode_system
    solution = solve_bvp(lambda s, y: ode_system(s, y, h, v), boundary_conditions, s, y_guess, tol=1e-5)
    
    if solution.success:
        return solution.success, solution
    else:
        return False, None

# Directions for adjacent points (up, down, left, right, diagonals)
adjacent_indices = [(-1, 0), (1, 0), (0, -1), (0, 1),  # up, down, left, right
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

def get_adjacent_unsolved_points(index, solved_points):
    row, col = index // num_points, index % num_points
    unsolved_adjacent = []
    for dr, dc in adjacent_indices:
        adj_row, adj_col = row + dr, col + dc
        if 0 <= adj_row < num_points and 0 <= adj_col < num_points:
            adj_index = adj_row * num_points + adj_col
            if adj_index not in solved_points:
                unsolved_adjacent.append(adj_index)
    return unsolved_adjacent

# List to store solutions
solutions = []
s = np.linspace(0, l, 500)

# Solve for the closest point to (0, 0)
closest_index = np.argmin(np.linalg.norm(grid_points, axis=1))
h_closest, v_closest = grid_points[closest_index]
initial_guess_closest = np.zeros((2, len(s)))  # Initial guess: theta = 0, theta' = 0

bl, solution_closest = solve_bvp_for_hv(h_closest, v_closest, initial_guess_closest)

# Store solved points to avoid redundancy
solved_points = set()
solved_points_list = []

# Store the solution in dictionaries
theta_dict = {}
theta_prime_dict = {}
x_tip = []
y_tip = []  
valid_tips_count = 0
total_solved_points = 0

if bl:
    total_solved_points += 1  
    theta_dict[closest_index] = solution_closest.sol(s)[0]
    theta_prime_dict[closest_index] = solution_closest.sol(s)[1]   
    y1 = np.cos(theta_dict[closest_index])
    y2 = np.sin(theta_dict[closest_index])
    x = np.cumsum(np.trapezoid(y1, x=s))
    y = np.cumsum(np.trapezoid(y2, x=s))

    # Append final tip positions
    x_tip.append(x[-1])
    y_tip.append(y[-1])
    valid_tips_count += 1
    solutions.append((h_closest, v_closest, solution_closest))
    solved_points.add(closest_index)
    solved_points_list.append(closest_index)

# Parallelized solution for remaining points
def solve_for_index(i):
    h_adj, v_adj = grid_points[i]
    initial_guess_adjacent = np.zeros((2, len(s)))
    initial_guess_adjacent[0] = theta_dict[random.choice(list(theta_dict.keys()))]
    initial_guess_adjacent[1] = theta_prime_dict[random.choice(list(theta_prime_dict.keys()))]
    bl, solution_random = solve_bvp_for_hv(h_adj, v_adj, initial_guess_adjacent)
    if bl:
        y1 = np.cos(solution_random.sol(s)[0])
        y2 = np.sin(solution_random.sol(s)[0])
        x = np.cumsum(np.trapezoid(y1, x=s))
        y = np.cumsum(np.trapezoid(y2, x=s))
        
        return i, solution_random, x[-1], y[-1]
    return None

with ThreadPoolExecutor() as executor:
    while len(solved_points) < len(grid_points):
        # Pick a random solved point
        current_index = random.choice(list(solved_points))
        # Get its adjacent unsolved points
        unsolved_adjacent = get_adjacent_unsolved_points(current_index, solved_points)
        if not unsolved_adjacent:
            continue
        # Solve for adjacent points in parallel
        futures = [executor.submit(solve_for_index, i) for i in unsolved_adjacent]
        for future in futures:
            result = future.result()
            if result:
                i, solution_random, x_tip_val, y_tip_val = result
                total_solved_points += 1  
                theta_dict[i] = solution_random.sol(s)[0]
                theta_prime_dict[i] = solution_random.sol(s)[1]
                x_tip.append(x_tip_val)
                y_tip.append(y_tip_val)
                valid_tips_count += 1
                solutions.append((grid_points[i][0], grid_points[i][1], solution_random))
                solved_points.add(i)
                solved_points_list.append(i)

print(f"Total solved points: {len(solved_points)}")
print(f"Valid tip positions: {valid_tips_count}")
print(f"Length of x_tip: {len(x_tip)}")
print(f"Length of y_tip: {len(y_tip)}")
print(f"Length of theta_values: {len(theta_dict)}")
print(len(solved_points_list))

plt.scatter(x_tip, y_tip, color='g', s=10, )
x_box = [0.5, 0.9, 0.9, 0.5, 0.5]
y_box = [-0.4, -0.4, 0.4, 0.4, -0.4]
plt.plot(x_box, y_box, 'b-', label='Target Box')
plt.title('Scatter plot of the tip positions inside the target box')
plt.grid(True)
plt.legend()
plt.show()
x_min, x_max = min(x_box), max(x_box)
y_min, y_max = min(y_box), max(y_box)

# Count how many points are inside the box
inside_box_count = 0
for x, y in zip(x_tip, y_tip):
    if x_min <= x <= x_max and y_min <= y <= y_max:
        inside_box_count += 1

print(f"Number of points inside the box: {inside_box_count}")
coordinates = list(zip(x_tip, y_tip))

# Count occurrences of each coordinate
coordinate_counts = Counter(coordinates)

# Find duplicates
duplicates = {coord: count for coord, count in coordinate_counts.items() if count > 1}

# Output duplicate points and their counts
print(f"Number of duplicate points: {len(duplicates)}")
print("Duplicate points and their counts:", duplicates)

# After calculating all solutions, create a dictionary to store the results
cheat_sheet_data = {
    'grid_points': grid_points,
    'theta_values': theta_dict,
    'theta_prime_values': theta_prime_dict
}

# Create a KD-tree for efficient nearest neighbor search
kdtree = cKDTree(grid_points)
cheat_sheet_data['kdtree'] = kdtree

# Save the cheat sheet data to a file
with open('cheat_sheet_data.pkl', 'wb') as f:
    pickle.dump(cheat_sheet_data, f)

print("Cheat sheet data saved to cheat_sheet_data.pkl")
