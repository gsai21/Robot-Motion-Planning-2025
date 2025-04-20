import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from heapq import heappop, heappush
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.ndimage import distance_transform_edt

# Constants
K = 15
NS = 2000

# Generate a 2D map: 0 = free, 1 = obstacle
def generate_map(size=1, obstacle_ratio=0.1, seed=42):
    np.random.seed(seed)
    grid = np.zeros((size, size))

    num_blobs = int(size * size * obstacle_ratio / 20)
    for _ in range(num_blobs):
        cx, cy = np.random.randint(10, size-10, size=2)
        radius = np.random.randint(3, 8)
        for i in range(max(0, cx-radius), min(size, cx+radius+1)):
            for j in range(max(0, cy-radius), min(size, cy+radius+1)):
                if np.sqrt((i-cx)*2 + (j-cy)*2) <= radius:
                    grid[i, j] = 1

    num_obstacles = int(obstacle_ratio * size * size * 0.3)
    for _ in range(num_obstacles):
        x, y = np.random.randint(0, size, size=2)
        grid[x, y] = 1

    return grid

def compute_distance_from_obstacles(grid):
    return distance_transform_edt(grid == 0)

def obstacle_distance_sampling(grid, dist_map, ns):
    free_coords = np.argwhere(grid == 0)
    dist_values = dist_map[grid == 0]
    dist_max = np.max(dist_values)

    pdist = np.exp(8 * (dist_max - dist_values) / dist_max) - 1
    pdist /= pdist.sum()

    indices = np.random.choice(len(free_coords), size=ns, p=pdist)
    samples = free_coords[indices]
    return samples

def is_collision_free(v1, v2, dist_map, grid):
    v1_int = v1.astype(int)
    v2_int = v2.astype(int)

    if grid[v1_int[0], v1_int[1]] == 1 or grid[v2_int[0], v2_int[1]] == 1:
        return False

    x0, y0 = int(v1[0]), int(v1[1])
    x1, y1 = int(v2[0]), int(v2[1])
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        if 0 <= x0 < grid.shape[0] and 0 <= y0 < grid.shape[1]:
            if grid[x0, y0] == 1:
                return False
        else:
            return False

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True

def build_roadmap(samples, dist_map, grid, k=K):
    tree = KDTree(samples)
    edges = {}
    all_edges = []

    for i, sample in enumerate(samples):
        edges[i] = []
        dists, indices = tree.query(sample, k=k + 1)
        for j, idx in enumerate(indices[1:]):
            if is_collision_free(sample, samples[idx], dist_map, grid):
                edges[i].append(idx)
                all_edges.append((sample, samples[idx]))

    return edges, all_edges

def dijkstra(samples, edges, start_idx, goal_idx):
    queue = [(0, start_idx, [])]
    visited = set()
    visited_nodes = []
    path_history = []
    node_indices_history = []

    while queue:
        cost, node, path = heappop(queue)

        if node in visited:
            continue

        visited.add(node)
        visited_nodes.append(samples[node])
        path = path + [node]
        node_indices_history.append(path.copy())
        current_path = [samples[i] for i in path]
        path_history.append(current_path.copy())

        if node == goal_idx:
            return current_path, path_history, visited_nodes, node_indices_history, cost

        for neighbor in edges[node]:
            if neighbor not in visited:
                edge_cost = np.linalg.norm(samples[node] - samples[neighbor])
                heappush(queue, (cost + edge_cost, neighbor, path))

    return [], path_history, visited_nodes, [], 0

def path_length(path):
    return sum(np.linalg.norm(path[i] - path[i+1]) for i in range(len(path)-1))

def optimize_path(path, dist_map, grid):
    if not path:
        return [], []

    optimization_history = [path.copy()]
    i = len(path) - 1
    optimized = [path[i]]

    while i > 0:
        for j in range(i - 1, -1, -1):
            if is_collision_free(path[j], path[i], dist_map, grid):
                optimized.append(path[j])
                i = j
                temp_path = optimized.copy()
                temp_path.reverse()
                optimization_history.append(temp_path.copy())
                break
        else:
            i -= 1
    optimized.reverse()

    def smooth_path(pts):
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(pts) - 2:
                if is_collision_free(pts[i], pts[i+2], dist_map, grid):
                    pts.pop(i+1)
                    changed = True
                else:
                    i += 1
            if changed:
                optimization_history.append(pts.copy())
        return pts

    final_path = smooth_path(optimized)
    return final_path, optimization_history

def point_in_list(point, point_list):
    return any(np.array_equal(point, p) for p in point_list)

def main():
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    grid = generate_map(size=100, obstacle_ratio=0.3)
    dist_map = compute_distance_from_obstacles(grid)

    start, goal = np.array([5, 5]), np.array([90, 90])
    samples = obstacle_distance_sampling(grid, dist_map, NS)
    samples = np.vstack([start, goal, samples])

    edges, all_edges = build_roadmap(samples, dist_map, grid, k=K)
    path, path_history, visited_nodes, node_indices_history, initial_cost = dijkstra(samples, edges, 0, 1)

    if not path:
        print("No path found!")
        return

    opt_path, opt_history = optimize_path(path, dist_map, grid)
    initial_path_length = path_length(path)
    optimized_path_length = path_length(opt_path)

    print(f"Initial path distance: {initial_path_length:.2f}")
    print(f"Optimized path distance: {optimized_path_length:.2f}")
    print(f"Distance reduction: {initial_path_length - optimized_path_length:.2f} ({(1 - optimized_path_length/initial_path_length)*100:.1f}%)")

    animation_data = []
    animation_data.append({
        'title': "Map and Sampling",
        'samples': samples[2:],
        'edges': [],
        'path': [],
        'visited': [],
        'text': "Obstacle-biased sampling"
    })

    roadmap_edges = []
    batch_size = max(1, len(all_edges) // 10)
    for i in range(0, len(all_edges), batch_size):
        roadmap_edges.extend(all_edges[i:i+batch_size])
        animation_data.append({
            'title': "Roadmap Construction",
            'samples': samples[2:],
            'edges': roadmap_edges.copy(),
            'path': [],
            'visited': [],
            'text': f"Building PRM edges: {min(len(roadmap_edges), len(all_edges))}/{len(all_edges)}"
        })

    visited_so_far = []
    visit_batch = max(1, len(visited_nodes) // 20)
    for i in range(0, len(visited_nodes), visit_batch):
        batch = visited_nodes[i:i+visit_batch]
        visited_so_far.extend(batch)

        current_path = []
        for idx, path in enumerate(path_history):
            if len(path) > 0 and point_in_list(path[-1], visited_so_far):
                if len(path) > len(current_path):
                    current_path = path

        animation_data.append({
            'title': "Path Finding",
            'samples': samples[2:],
            'edges': all_edges,
            'path': current_path,
            'visited': visited_so_far.copy(),
            'text': f"Exploring: {len(visited_so_far)}/{len(visited_nodes)} nodes visited"
        })

    animation_data.append({
        'title': "Initial Path Found",
        'samples': samples[2:],
        'edges': all_edges,
        'path': path,
        'visited': visited_nodes,
        'text': f"Initial path length: {initial_path_length:.2f}"
    })

    for i, opt_step in enumerate(opt_history):
        animation_data.append({
            'title': "Path Optimization",
            'samples': samples[2:],
            'edges': all_edges,
            'path': path,
            'optimized_path': opt_step,
            'visited': [],
            'text': f"Optimization step {i+1}/{len(opt_history)}"
        })

    animation_data.append({
        'title': "Final Optimized Path",
        'samples': samples[2:],
        'edges': all_edges,
        'path': path,
        'optimized_path': opt_path,
        'visited': [],
        'text': f"Initial: {initial_path_length:.2f}, Optimized: {optimized_path_length:.2f} ({(1 - optimized_path_length/initial_path_length)*100:.1f}% reduction)"
    })

    def update(frame):
        ax.clear()
        data = animation_data[frame]
        ax.imshow(grid.T, origin='lower', cmap='gray_r')  # white = free, black = obstacle

        if len(data['samples']) > 0:
            sample_x, sample_y = zip(*data['samples'])
            ax.scatter(sample_y, sample_x, s=2, c='lightblue', alpha=0.6)

        if len(data['edges']) > 0:
            edge_lines = [((v1[1], v1[0]), (v2[1], v2[0])) for v1, v2 in data['edges']]
            lc = LineCollection(edge_lines, colors='lightgray', linewidths=0.5, alpha=0.3)
            ax.add_collection(lc)

        if len(data['visited']) > 0:
            visited_x, visited_y = zip(*data['visited'])
            ax.scatter(visited_y, visited_x, s=5, c='yellow', alpha=0.7)

        if len(data['path']) > 0:
            px, py = zip(*data['path'])
            style = 'b--' if 'optimized_path' in data else 'b-'
            alpha = 0.3 if 'optimized_path' in data else 1
            ax.plot(py, px, style, linewidth=2, alpha=alpha)

        if 'optimized_path' in data and len(data['optimized_path']) > 0:
            ox, oy = zip(*data['optimized_path'])
            ax.plot(oy, ox, 'r-', linewidth=2.5)

        ax.plot(start[1], start[0], 'ro', markersize=8)
        ax.plot(goal[1], goal[0], 'go', markersize=8)
        ax.set_title(data['title'])
        ax.text(5, 5, data['text'], color='black', backgroundcolor='white', fontsize=9)

        handles = [
            mpatches.Patch(color='lightblue', alpha=0.6, label='Sampling Points'),
            mpatches.Patch(color='yellow', alpha=0.7, label='Explored Nodes'),
            mpatches.Patch(color='blue', alpha=0.7, label='Initial Path'),
            mpatches.Patch(color='red', alpha=0.7, label='Optimized Path'),
            mpatches.Patch(color='red', label='Start'),
            mpatches.Patch(color='green', label='Goal')
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=8)
        ax.set_xlim(0, grid.shape[1])
        ax.set_ylim(0, grid.shape[0])
        return ax

    anim = FuncAnimation(fig, update, frames=len(animation_data), interval=500, blit=False)
    
    # Save animation as MP4 (requires ffmpeg)
    # anim.save("prm_path_planning.mp4", fps=2, extra_args=['-vcodec', 'libx264'])

    plt.tight_layout()
    plt.show()
    return anim

animation = main()