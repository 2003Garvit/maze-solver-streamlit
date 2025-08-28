# app.py
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------- THEME & STYLES --------------
st.set_page_config(
    page_title="Maze Solver — Streamlit",
    page_icon="🌀",
    layout="wide",
)

# custom CSS for a unique neon vibe
st.markdown(
    """
    <style>
    :root {
        --bg: #0b0f19;
        --panel: #111726;
        --ink: #d7e1ff;
        --muted: #7a88b8;
        --accent: #8a7ef8;
        --accent2: #00f0ff;
        --success: #2fffa8;
        --danger: #ff6b6b;
        --warning: #ffd166;
    }
    .appview-container, .main, .block-container { background: var(--bg) !important; }
    .stMarkdown, .stText, .stSelectbox, .stButton, .stSlider, .stNumberInput, .stExpander { color: var(--ink) !important; }
    .css-1dp5vir, .st-emotion-cache-1r4qj8v, .st-emotion-cache-16idsys, .st-emotion-cache-13l9j5u, .st-emotion-cache-1d391kg {
        background: var(--panel) !important; border-radius: 16px;
    }
    .stButton>button {
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        color: #0d1021; border: none; border-radius: 999px; padding: 0.6rem 1rem; font-weight: 700;
        box-shadow: 0 6px 24px rgba(138,126,248,0.35);
    }
    .stButton>button:hover { filter: brightness(1.05); transform: translateY(-1px); }
    .pill {
        display:inline-block; padding: .25rem .6rem; border-radius: 999px; font-size: .8rem; font-weight: 700;
        color:#0b0f19; background: linear-gradient(90deg, var(--success), var(--accent2));
        margin-left:.4rem;
    }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; display:inline-block; margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------- UTILS --------------
Cell = Tuple[int, int]
Grid = np.ndarray

@dataclass
class Maze:
    grid: Grid  # 0 wall, 1 path
    start: Cell
    goal: Cell

def in_bounds(grid: Grid, r: int, c: int) -> bool:
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

def neighbors4(r: int, c: int) -> List[Cell]:
    return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

# -------------- MAZE GENERATION --------------
def make_empty(h: int, w: int) -> Grid:
    g = np.zeros((h, w), dtype=np.uint8)
    return g

def generate_maze_prims(h: int, w: int, seed: Optional[int]=None) -> Grid:
    """ Randomized Prim's algorithm to generate a perfect maze. """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # enforce odd dimensions for clean walls
    if h % 2 == 0: h += 1
    if w % 2 == 0: w += 1

    grid = np.zeros((h, w), dtype=np.uint8)  # 0 wall, 1 passage
    start_r, start_c = 1, 1
    grid[start_r, start_c] = 1

    walls = []
    for nr, nc in neighbors4(start_r, start_c):
        if in_bounds(grid, nr, nc):
            walls.append((nr, nc, start_r, start_c))

    while walls:
        idx = random.randrange(len(walls))
        wr, wc, pr, pc = walls.pop(idx)
        if not in_bounds(grid, wr, wc): 
            continue

        # check if wall divides two cells and one is unvisited
        if grid[wr, wc] == 0:
            # determine opposite cell
            dr = wr - pr
            dc = wc - pc
            opp_r = wr + dr
            opp_c = wc + dc

            if in_bounds(grid, opp_r, opp_c) and grid[opp_r, opp_c] == 0:
                grid[wr, wc] = 1
                grid[opp_r, opp_c] = 1

                for nr, nc in neighbors4(opp_r, opp_c):
                    if in_bounds(grid, nr, nc) and grid[nr, nc] == 0:
                        walls.append((nr, nc, opp_r, opp_c))

    # ensure borders are walls
    grid[0, :] = 0; grid[-1, :] = 0; grid[:, 0] = 0; grid[:, -1] = 0
    grid[1,1] = 1
    grid[-2,-2] = 1
    return grid

# -------------- SOLVERS --------------
def bfs_solve(grid: Grid, start: Cell, goal: Cell, yield_steps: bool = True):
    from collections import deque
    q = deque([start])
    came: Dict[Cell, Optional[Cell]] = {start: None}
    visited_order = []
    while q:
        cur = q.popleft()
        visited_order.append(cur)
        if cur == goal:
            break
        r, c = cur
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid, nr, nc): 
                continue
            if grid[nr, nc] == 0: 
                continue
            if (nr, nc) in came:
                continue
            came[(nr, nc)] = cur
            q.append((nr, nc))
            if yield_steps:
                yield ("visit", (nr, nc), dict(came))
    # reconstruct path
    path = []
    cur = goal
    if cur not in came:
        yield ("fail", None, dict(came))
        return
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    yield ("done", path, dict(came))

def dijkstra_solve(grid: Grid, start: Cell, goal: Cell, yield_steps: bool=True):
    import heapq
    pq = [(0, start)]
    dist: Dict[Cell, float] = {start: 0.0}
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    while pq:
        d, cur = heapq.heappop(pq)
        if cur == goal:
            break
        r, c = cur
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid, nr, nc) or grid[nr, nc] == 0:
                continue
            nd = d + 1
            if nd < dist.get((nr, nc), float("inf")):
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = cur
                heapq.heappush(pq, (nd, (nr, nc)))
                if yield_steps:
                    yield ("visit", (nr, nc), dict(prev))
    if goal not in prev:
        yield ("fail", None, dict(prev))
        return
    # reconstruct
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    yield ("done", path, dict(prev))

def a_star_solve(grid: Grid, start: Cell, goal: Cell, yield_steps: bool=True):
    import heapq
    def h(a: Cell, b: Cell) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = []
    heapq.heappush(open_set, (0 + h(start, goal), 0, start))
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    g_score: Dict[Cell, float] = {start: 0.0}

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            break
        r, c = current
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid, nr, nc) or grid[nr, nc] == 0:
                continue
            tentative_g = g + 1
            if tentative_g < g_score.get((nr, nc), float("inf")):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + h((nr, nc), goal)
                heapq.heappush(open_set, (f, tentative_g, (nr, nc)))
                if yield_steps:
                    yield ("visit", (nr, nc), dict(came_from))

    if goal not in came_from:
        yield ("fail", None, dict(came_from))
        return
    # reconstruct
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    yield ("done", path, dict(came_from))

SOLVERS = {
    "Breadth-First Search (BFS)": bfs_solve,
    "Dijkstra": dijkstra_solve,
    "A* (Manhattan)": a_star_solve,
}

# -------------- RENDERING --------------
def render_grid(grid: Grid, start: Cell, goal: Cell, visited: Optional[List[Cell]]=None, path: Optional[List[Cell]]=None, ax=None):
    """Render the maze using imshow with neon palette."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    # walls
    img[grid == 0] = np.array([0.05, 0.06, 0.12])
    # passages
    img[grid == 1] = np.array([0.09, 0.15, 0.28])

    # visited shimmer
    if visited:
        for r, c in visited:
            img[r, c] = np.array([0.30, 0.26, 0.65])  # purple

    # path highlight
    if path:
        for r, c in path:
            img[r, c] = np.array([0.00, 0.70, 0.62])  # teal

    # start/goal
    sr, sc = start
    gr, gc = goal
    img[sr, sc] = np.array([0.80, 1.00, 0.00])  # lime
    img[gr, gc] = np.array([1.00, 0.39, 0.39])  # red

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    return ax

# -------------- APP --------------
st.markdown("# 🌀 Maze Solver")
st.markdown(
    "A stylish, interactive **maze generator + pathfinding visualizer**.\
     Pick a maze size, choose an algorithm, and watch the solver animate in a neon cyber vibe. \
     Built with Streamlit. <span class='pill'>Deploy-ready</span>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("Tip: larger mazes + slower speed = dramatic effect ✨")

    cols = st.columns(2)
    with cols[0]:
        height = st.number_input("Height", min_value=15, max_value=101, value=31, step=2, help="Use odd numbers for clean walls")
    with cols[1]:
        width = st.number_input("Width", min_value=15, max_value=101, value=49, step=2, help="Use odd numbers for clean walls")

    seed_mode = st.selectbox("Seed", ["Random", "Fixed"], index=0)
    seed = None
    if seed_mode == "Fixed":
        seed = st.number_input("Seed value", min_value=0, max_value=999999, value=42, step=1)

    algo = st.selectbox("Solver", list(SOLVERS.keys()), index=2)
    speed = st.slider("Animation speed (ms per step)", min_value=0, max_value=200, value=10, step=5)

    st.markdown("### Start / Goal")
    start_r = st.number_input("Start row", min_value=1, max_value=height-2, value=1, step=2)
    start_c = st.number_input("Start col", min_value=1, max_value=width-2, value=1, step=2)
    goal_r = st.number_input("Goal row", min_value=1, max_value=height-2, value=height-2, step=2)
    goal_c = st.number_input("Goal col", min_value=1, max_value=width-2, value=width-2, step=2)

    st.markdown("---")
    gen_btn = st.button("🎲 Generate Maze", use_container_width=True)
    solve_btn = st.button("🚀 Solve", use_container_width=True)

# state
if "maze" not in st.session_state or gen_btn:
    grid = generate_maze_prims(int(height), int(width), seed=seed if seed_mode=="Fixed" else None)
    st.session_state["maze"] = Maze(grid=grid, start=(int(start_r), int(start_c)), goal=(int(goal_r), int(goal_c)))
else:
    # keep previous grid if not regenerating, but update s/g
    m: Maze = st.session_state["maze"]
    m.start = (int(start_r), int(start_c))
    m.goal = (int(goal_r), int(goal_c))
    st.session_state["maze"] = m

maze: Maze = st.session_state["maze"]
# ensure start/goal on passages
sr, sc = maze.start
gr, gc = maze.goal
if maze.grid[sr, sc] == 0: maze.grid[sr, sc] = 1
if maze.grid[gr, gc] == 0: maze.grid[gr, gc] = 1

legend = st.columns([1,1,1,1,1])
with legend[0]:
    st.markdown("<div class='legend-dot' style='background:#0E1020;border:1px solid #0E1020'></div> Walls", unsafe_allow_html=True)
with legend[1]:
    st.markdown("<div class='legend-dot' style='background:#173e66;'></div> Passages", unsafe_allow_html=True)
with legend[2]:
    st.markdown("<div class='legend-dot' style='background:#4c42a6;'></div> Visited", unsafe_allow_html=True)
with legend[3]:
    st.markdown("<div class='legend-dot' style='background:#00b3a0;'></div> Path", unsafe_allow_html=True)
with legend[4]:
    st.markdown("<div class='legend-dot' style='background:#c6ff00;'></div> Start &nbsp;&nbsp; <span class='legend-dot' style='background:#ff6363;'></span> Goal", unsafe_allow_html=True)

left, right = st.columns([2,1])

with left:
    # static initial render
    fig, ax = plt.subplots(figsize=(8, 8*(maze.grid.shape[0]/maze.grid.shape[1])))
    render_grid(maze.grid, maze.start, maze.goal, ax=ax)
    plot_spot = st.empty()
    plot_spot.pyplot(fig, clear_figure=True)

with right:
    with st.expander("ℹ️ About this project", expanded=True):
        st.write("""
        - **Generation:** Randomized Prim's algorithm → perfect mazes (single unique path between any two cells).
        - **Solvers:** BFS, Dijkstra, and A* (Manhattan heuristic).
        - **UI:** Custom neon theme, responsive layout, and smooth animation.
        - **Tip:** Use *Fixed* seed to reproduce the same maze for demos.
        """)
    st.markdown("### Stats")
    stat_area = st.empty()

# solve & animate
if solve_btn:
    solver_fn = SOLVERS[algo]
    visited: List[Cell] = []
    total_steps = 0
    frames = 0
    t0 = time.perf_counter()

    placeholder = left.container()
    plot_ph = placeholder.empty()

    for status, payload, meta in solver_fn(maze.grid, maze.start, maze.goal, yield_steps=True):
        if status == "visit":
            visited.append(payload)
            total_steps += 1
            # redraw
            fig, ax = plt.subplots(figsize=(8, 8*(maze.grid.shape[0]/maze.grid.shape[1])))
            render_grid(maze.grid, maze.start, maze.goal, visited=visited, ax=ax)
            plot_ph.pyplot(fig, clear_figure=True)
            frames += 1
            if speed > 0:
                time.sleep(speed/1000.0)
        elif status == "fail":
            st.error("No path found 😢 (this shouldn't happen in a perfect maze).")
            break
        elif status == "done":
            path = payload
            fig, ax = plt.subplots(figsize=(8, 8*(maze.grid.shape[0]/maze.grid.shape[1])))
            render_grid(maze.grid, maze.start, maze.goal, visited=visited, path=path, ax=ax)
            plot_ph.pyplot(fig, clear_figure=True)
            dt = time.perf_counter() - t0
            with right:
                stat_area.markdown(f"""
                **Algorithm:** {algo}  
                **Visited cells:** {len(visited)}  
                **Path length:** {len(path)}  
                **Run time:** {dt:.3f}s  
                **Frames:** {frames}
                """)
            break

st.markdown("---")
st.caption("Built with ❤️ in Python + Streamlit. Perfect for demos and college projects.")
