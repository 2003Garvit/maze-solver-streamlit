# 🌀 Maze Solver — Streamlit

A clean, deploy-ready **Maze Generator + Solver** in Python with a **neon UI**.  
Algorithms included: **Prim's** for generation, **BFS**, **Dijkstra**, and **A\*** for solving.

https://github.com/ (optional to mirror)

---

## ✨ Features
- Stylish dark **neon** theme with smooth animation
- Randomized **Prim's algorithm** generates perfect mazes
- Choose solver: **BFS**, **Dijkstra**, or **A\* (Manhattan)**
- Adjustable maze size, seed, start/goal, and animation speed
- Single-file app (`app.py`) — easy to read and modify

---

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal.

---

## ☁️ One-click Deploy

### Option A — Streamlit Community Cloud (Free)
1. Push these files to a public GitHub repo.
2. Go to https://share.streamlit.io/ → New app → pick your repo/branch and `app.py` as the entry point.
3. Hit **Deploy**. Done.

### Option B — Hugging Face Spaces (Gradio or Streamlit)
1. Create a new **Space** → choose **Streamlit**.
2. Upload `app.py` and `requirements.txt`.
3. The app will build and go live automatically.

> Tip: Prefer **odd** values for height/width to keep wall layout clean.

---

## 🧠 How it works (high level)
- **Prim's** builds a *perfect* maze (exactly one simple path between any two cells).
- **BFS** finds shortest path in an unweighted grid using a queue.
- **Dijkstra** generalizes BFS (equivalent here since all edges cost 1).
- **A\*** uses Manhattan heuristic to speed up search toward the goal.

---

## 📁 Project Structure
```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧩 Customize ideas
- Add **Recursive Division** generator
- Add diagonal moves or weighted tiles
- Support **click-to-set** start/goal using streamlit-drawable-canvas
- Export solved path as **GIF** or **CSV of coordinates**
- Add a **Race Mode** to compare algorithms side-by-side
