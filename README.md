# Lightweight Reinforcement Learning for Multi-Agent Pathfinding (RL-MAPF)

This repository contains an extended version of the research codebase developed as part of an MS Thesis project on Multi-Agent Pathfinding (MAPF). While the original training experiments, model development, and empirical evaluations were conducted in Google Colab, this repository modifies and packages the work into a standalone system featuring a FastAPI backend and an interactive web visualizer.

The core architecture implements a communication-enhanced Transformer model trained via Advantage Actor-Critic (A2C) combined with Imitation Learning (IL) and an automated Curriculum Learning pipeline.

---

## Key Features

* **Thesis Research Foundations**: Built on the Transformer policy and training methodology originally developed and evaluated in Google Colab.
* **Interactive Visualization Suite**: Extends the original notebook scripts with a React-based frontend for real-time path inspection and dynamic map editing.
* **Custom Map Designer**: Draw custom obstacle walls, place specific agent starting positions, and assign goal locations interactively.
* **Dual-Theme Support**: Full UI support for both light and dark visualization modes.
* **Backend Inference Engine**: Serves model evaluations and native Pogema SVG animation renders via a local FastAPI service.
* **Automated Curriculum Progression**: Monitors agent performance across a 13-tier difficulty structure based on evaluation success thresholds.

---

## Repository Structure

```text
├── backend/
│   ├── app/
│   │   ├── api.py               # FastAPI endpoints for inference and grid solving
│   │   ├── models.py            # Pydantic schemas for request validation
│   │   └── mapf/
│   │       ├── environment.py   # Pogema environment wrapper and CurriculumManager
│   │       ├── model.py         # Transformer policy and value network architecture
│   │       └── training.py      # A2C + IL rollout loop and evaluation routines
│   └── main.py                  # Entry point for backend server and local trainer
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React UI (Grid editor, visualizer, theme switcher)
│   │   └── index.css            # Global web styles
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## System Requirements & Prerequisites

* **Python**: Version 3.10 or higher
* **Node.js**: Version 18.x or higher
* **PyTorch**: Recommended with CUDA support if training locally (pre-trained weights or Colab-exported checkpoints can be loaded directly)

## Installation & Local Setup
1. Backend Setup

    1. Navigate to the backend directory:

    ```bash
    cd backend
    ```

    2. Create and activate a virtual environment:
    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    ```

    3. Install required dependencies:
    ```bash
    pip install torch numpy pogema fastapi uvicorn pydantic
    ```

    4. Launch the local backend service:
     ```bash
    python main.py
    ```

2. Frontend Setup
    1. Open a new terminal window and navigate to the frontend directory:
    ```bash
    cd frontend
    ```

    2. Install Node dependencies:
    ```bash
    npm install
    ```

    3. Start the Vite development server:
    ```bash
    npm run dev
    ```
    Open your browser and navigate to `http://localhost:5173`.

## Workflow & Development History

### Original Research Phase (Google Colab)
The initial thesis research focused on evaluating deep reinforcement learning architectures and reward structures for Multi-Agent Path Finding (MAPF). Executed in Google Colab using GPU acceleration, this phase comprised:

* **Neural Architecture Search & Comparison**: Evaluated sequential and attention-based backbones (LSTM, GRU, Transformers, MLP) integrated with a CNN feature extractor for local observations, a goal-vector dense layer, and an inter-agent communication layer inspired by PRIMAL.
* **Hybrid RL & Imitation Learning Pipeline**: Developed a framework combining Advantage Actor-Critic (A2C) and Proximal Policy Optimization (PPO) with Imitation Learning guided by expert solvers (such as PBS) to accelerate convergence.
* **Reward Engineering & Curriculum Design**: Analyzed multi-objective reward functions (balancing goal reach, agent cooperation, expert trajectory alignment, and collision penalties) alongside custom Curriculum Learning progression schedules.
* **Experimental Benchmarking**: Conducted extensive training rollouts to measure policy stability, speedup from expert demonstration, and level-transition behavior across neural network architectures.

### Visualization & Deployment Phase (This Repository)
This repository refactors and extends the original Colab experiments into a modular, production-ready system for local execution and interactive analysis:

* **Modular Refactoring**: Ported training logic, custom environment wrappers, and model definitions from Colab notebooks into structured Python packages.
* **REST API Integration**: Built a FastAPI backend service to host model inference and compute on-demand MAPF solutions for custom user configurations.
* **Interactive Web Visualizer**: Developed a React-based frontend featuring a dynamic grid editor to visually test trained policies on custom maps, agent layouts, and obstacles outside standard test sets.

## Curriculum Progression Mechanics

The underlying agent policy trains on a 13-stage curriculum designed to prevent policy collapse as complexity scales:

| Level | Agents | Grid Size | Density | Max Steps |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1 | 8 x 8 | 0.0 | 256 |
| 2 | 2 | 12 x 12 | 0.0 | 128 |
| 3 | 4 | 16 x 16 | 0.0 | 128 |
| 4 | 8 | 24 x 24 | 0.0 | 256 |
| 5 | 16 | 32 x 32 | 0.0 | 256 |
| 6 | 32 | 64 x 64 | 0.0 | 256 |
| 7 | 2 | 8 x 8 | 0.1 | 256 |
| 8 | 4 | 16 x 16 | 0.1 | 256 |
| 9 | 8 | 24 x 24 | 0.1 | 256 |
| 10 | 2 | 12 x 12 | 0.2 | 256 |
| 11 | 4 | 16 x 16 | 0.2 | 256 |
| 12 | 8 | 24 x 24 | 0.2 | 256 |
| 13 | 2 | 12 x 12 | 0.3 | 256 |

Promotions to higher difficulty levels occur when the evaluation success rate satisfies:

* **Success Rate >= 80.0% for 2 consecutive evaluation cycles**

## Citation & Academic Context

If you reference this work or the underlying architecture in your research, please cite the original MS Thesis:

```bibtex
@mastersthesis{gregory_rl_mapf_2024,
  author       = {Gregorios Moulkiotis},
  title        = {Χρήση Αρχιτεκτονικών Transformer και Ενισχυτικής Μάθησης με Μάθηση Μίμησης για το Πρόβλημα της Πολυπρακτορικής Εύρεσης Μονοπατιών},
  school       = {Department of Informatics and Telecommunications, National and Kapodistrian University of Athens},
  year         = {2024},
  type         = {Master's Thesis},
  url          = {[https://pergamos.lib.uoa.gr/item/uoadl:5428603](https://pergamos.lib.uoa.gr/item/uoadl:5428603)}
}