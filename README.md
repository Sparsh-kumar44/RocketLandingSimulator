# 🚀 Rocket Landing Simulator — Deep Q-Learning

A 2D rocket landing simulator built with **Python, Pygame, and PyTorch**, featuring custom physics and a Deep Q-Network (DQN) agent trained to learn controlled rocket landings.

The project combines real-time simulation, reinforcement learning, reward engineering, and interactive visualization.

<!-- Replace this with a real screenshot or GIF -->
<p align="center">
  <img src="assets/screenshots/simulator.png" alt="Rocket Landing Simulator" width="700">
</p>

<p align="center">
  <strong>Simulate. Train. Land.</strong>
</p>

## ✨ Features

- **Custom 2D Physics:** Simulates rocket movement, gravity, thrust, and rotation.
- **Deep Q-Network (DQN):** Uses a neural network to estimate action values and learn a landing policy.
- **Experience Replay:** Trains on sampled past experiences to improve learning stability.
- **Target Network:** Uses a separate target network to stabilize Q-value targets.
- **Interactive Visualization:** Displays the rocket, engine flames, landing pad, and flight information in real time.
- **Landing Evaluation:** Tracks landing conditions and distinguishes successful landings from crashes.

## 🛠️ Tech Stack

| Technology | Purpose                                          |
| ---------- | ------------------------------------------------ |
| Python     | Core implementation                              |
| Pygame     | Simulation and visualization                     |
| PyTorch    | DQN model and training                           |
| NumPy      | Numerical operations and replay-batch processing |

## 📁 Project Structure

```text
RocketLandingSimulator/
├── assets/
│   └── screenshots/
├── settings.py
├── rocket.py
├── physics.py
├── environment.py
├── model.py
├── memory.py
├── agent.py
├── train.py
├── play.py
├── renderer.py
├── effects.py
├── utils.py
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/Sparsh-kumar44/RocketLandingSimulator.git
cd RocketLandingSimulator
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## 🎮 Usage

**Run the manually controlled simulator**

```bash
python main.py
```

**Train the DQN agent**

```bash
python train.py
```

**Run the trained agent**

```bash
python play.py
```

Train the agent before running the trained version so that the expected model checkpoint is available.

## 🕹️ Controls

| Key           | Action                            |
| ------------- | --------------------------------- |
| ↑ Up Arrow    | Activate main engine              |
| ← Left Arrow  | Activate left thruster            |
| → Right Arrow | Activate right thruster           |
| R             | Restart after the simulation ends |

## 🧠 Reinforcement Learning Approach

The project uses **Deep Q-Learning**, a value-based reinforcement-learning algorithm.

The agent observes the rocket's state, selects an action, receives a reward, and learns to improve its decisions through repeated interactions with the environment.

The DQN implementation includes:

- An online (policy) network to estimate action values.
- A target network to calculate more stable training targets.
- Epsilon-greedy exploration to balance exploration and exploitation.
- Experience replay to sample transitions from previous interactions.
- A reward function that encourages safe landings and penalizes crashes and unsafe impact conditions.

## 🏗️ System Architecture

```text
Rocket Environment
       ↓
   State Vector
       ↓
    DQN Agent
       ↓
     Action
       ↓
  Physics Update
       ↓
 Reward + Next State
       ↓
  Experience Replay
       ↓
  Network Training
```

## 📸 Results

Add screenshots and a short GIF showing the simulator and the trained agent in action.

Once training experiments are complete, include measured results such as landing success rate, average reward, and crash rate. Avoid reporting unmeasured performance figures.

## 🔮 Future Improvements

- Plot reward and landing success rate across training episodes.
- Compare different reward functions and hyperparameters.
- Add evaluation across multiple random initial conditions.
- Visualize the learned landing policy.
- Record training and inference demonstrations.

## 📄 License

This project is distributed under the license specified in the repository's `LICENSE` file.
