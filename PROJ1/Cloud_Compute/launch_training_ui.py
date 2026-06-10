import glob
import html
import os
import subprocess
import threading
import time
from datetime import datetime
from io import BytesIO
from flask import Flask, Response, redirect, request, send_from_directory

## Get path to current file and define directories for logs and PNG exports (/log and /png )
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
LOG_DIR = os.path.join(BASE_DIR, "log")
PNG_DIR = os.path.join(BASE_DIR, "tensorboard_png")

app = Flask(__name__)
process = None
play_process = None
log_file_path = None
play_log_file_path = None
browser_env = None
browser_state = None
browser_action = 0
browser_lock = threading.Lock()

def rel(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

def latest_file(pattern):
    files = glob.glob(rel(pattern))
    return os.path.basename(max(files, key=os.path.getmtime)) if files else ""

def latest_run_dir():
    files = glob.glob(os.path.join(BASE_DIR, "runs", "*"))
    return os.path.relpath(max(files, key=os.path.getmtime), BASE_DIR) if files else "runs/ppo_lunar_lander"

def read_log(path, max_chars=30000):
    if not path:
        return "No console output yet."
    try:
        with open(path, "r", errors="replace") as f:
            content = f.read()
        return content[-max_chars:] if len(content) > max_chars else content
    except FileNotFoundError:
        return "Log file not found yet."

def open_log(run_id, suffix="train"):
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"{run_id}_{suffix}.log")

### Slider helper function
## Helper function to create an HTML slider input for the training parameters. It generates a label and an input of type range, with JavaScript to update the displayed value as the slider is moved.
def slider(name, label, value, min_value, max_value, step):
    return f"""
        <label for="{name}">{label}: <span id="{name}_value">{value}</span></label><br>
        <input id="{name}" name="{name}" type="range" min="{min_value}" max="{max_value}" time_step="{step}" value="{value}"
               oninput="document.getElementById('{name}_value').innerText = this.value" style="width: 420px;"><br><br>
    """

### Home route
## The main route of the Flask application that serves the control panel for training and playing the LunarLander PPO agent. It displays the current status of training and play processes, provides links to logs and status pages, and contains forms to start/stop training, launch play windows, render trained agents, and export TensorBoard PNGs.
@app.route("/") # Flask route decorator (When someone visits the root URL /, run the function directly below this line.)
def home():
    running = process is not None and process.poll() is None
    playing = play_process is not None and play_process.poll() is None
    latest_model = latest_file("*.pt") or "ppo_lunar_lander.pt"
    latest_run = latest_run_dir()

    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px; max-width: 900px;">
        <h1>LunarLander PPO Control Panel</h1>
        <p>Training status: {"Training running" if running else "Training not running"}</p>
        <p>External pygame play status: {"Play running" if playing else "Play not running"}</p>
        <p><a href="/logs" target="_blank">Open full console output</a> | <a href="/status" target="_blank">Process status</a></p>

        <h2>Live Console Output</h2>
        <iframe src="/logs_embed" style="width: 100%; height: 280px; border: 1px solid #ccc; background: #111;"></iframe>

        <h2>Train PPO Agent</h2>
        <form action="/run" method="post">
            <label for="run_name">Run name:</label><br>
            <input id="run_name" name="run_name" type="text" value="ppo_lunar_lander" style="width: 420px;"><br><br>
            {slider("timestep", "Update timestep", "2048", "256", "8192", "256")}
            {slider("epochs", "PPO epochs", "10", "1", "30", "1")}
            {slider("ppo_clip", "PPO clip ppo_clip", "0.2", "0.05", "0.5", "0.01")}
            {slider("disc_coef", "Discount disc_coef", "0.99", "0.90", "0.999", "0.001")}
            {slider("lr_actor", "Actor learning rate", "0.0003", "0.0001", "0.005", "0.0001")}
            {slider("lr_critic", "Critic learning rate", "0.001", "0.0001", "0.005", "0.0001")}
            {slider("episodes", "Training episodes", "2000", "100", "10000", "100")}
            {slider("ep_max_steps", "Max episode steps", "1000", "100", "2000", "100")}
            {slider("ep_reward_penalty", "Per-time_step reward penalty", "-0.01", "-0.20", "0", "0.01")}
            {slider("ep_timeout_penalty", "Timeout penalty", "-25.0", "-200", "0", "5")}
            {slider("parallel_envs", "Parallel environments", "1", "1", "16", "1")}
            {slider("rollout_workers", "Rollout workers", "1", "1", "16", "1")}
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Run Training</button>
        </form>

        <form action="/stop" method="post" style="margin-top: 10px;">
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Stop Training</button>
        </form>

        <h2>Render Trained Agent</h2>
        <form action="/render" method="get">
            <label for="model">Model path:</label><br>
            <input id="model" name="model" type="text" value="{latest_model}" style="width: 420px;"><br><br>
            <label for="render_episodes">Episodes:</label><br>
            <input id="render_episodes" name="episodes" type="number" value="3" min="1" max="50"><br><br>
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Open Browser Render</button>
        </form>

        <h2>Play In Browser</h2>
        <p><a href="/browser_play" style="font-size: 18px;">Open Browser Keyboard Play</a></p>

        <h2>Play Out of Browser (pygame)</h2>
        <form action="/play" method="post">
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Launch External Keyboard Play Window</button>
        </form>

        <h2>Export TensorBoard PNG</h2>
        <form action="/export" method="post">
            <label for="log_dir">TensorBoard log dir:</label><br>
            <input id="log_dir" name="log_dir" type="text" value="{latest_run}" style="width: 420px;"><br><br>
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Export and View PNGs</button>
        </form>

        <p><a href="http://localhost:6006" target="_blank">Open TensorBoard</a></p>
    </body>
    </html>
    """

## Run training
@app.route("/run", methods=["POST"]) # Flask route decorator (When someone visits the URL /run, run the function directly below this line.)
def run():
    global process, log_file_path
    if process is not None and process.poll() is None:
        return redirect("/")

    timestep = request.form.get("timestep", "2048")
    epochs = request.form.get("epochs", "10")
    epsilon = request.form.get("ppo_clip", "0.2")
    gamma = request.form.get("disc_coef", "0.99")
    lr_actor = request.form.get("lr_actor", "0.0003")
    lr_critic = request.form.get("lr_critic", "0.001")
    episodes = request.form.get("episodes", "2000")
    ep_max_steps = request.form.get("ep_max_steps", "1000")
    ep_reward_penalty = request.form.get("ep_reward_penalty", "-0.01")
    ep_timeout_penalty = request.form.get("ep_timeout_penalty", "-25.0")
    parallel_envs = request.form.get("parallel_envs", "1")
    rollout_workers = request.form.get("rollout_workers", "1")
    run_name = request.form.get("run_name", "ppo_lunar_lander")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"
    tensorboard_dir = f"runs/{run_id}"
    model_path = f"{run_id}.pt"
    log_file_path = open_log(run_id, "train")
    log_file = open(log_file_path, "w", buffering=1)

    print(f"Starting training run {run_id}", file=log_file, flush=True)
    print(f"TensorBoard log dir: {tensorboard_dir}", file=log_file, flush=True)
    print(f"Model path: {model_path}", file=log_file, flush=True)
    print(f"Parallel envs: {parallel_envs}", file=log_file, flush=True)
    print(f"Rollout workers: {rollout_workers}", file=log_file, flush=True)

    process = subprocess.Popen([
        "python", "-u", "single_file_version.py", "train",
        "--timestep", timestep,
        "--epochs", epochs,
        "--ppo_clip", epsilon,
        "--disc_coef", gamma,
        "--lr-actor", lr_actor,
        "--lr-critic", lr_critic,
        "--episodes", episodes,
        "--ep-max-steps", ep_max_steps,
        "--ep-reward-penalty", ep_reward_penalty,
        "--ep-timeout-penalty", ep_timeout_penalty,
        "--parallel-envs", parallel_envs,
        "--rollout-workers", rollout_workers,
        "--log-dir", tensorboard_dir,
        "--model", model_path,
    ], cwd=BASE_DIR, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return redirect("/")

## Stop training
@app.route("/stop", methods=["POST"])
def stop():
    global process
    if process is not None and process.poll() is None:
        process.terminate()
    return redirect("/")

## Start external pygame play window
@app.route("/play", methods=["POST"])
def play():
    global play_process, play_log_file_path
    if play_process is None or play_process.poll() is not None:
        run_id = datetime.now().strftime("play_%Y%m%d_%H%M%S")
        play_log_file_path = open_log(run_id, "pygame")
        log_file = open(play_log_file_path, "w", buffering=1)
        play_process = subprocess.Popen(["python", "-u", "single_file_version.py", "play"], cwd=BASE_DIR, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return redirect("/")

## Status endpoint for AJAX polling
@app.route("/status")
def status():
    return {
        "training_running": process is not None and process.poll() is None,
        "training_returncode": None if process is None else process.poll(),
        "play_running": play_process is not None and play_process.poll() is None,
        "play_returncode": None if play_process is None else play_process.poll(),
        "training_log": log_file_path,
        "play_log": play_log_file_path,
    }

## Logs page with auto-refresh every 2 seconds
@app.route("/logs")
def logs():
    selected = request.args.get("which", "train")
    path = play_log_file_path if selected == "play" else log_file_path
    content = html.escape(read_log(path))
    return f"""
    <html>
    <head><meta http-equiv="refresh" content="2"></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>Console Output</h1>
        <p><a href="/">Back</a> | <a href="/logs?which=train">Training log</a> | <a href="/logs?which=play">Play log</a></p>
        <pre style="background:#111;color:#eee;padding:20px;white-space:pre-wrap;min-height:500px;">{content}</pre>
    </body>
    </html>
    """

# Logs embedded in an iframe with auto-refresh every 2 seconds, showing only the last 12000 characters for performance
@app.route("/logs_embed")
def logs_embed():
    content = html.escape(read_log(log_file_path, max_chars=12000))
    return f"""
    <html>
    <head><meta http-equiv="refresh" content="2"></head>
    <body style="margin:0;background:#111;color:#eee;font-family:monospace;">
        <pre style="white-space:pre-wrap;margin:0;padding:12px;">{content}</pre>
    </body>
    </html>
    """

# Endpoint to serve log text for AJAX fetching (not used in current version but can be useful for future enhancements)
@app.route("/log_text")
def log_text():
    selected = request.args.get("which", "train")
    path = play_log_file_path if selected == "play" else log_file_path
    return Response(read_log(path), mimetype="text/plain")

# Browser-based play interface with keyboard controls (WASD or arrow keys) and live game rendering using MJPEG streaming
@app.route("/browser_play")
def browser_play():
    return """
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>LunarLander Browser Keyboard Play</h1>
        <p><a href="/">Back</a></p>
        <p>Click the game image first. Controls: W = main engine, A = left engine, D = right engine, S/no key = do nothing.</p>
        <p>Discrete LunarLander only accepts one action at a time, so combined keys like W+A are not separate actions.</p>
        <button onclick="resetGame()" style="font-size: 18px; padding: 8px 18px;">Reset Episode</button>
        <p>Current action: <span id="action_label">0</span></p>
        <img id="game" tabindex="0" src="/browser_play_stream" style="border: 1px solid #ccc; max-width: 100%; outline: none;" autofocus>

        <script>
            const labels = {0: "0 no-op", 1: "1 left engine", 2: "2 main engine", 3: "3 right engine"};
            let currentAction = 0;
            function sendAction(action) { currentAction = action; document.getElementById("action_label").innerText = labels[action]; fetch(`/browser_action?action=${action}`, {method: "POST"}); }
            function resetGame() { fetch("/browser_reset", {method: "POST"}); sendAction(0); }
            function keyToAction(key) { key = key.toLowerCase(); if (key === "a") return 1; if (key === "w") return 2; if (key === "d") return 3; if (key === "s") return 0; return currentAction; }
            document.addEventListener("keydown", function(event) { if (["w", "a", "s", "d", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) event.preventDefault(); if (event.key === "ArrowLeft") sendAction(1); else if (event.key === "ArrowUp") sendAction(2); else if (event.key === "ArrowRight") sendAction(3); else sendAction(keyToAction(event.key)); });
            document.addEventListener("keyup", function(event) { if (["w", "a", "d", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) sendAction(0); });
            window.onload = function() { document.getElementById("game").focus(); };
        </script>
    </body>
    </html>
    """

### GYM ENV !!!
## Global variables to hold the Gym environment, current state, and action for the browser play interface. A lock is used to synchronize access to these variables between the streaming generator and the action update route.
def ensure_browser_env():
    global browser_env, browser_state
    import gymnasium as gym
    if browser_env is None:
        browser_env = gym.make("LunarLander-v2", render_mode="rgb_array")
        browser_state, _ = browser_env.reset()
 
### ENCODE JPEG FRAME !!!
## Function to encode a single video frame (numpy array) as JPEG bytes for streaming to the browser. Uses PIL to convert the array to an image and save it to a bytes buffer.
def encode_frame(frame):
    from PIL import Image
    image = Image.fromarray(frame)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

### MJPEG STREAMING GENERATOR !!!
## Generator function that continuously renders frames from the Gym environment, applies the current action, and yields the frames as MJPEG stream data. It also handles episode termination and resetting the environment when needed.
def browser_play_frames():
    global browser_state, browser_action
    ensure_browser_env()
    while True:
        with browser_lock:
            frame = browser_env.render()
            action = browser_action
            browser_state, _, terminated, truncated, _ = browser_env.step(action)
            if terminated or truncated:
                browser_state, _ = browser_env.reset()
                browser_action = 0
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encode_frame(frame) + b"\r\n"
        time.sleep(1 / 30)

### Browser play stream route
## Route to serve the MJPEG stream for the browser play interface. It calls the generator function and sets the appropriate MIME type for streaming video.
@app.route("/browser_play_stream")
def browser_play_stream():
    return Response(browser_play_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

### Browser action update route
## Route to receive action updates from the browser via POST requests. It updates the global action variable
@app.route("/browser_action", methods=["POST"])
def browser_action_route():
    global browser_action
    action = request.args.get("action", "0")
    try:
        action = int(action)
    except ValueError:
        action = 0
    if action not in {0, 1, 2, 3}:
        action = 0
    with browser_lock:
        browser_action = action
    return {"action": action}

### Browser reset route
## Route to reset the Gym environment when the user clicks the reset button in the browser interface.
@app.route("/browser_reset", methods=["POST"])
def browser_reset():
    global browser_state, browser_action
    ensure_browser_env()
    with browser_lock:
        browser_state, _ = browser_env.reset()
        browser_action = 0
    return {"reset": True}

### Browser render page
## Route to display a page with an embedded MJPEG stream of the trained agent playing in the Gym environment. The user can specify which model to load and how many episodes to render.
@app.route("/render")
def render_page():
    model = request.args.get("model", latest_file("*.pt") or "ppo_lunar_lander.pt")
    episodes = request.args.get("episodes", "3")
    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>LunarLander Browser Render</h1>
        <p>Model: <code>{html.escape(model)}</code></p>
        <p><a href="/">Back</a></p>
        <img src="/render_stream?model={html.escape(model)}&episodes={episodes}" style="border: 1px solid #ccc; max-width: 100%;">
    </body>
    </html>
    """

### Render stream generator
## Function to render frames from the Gym environment using a trained model. It loads the specified model, runs the environment for a given number of episodes, and yields the rendered frames as an MJPEG stream. The agent selects actions based on the loaded policy.
def render_frames(model_path, episodes):
    import gymnasium as gym
    import torch
    from single_file_version import PPOAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, device)
    agent.load(rel(model_path))
    agent.policy.eval()

    try:
        for _ in range(int(episodes)):
            state, _ = env.reset()
            done = False
            while not done:
                frame = env.render()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encode_frame(frame) + b"\r\n"
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = torch.argmax(agent.policy.actor(state_tensor)).item()
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                time.sleep(1 / 30)
    finally:
        env.close()

### Render stream route
## Route to serve the MJPEG stream of the rendered trained agent. It takes the model path and number of episodes as query parameters and calls the render_frames generator to produce the stream.
@app.route("/render_stream")
def render_stream():
    model = request.args.get("model", latest_file("*.pt") or "ppo_lunar_lander.pt")
    episodes = request.args.get("episodes", "3")
    if not os.path.exists(rel(model)):
        return f"Model not found: {model}", 404
    return Response(render_frames(model, episodes), mimetype="multipart/x-mixed-replace; boundary=frame")

### Export TensorBoard PNGs
## Route to handle exporting TensorBoard logs to PNG images. It runs a subprocess that calls the export function in the single_file_version.py script, which should contain the logic to read TensorBoard logs and save visualizations as PNG files. After exporting, it redirects the user to the /exports page to view the generated images.
@app.route("/export", methods=["POST"])
def export():
    global log_file_path
    log_dir = request.form.get("log_dir", "runs/ppo_lunar_lander")
    run_id = datetime.now().strftime("export_%Y%m%d_%H%M%S")
    log_file_path = open_log(run_id, "export")
    with open(log_file_path, "w", buffering=1) as log_file:
        subprocess.run(["python", "-u", "single_file_version.py", "export", "--log-dir", log_dir], cwd=BASE_DIR, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
    return redirect("/exports")

### Exports page
## Route to display the exported PNG images from TensorBoard logs. It looks for PNG files in
@app.route("/exports")
def exports():
    os.makedirs(PNG_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(PNG_DIR, "*.png")))
    cards = "".join(
        f"<h3>{os.path.basename(img)}</h3><img src='/export_file/{os.path.basename(img)}' style='max-width: 720px; border: 1px solid #ccc;'><br><br>"
        for img in images
    ) or "<p>No PNGs found. Export a TensorBoard log directory first.</p>"
    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>Exported TensorBoard PNGs</h1>
        <p><a href="/">Back</a> | <a href="/logs">View export console output</a></p>
        {cards}
    </body>
    </html>
    """

### Export file route
## Route to serve the exported PNG files for display on the /exports page. It uses Flask's send_from_directory to serve files from the PNG_DIR.
@app.route("/export_file/<path:filename>")
def export_file(filename):
    return send_from_directory(PNG_DIR, filename)

if __name__ == "__main__":
    app.run(port=5000, threaded=True)
