import glob
import os
import subprocess
import threading
import time
from datetime import datetime
from io import BytesIO

from flask import Flask, Response, redirect, request, send_from_directory

app = Flask(__name__)
process = None
play_process = None
browser_env = None
browser_state = None
browser_action = 0
browser_lock = threading.Lock()


def slider(name, label, value, min_value, max_value, step):
    return f"""
        <label for="{name}">{label}: <span id="{name}_value">{value}</span></label><br>
        <input id="{name}" name="{name}" type="range" min="{min_value}" max="{max_value}" step="{step}" value="{value}"
               oninput="document.getElementById('{name}_value').innerText = this.value" style="width: 420px;"><br><br>
    """


def latest_file(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else ""


@app.route("/")
def home():
    running = process is not None and process.poll() is None
    playing = play_process is not None and play_process.poll() is None
    latest_model = latest_file("*.pt") or "ppo_lunar_lander.pt"
    latest_run = latest_file("runs/*") or "runs/ppo_lunar_lander"

    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px; max-width: 760px;">
        <h1>LunarLander PPO Control Panel</h1>
        <h2>Train PPO Agent</h2>
        <form action="/run" method="post">
            <label for="run_name">Run name:</label><br>
            <input id="run_name" name="run_name" type="text" value="ppo_lunar_lander" style="width: 420px;"><br><br>
            {slider("timestep", "Update timestep", "2048", "256", "8192", "256")}
            {slider("epochs", "PPO epochs", "10", "1", "30", "1")}
            {slider("epsilon", "PPO clip epsilon", "0.2", "0.05", "0.5", "0.01")}
            {slider("gamma", "Discount gamma", "0.99", "0.90", "0.999", "0.001")}
            {slider("lr_actor", "Actor learning rate", "0.0003", "0.0001", "0.005", "0.0001")}
            {slider("lr_critic", "Critic learning rate", "0.001", "0.0001", "0.005", "0.0001")}
            {slider("episodes", "Training episodes", "2000", "100", "10000", "100")}
            {slider("ep_max_steps", "Max episode steps", "1000", "100", "2000", "100")}
            {slider("ep_reward_penalty", "Per-step reward penalty", "-0.01", "-0.20", "0", "0.01")}
            {slider("ep_timeout_penalty", "Timeout penalty", "-25.0", "-200", "0", "5")}
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
        <form action="/browser_play" method="post">
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Launch External Keyboard Play Window</button>
        </form>

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


@app.route("/run", methods=["POST"])
def run():
    global process
    if process is not None and process.poll() is None:
        return redirect("/")

    timestep = request.form.get("timestep", "2048")
    epochs = request.form.get("epochs", "10")
    epsilon = request.form.get("epsilon", "0.2")
    gamma = request.form.get("gamma", "0.99")
    lr_actor = request.form.get("lr_actor", "0.0003")
    lr_critic = request.form.get("lr_critic", "0.001")
    episodes = request.form.get("episodes", "2000")
    ep_max_steps = request.form.get("ep_max_steps", "1000")
    ep_reward_penalty = request.form.get("ep_reward_penalty", "-0.01")
    ep_timeout_penalty = request.form.get("ep_timeout_penalty", "-25.0")
    run_name = request.form.get("run_name", "ppo_lunar_lander")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"
    log_dir = f"runs/{run_id}"
    model_path = f"{run_id}.pt"

    process = subprocess.Popen([
        "python", "single_file_version.py", "train",
        "--timestep", timestep,
        "--epochs", epochs,
        "--epsilon", epsilon,
        "--gamma", gamma,
        "--lr-actor", lr_actor,
        "--lr-critic", lr_critic,
        "--episodes", episodes,
        "--ep-max-steps", ep_max_steps,
        "--ep-reward-penalty", ep_reward_penalty,
        "--ep-timeout-penalty", ep_timeout_penalty,
        "--log-dir", log_dir,
        "--model", model_path,
    ])
    return redirect("/")


@app.route("/stop", methods=["POST"])
def stop():
    global process
    if process is not None and process.poll() is None:
        process.terminate()
    return redirect("/")


@app.route("/play", methods=["POST"])
def play():
    global play_process
    if play_process is None or play_process.poll() is not None:
        play_process = subprocess.Popen(["python", "single_file_version.py", "play"])
    return redirect("/")


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

            function sendAction(action) {
                currentAction = action;
                document.getElementById("action_label").innerText = labels[action];
                fetch(`/browser_action?action=${action}`, {method: "POST"});
            }

            function resetGame() {
                fetch("/browser_reset", {method: "POST"});
                sendAction(0);
            }

            function keyToAction(key) {
                key = key.toLowerCase();
                if (key === "a") return 1;
                if (key === "w") return 2;
                if (key === "d") return 3;
                if (key === "s") return 0;
                return currentAction;
            }

            document.addEventListener("keydown", function(event) {
                if (["w", "a", "s", "d", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) event.preventDefault();
                if (event.key === "ArrowLeft") sendAction(1);
                else if (event.key === "ArrowUp") sendAction(2);
                else if (event.key === "ArrowRight") sendAction(3);
                else sendAction(keyToAction(event.key));
            });

            document.addEventListener("keyup", function(event) {
                if (["w", "a", "d", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) sendAction(0);
            });

            window.onload = function() { document.getElementById("game").focus(); };
        </script>
    </body>
    </html>
    """


def ensure_browser_env():
    global browser_env, browser_state
    import gymnasium as gym
    if browser_env is None:
        browser_env = gym.make("LunarLander-v2", render_mode="rgb_array")
        browser_state, _ = browser_env.reset()


def encode_frame(frame):
    from PIL import Image
    image = Image.fromarray(frame)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


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


@app.route("/browser_play_stream")
def browser_play_stream():
    return Response(browser_play_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


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


@app.route("/browser_reset", methods=["POST"])
def browser_reset():
    global browser_state, browser_action
    ensure_browser_env()
    with browser_lock:
        browser_state, _ = browser_env.reset()
        browser_action = 0
    return {"reset": True}


@app.route("/render")
def render_page():
    model = request.args.get("model", latest_file("*.pt") or "ppo_lunar_lander.pt")
    episodes = request.args.get("episodes", "3")
    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>LunarLander Browser Render</h1>
        <p>Model: <code>{model}</code></p>
        <p><a href="/">Back</a></p>
        <img src="/render_stream?model={model}&episodes={episodes}" style="border: 1px solid #ccc; max-width: 100%;">
    </body>
    </html>
    """


def render_frames(model_path, episodes):
    import gymnasium as gym
    import torch
    from single_file_version import PPOAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, device)
    agent.load(model_path)
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


@app.route("/render_stream")
def render_stream():
    model = request.args.get("model", latest_file("*.pt") or "ppo_lunar_lander.pt")
    episodes = request.args.get("episodes", "3")
    if not os.path.exists(model):
        return f"Model not found: {model}", 404
    return Response(render_frames(model, episodes), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/export", methods=["POST"])
def export():
    log_dir = request.form.get("log_dir", "runs/ppo_lunar_lander")
    subprocess.run(["python", "single_file_version.py", "export", "--log-dir", log_dir], check=False)
    return redirect("/exports")


@app.route("/exports")
def exports():
    os.makedirs("tensorboard_pngs", exist_ok=True)
    images = sorted(glob.glob("tensorboard_pngs/*.png"))
    cards = "".join(
        f"<h3>{os.path.basename(img)}</h3><img src='/export_file/{os.path.basename(img)}' style='max-width: 720px; border: 1px solid #ccc;'><br><br>"
        for img in images
    ) or "<p>No PNGs found. Export a TensorBoard log directory first.</p>"
    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>Exported TensorBoard PNGs</h1>
        <p><a href="/">Back</a></p>
        {cards}
    </body>
    </html>
    """


@app.route("/export_file/<path:filename>")
def export_file(filename):
    return send_from_directory("tensorboard_pngs", filename)


if __name__ == "__main__":
    app.run(port=5000, threaded=True)
