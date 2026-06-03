import glob
import os
import subprocess
from datetime import datetime
from io import BytesIO
import gymnasium as gym
import torch
from PIL import Image
from single_file_version import PPOAgent
from flask import Flask, Response, redirect, request, send_from_directory
app = Flask(__name__)
process = None
play_process = None

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
        <p>Training status: {"Training running" if running else "Training not running"}</p>
        <p>Play status: {"Play running" if playing else "Play not running"}</p>

        <h2>Train</h2>
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

        <h2>Render trained agent in browser</h2>
        <form action="/render" method="get">
            <label for="model">Model path:</label><br>
            <input id="model" name="model" type="text" value="{latest_model}" style="width: 420px;"><br><br>
            <label for="render_episodes">Episodes:</label><br>
            <input id="render_episodes" name="episodes" type="number" value="3" min="1" max="50"><br><br>
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Open Browser Render</button>
        </form>

        <h2>Play manually</h2>
        <form action="/play" method="post">
            <button type="submit" style="font-size: 18px; padding: 10px 20px;">Launch Keyboard Play Window</button>
        </form>

        <h2>Export TensorBoard PNGs</h2>
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
                image = Image.fromarray(frame)
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.getvalue() + b"\r\n"

                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = torch.argmax(agent.policy.actor(state_tensor)).item()
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
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
    app.run(port=5000)
