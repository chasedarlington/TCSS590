import subprocess
from datetime import datetime
from flask import Flask, redirect, request

app = Flask(__name__)
process = None


def slider(name, label, value, min_value, max_value, step):
    return f"""
        <label for="{name}">{label}: <span id="{name}_value">{value}</span></label><br>
        <input
            id="{name}"
            name="{name}"
            type="range"
            min="{min_value}"
            max="{max_value}"
            step="{step}"
            value="{value}"
            oninput="document.getElementById('{name}_value').innerText = this.value"
            style="width: 420px;"
        ><br><br>
    """


@app.route("/")
def home():
    running = process is not None and process.poll() is None

    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px; max-width: 700px;">
        <h1>LunarLander PPO Control Panel</h1>

        <p>Status: {"Training running" if running else "Training not running"}</p>

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

            <button type="submit" style="font-size: 20px; padding: 10px 20px;">
                Run Training
            </button>
        </form>

        <br>

        <form action="/stop" method="post">
            <button type="submit" style="font-size: 20px; padding: 10px 20px;">
                Stop Training
            </button>
        </form>

        <p>
            <a href="http://localhost:6006" target="_blank">Open TensorBoard</a>
        </p>
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
        "python",
        "single_file_version.py",
        "train",

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


if __name__ == "__main__":
    app.run(port=5000)
