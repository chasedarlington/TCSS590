import subprocess
from datetime import datetime
from flask import Flask, redirect, request

app = Flask(__name__)
process = None


@app.route("/")
def home():
    running = process is not None and process.poll() is None

    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>LunarLander PPO Control Panel</h1>

        <p>Status: {"Training running" if running else "Training not running"}</p>

        <form action="/run" method="post">
            
            <label>Run name:</label><br>
            <input name="run_name" type="text" value="ppo_lunar_lander"><br><br>
                                    
            <label>Timestep:</label><br>
            <input name="timestep" type="number" value="2048"><br><br>
            
            <label>PPO epochs:</label><br>
            <input name="epochs" type="number" value="10"><br><br>
            
            <label>Epsilon:</label><br>
            <input name="epsilon" type="number" step="0.01" value="0.2"><br><br>
            
            <label>Gamma:</label><br>
            <input name="gamma" type="number" step="0.01" value="0.99"><br><br>
            
            <label>Actor Learning Rate:</label><br>
            <input name="lr_actor" type="number" step="0.0001" value="0.0003"><br><br>
            
            <label>Critic Learning Rate:</label><br>
            <input name="lr_critic" type="number" step="0.0001" value="0.001"><br><br>
            
            <label>Episodes:</label><br>
            <input name="episodes" type="number" value="2000"><br><br>
            
            <label>Maximum Steps per Episode:</label><br>
            <input name="ep_max_steps" type="number" value="1000"><br><br>
            
            <label>Reward Penalty per Timestep:</label><br>
            <input name="ep_reward_penalty" type="number" step="0.01" value="-0.01"><br><br>
            
            <label>Timeout Penalty (exceeding max steps per episode):</label><br>
            <input name="ep_timeout_penalty" type="number" step="1" value="-25.0"><br><br>

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