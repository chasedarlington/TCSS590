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
            <label>Episodes:</label><br>
            <input name="episodes" type="number" value="2000"><br><br>

            <label>Max episode steps:</label><br>
            <input name="max_steps" type="number" value="1000"><br><br>

            <label>Run name:</label><br>
            <input name="run_name" type="text" value="ppo_lunar_lander"><br><br>

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

    episodes = request.form.get("episodes", "2000")
    max_steps = request.form.get("max_steps", "1000")
    run_name = request.form.get("run_name", "ppo_lunar_lander")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"

    log_dir = f"runs/{run_id}"
    model_path = f"{run_id}.pt"

    process = subprocess.Popen([
        "python",
        "single_file_version.py",
        "--episodes", episodes,
        "--max-steps", max_steps,
        "--log-dir", log_dir,
        "--model-path", model_path,
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