"""FedSentry web interface (single page).

Starts a federated LoRA fine-tuning run, shows the held-out loss per round while it runs,
and verifies that the aggregated adapter really changed the model. The page launches the
existing server and client modules as separate processes, so it runs exactly the same
pipeline as the command line.

Two modes:
  * one laptop    - this page starts the server and all three clients locally;
  * three laptops - this page starts the server (and optionally client 0); the other
                    laptops start their clients with the commands shown on the page.

Run from the repo root:
    streamlit run app/Home.py
"""

import gc
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
PARTITIONS_DIR = ROOT / "data" / "partitions"
ADAPTER_PATH = ROOT / "server" / "federated_adapter.npz"
NUM_CLIENTS = 3
ROUND_RE = re.compile(r"Round\s+(\d+):\s+loss=([0-9.]+),\s+num_clients=(\d+)")


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def shard_sizes(client_id):
    """(training examples, held-out examples) of one shard."""
    try:
        from client.client import load_client_data, load_holdout_data
        return (len(load_client_data(client_id, str(PARTITIONS_DIR))),
                len(load_holdout_data(client_id, str(PARTITIONS_DIR))))
    except Exception:  # noqa: BLE001 - fall back to counting lines
        path = PARTITIONS_DIR / f"client_{client_id}.jsonl"
        if not path.exists():
            return None, None
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip()), 0


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def read_log(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_rounds(server_log):
    return [{"round": int(r), "loss": float(loss), "num_clients": int(n)}
            for r, loss, n in ROUND_RE.findall(server_log)]


def tail(text, n=20):
    return "\n".join(text.strip().splitlines()[-n:])


def phase(server_log, distributed):
    if "configure_fit: strategy sampled" in server_log:
        return "Training in progress..."
    if "Received initial parameters" in server_log:
        return "At least one client has joined. Waiting for the others (round 1 starts when all 3 are connected)..."
    if distributed:
        return "Server is running. Waiting for the first client to connect..."
    return "Clients are loading the model and joining the federation..."


def client_command(cid, ip, port, cfg):
    return (f'cd <path to the fedsentry folder>\n'
            f'$env:FEDSENTRY_MAX_EXAMPLES="{cfg["max_examples"]}"; $env:FEDSENTRY_MAX_STEPS="{cfg["max_steps"]}"; '
            f'$env:FEDSENTRY_LR="{cfg["lr"]}"\n'
            f'python -m client.client --client_id {cid} --server_address {ip}:{port}')


def free_gpu_memory():
    st.cache_resource.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


@st.cache_resource(show_spinner=False)
def load_demo_model():
    """Base model with an untrained adapter, plus that adapter's arrays (for resetting)."""
    from model import inference
    from model.model_utils import get_adapter_parameters
    model, tokenizer = inference.build_model()
    return model, tokenizer, get_adapter_parameters(model)


# ---------------------------------------------------------------- training run
def run_federated(cfg, live):
    LOG_DIR.mkdir(exist_ok=True)
    port = cfg["port"]
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "FEDSENTRY_MAX_EXAMPLES": str(cfg["max_examples"]), "FEDSENTRY_MAX_STEPS": str(cfg["max_steps"]),
                "FEDSENTRY_LR": str(cfg["lr"])})
    client_env = dict(env)
    if cfg["use_cpu"]:
        client_env["CUDA_VISIBLE_DEVICES"] = "-1"

    procs, files = [], []

    def launch(name, args, process_env):
        f = open(LOG_DIR / f"{name}.log", "w", encoding="utf-8")
        files.append(f)
        p = subprocess.Popen([sys.executable, "-m", *args], cwd=ROOT, env=process_env,
                             stdout=f, stderr=subprocess.STDOUT)
        procs.append((name, p))
        return p

    start = time.time()
    result = {"ok": False, "rows": [], "message": "", "seconds": 0.0}
    server_log = LOG_DIR / "server.log"
    try:
        if port_in_use(port):
            raise RuntimeError(f"Port {port} is already in use. Close the old server or choose another port.")
        live["status"].info("Starting the federated server...")
        server = launch("server", ["server.server", "--rounds", str(cfg["rounds"]), "--min_clients", str(NUM_CLIENTS),
                                   "--address", f"0.0.0.0:{port}"], env)
        while "gRPC server running" not in read_log(server_log):
            if server.poll() is not None:
                raise RuntimeError("The server stopped during start-up. See the logs below.")
            if time.time() - start > 120:
                raise RuntimeError("The server did not start within 2 minutes.")
            time.sleep(1)

        for k in cfg["local_clients"]:
            launch(f"client_{k}", ["client.client", "--client_id", str(k), "--server_address", f"localhost:{port}"],
                   client_env)

        while server.poll() is None:
            log = read_log(server_log)
            rows = parse_rounds(log)
            if rows:
                live["status"].info(f"Round {len(rows)} of {cfg['rounds']} complete ({time.time() - start:.0f} s elapsed)")
                live["chart"].line_chart(pd.DataFrame(rows).set_index("round")[["loss"]])
            else:
                live["status"].info(f"{phase(log, cfg['distributed'])} ({time.time() - start:.0f} s)")
            live["progress"].progress(min(len(rows) / cfg["rounds"], 1.0))
            for name, p in procs[1:]:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f"{name} stopped with an error (exit code {p.returncode}). See the logs below.")
            if time.time() - start > 3600:
                raise RuntimeError("Timed out after 60 minutes.")
            time.sleep(2)

        rows = parse_rounds(read_log(server_log))
        result["rows"] = rows
        result["ok"] = server.returncode == 0 and len(rows) == cfg["rounds"]
        if not result["ok"]:
            result["message"] = "The server finished, but not every round completed. See the logs below."
    except Exception as e:  # noqa: BLE001 - shown to the user
        result["message"] = str(e)
        result["rows"] = parse_rounds(read_log(server_log))
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in files:
            f.close()
    result["seconds"] = time.time() - start
    return result


# ---------------------------------------------------------------- comparison view
def render_comparison(r):
    from model import compare

    st.markdown("**1. Weight check: did the adapter file really update the model?**")
    match = (f"yes (max difference {r['file_max_diff']:.1e})" if r["file_max_diff"] < 1e-6
             else f"NO (max difference {r['file_max_diff']:.2e})")
    st.dataframe(pd.DataFrame([
        ("Adapter file", f"{Path(r['adapter_file']).name}, saved {r['adapter_modified']}"),
        ("SHA-256 of the file", r["adapter_sha256"][:32] + "..."),
        ("Adapter arrays / parameters", f"{r['num_arrays']} arrays, {r['num_params']:,} parameters"),
        ("Arrays whose values changed", f"{r['arrays_changed']} of {r['num_arrays']}"),
        ("Size of the update (L2 norm)", f"{r['update_norm']:.4f}"),
        ("Mean abs of the B matrices, before to after", f"{r['lora_b_before']:.6f} to {r['lora_b_after']:.6f}"),
        ("Loaded weights equal the file", match),
        ("All other (base) weights unchanged", "yes" if r["base_unchanged"] else "NO"),
    ], columns=["Check", "Result"]), hide_index=True)

    st.markdown(f"**2. Loss on {r['n_test']} held-out examples (never used for training)**")
    a, b, c = st.columns(3)
    a.metric("Base model loss", f"{r['loss_before']:.3f}")
    b.metric("Federated model loss", f"{r['loss_after']:.3f}",
             delta=f"{r['loss_after'] - r['loss_before']:+.3f}", delta_color="inverse")
    c.metric("Perplexity, base to federated", f"{r['ppl_before']:.1f} to {r['ppl_after']:.1f}")
    st.bar_chart(pd.DataFrame({"held-out loss": [r["loss_before"], r["loss_after"]]},
                              index=["Base model", "Federated model"]))

    st.markdown(f"**3. Answers to {r['n_gen']} held-out instructions, compared with the reference answers**")
    d, e = st.columns(2)
    d.metric("Word overlap with reference (F1), base", f"{r['f1_before']:.3f}")
    e.metric("Word overlap with reference (F1), federated", f"{r['f1_after']:.3f}",
             delta=f"{r['f1_after'] - r['f1_before']:+.3f}")
    st.caption(f"The federated answer overlapped more with the reference on {r['f1_wins']} of {r['n_gen']} "
               f"instructions (first {compare.F1_WORDS} words of each answer are compared).")
    for i, ex in enumerate(r["examples"][:4], 1):
        with st.expander(f"Example {i}: {ex['instruction'][:90]}"):
            st.markdown("**Reference answer (from the dataset)**")
            st.write(ex["reference"])
            left, right = st.columns(2)
            left.markdown(f"**Base model** (F1 {ex['f1_before']:.2f})")
            left.write(ex["before"])
            right.markdown(f"**Federated model** (F1 {ex['f1_after']:.2f})")
            right.write(ex["after"])

    gain = (r["loss_before"] - r["loss_after"]) / r["loss_before"] * 100
    if gain > 0 and r["arrays_changed"] > 0 and r["base_unchanged"]:
        st.success(f"The adapter changed {r['arrays_changed']} of {r['num_arrays']} arrays, left every base weight "
                   f"untouched, and lowered the loss on unseen examples by {gain:.1f}%.")
    else:
        st.warning("The weights changed, but the loss on unseen examples did not improve. Try more rounds or a "
                   "higher learning rate (see the settings above).")
    st.download_button("Download this comparison (Markdown)", compare.format_report(r), "comparison.md")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="FedSentry", layout="wide")
st.title("FedSentry")
st.caption("Several participants fine-tune one shared small language model (Qwen2.5-0.5B-Instruct) with LoRA, "
           "without sharing their data.")

st.subheader("1. Participants")
cols = st.columns(NUM_CLIENTS)
for k, col in enumerate(cols):
    n_train, n_test = shard_sizes(k)
    col.metric(f"Client {k}", f"{n_train} train / {n_test} test" if n_train is not None else "data missing")
st.caption("Each client trains only on its own private shard. A few examples per shard are kept back as a test set "
           "that is never used for training. Only the LoRA adapter (about 2 MB) is sent to the server.")

st.subheader("2. Deployment")
mode = st.radio("Where do the clients run?", ["One laptop (all three clients run on this laptop)",
                                              "Three laptops (this laptop is the server)"], horizontal=True)
distributed = mode.startswith("Three")
cfg_net = {"port": free_port(), "local_clients": list(range(NUM_CLIENTS))}
server_ip = lan_ip()
if distributed:
    n1, n2, n3 = st.columns(3)
    server_ip = n1.text_input("This laptop's IP address", lan_ip(), help="Check with ipconfig if it looks wrong.")
    port = int(n2.number_input("Port", 1024, 65535, 8080))
    local_ids = n3.multiselect("Clients running on THIS laptop", list(range(NUM_CLIENTS)),
                               default=[c for c in (0, 2) if c < NUM_CLIENTS])
    cfg_net = {"port": port, "local_clients": sorted(local_ids)}
    with st.expander("3-laptop checklist"):
        st.markdown(
            "1. All laptops on the same network. Use a phone hotspot: college Wi-Fi often blocks laptop-to-laptop traffic.\n"
            "2. Every laptop has the same code (`git pull` on `main`) and has run the model once with internet.\n"
            "3. On this laptop, allow the port in the firewall (admin PowerShell): "
            f"`netsh advfirewall firewall add rule name=FedSentry dir=in action=allow protocol=TCP localport={port}`\n"
            f"4. Press **Start** below, then run the commands shown on the other laptops. "
            f"To test the link first: `Test-NetConnection {server_ip} -Port {port}`.")

st.subheader("3. Training settings")
c1, c2, c3, c4 = st.columns(4)
rounds = int(c1.number_input("Federated rounds", 1, 20, 5))
max_examples = int(c2.number_input("Max training examples per client (200 = all)", 5, 200, 200))
max_steps = int(c3.number_input("Max training steps per round", 1, 200, 40))
lr = c4.selectbox("Learning rate", ["1e-4", "2e-4", "3e-4"], index=0)
use_cpu = st.checkbox("Run the local clients on the CPU (slower, saves GPU memory)", value=False)
cfg = {**cfg_net, "rounds": rounds, "max_examples": max_examples, "max_steps": max_steps, "lr": lr,
       "use_cpu": use_cpu, "distributed": distributed}

if distributed:
    remote = [k for k in range(NUM_CLIENTS) if k not in cfg["local_clients"]]
    st.markdown("**Run these on the other laptops** (start them right after pressing Start):")
    for k in remote:
        st.code(client_command(k, server_ip, cfg["port"], cfg), language="powershell")

if st.button("Start federated training", type="primary"):
    free_gpu_memory()
    st.session_state.pop("result", None)
    st.session_state.pop("comparison", None)
    live = {"status": st.empty(), "progress": st.empty(), "chart": st.empty()}
    st.session_state["result"] = run_federated(cfg, live)
    for placeholder in live.values():
        placeholder.empty()

st.subheader("4. Results")
result = st.session_state.get("result")
if result is None:
    st.write("No run yet. Choose the settings above and press the button.")
else:
    rows = result["rows"]
    if result["ok"]:
        st.success(f"Finished {len(rows)} rounds with {NUM_CLIENTS} clients in {result['seconds']:.0f} s.")
    else:
        st.error(result["message"] or "The run did not finish.")
    if rows:
        df = pd.DataFrame(rows)
        left, right = st.columns([2, 1])
        left.line_chart(df.set_index("round")[["loss"]])
        right.dataframe(df, hide_index=True)
        st.caption("Loss of the aggregated adapter after each round on the clients' held-out examples, weighted by "
                   "example count. Lower is better.")
    with st.expander("Logs"):
        names = ["server"] + [f"client_{k}" for k in cfg["local_clients"]]
        for tab, name in zip(st.tabs(names), names):
            tab.code(tail(read_log(LOG_DIR / f"{name}.log")) or "(empty)", language="text")

st.subheader("5. Verify that the model was updated")
if not ADAPTER_PATH.exists():
    st.info("Run a federated training first. The aggregated adapter is saved when the first round ends.")
else:
    st.caption(f"Adapter on disk: {ADAPTER_PATH.name}, saved {time.ctime(ADAPTER_PATH.stat().st_mtime)}. "
               "Make sure this is from the run you just did.")
    n_samples = int(st.number_input("Held-out instructions to answer", 4, 30, 12))
    if st.button("Compare the base model with the federated model"):
        with st.spinner("Loading the model and running the comparison (about a minute on a GPU)..."):
            from model import compare
            st.session_state["comparison"] = compare.run_comparison(ADAPTER_PATH, n_samples, 64, load_demo_model())
    if st.session_state.get("comparison"):
        render_comparison(st.session_state["comparison"])

st.subheader("6. Try your own question")
if ADAPTER_PATH.exists():
    prompt = st.text_input("Instruction", "Give three tips for staying healthy.")
    if st.button("Answer with and without the federated adapter") and prompt.strip():
        with st.spinner("Generating..."):
            from model import inference
            from model.model_utils import set_adapter_parameters
            model, tokenizer, base_arrays = load_demo_model()
            set_adapter_parameters(model, base_arrays)
            before = inference.generate_answer(model, tokenizer, prompt)
            set_adapter_parameters(model, inference.load_adapter_arrays(ADAPTER_PATH))
            after = inference.generate_answer(model, tokenizer, prompt)
        left, right = st.columns(2)
        left.markdown("**Before** (base model)")
        left.write(before)
        right.markdown("**After** (federated adapter)")
        right.write(after)