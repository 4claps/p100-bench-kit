# P100 Bench Kit

One Docker image that benchmarks LLM inference on your NVIDIA GPUs and
produces a result file you can share. It is built for Tesla P100s but runs
on anything the bundled llama.cpp builds support (P100, P40/P4, V100; set
`CUDA_ARCHS` to add more).

You open a web page, pick a model, an engine, which GPUs and which tests,
choose how many runs each test does, and watch it live. Every run is kept in
a history you can compare, mark invalid, and export.

## What it measures

| Module | What | Typical time |
|---|---|---|
| **Speed** | `llama-bench` prompt processing and decode t/s at empty and 16k context, per KV cache type | 5–15 min |
| **Server** | real `llama-server` requests: a ~2k-token prose summary and a code prompt, fixed output length, MTP on and off | 5–10 min |
| **RAM offload** | MoE expert layers in system RAM (`-ot exps=CPU`); decode is then bound by RAM bandwidth, so this separates DDR3 from DDR4 boxes | 5–10 min |
| **Agent** | the Hermes agent tool-use battery (trap tasks from hermes-agent's `toolperf_abeval`, MIT); pass rate and wall time per task | ~2 min per task per run |

Every test cools the GPUs to 48 °C before it starts and records GPU
temperature, SM and memory clocks, power, utilization, VRAM, PCIe link, CPU
load, clock, temperature and package power, RAM and fan speeds once a second.
Two guardrails watch every sample: a busy GPU stuck below 1000 MHz (stuck
clocks have invalidated real results before) and a GPU above 80 °C. Each
warns once when the condition has held for 10 s and, by default, aborts the run
(untick it on New Run to only warn).

## Quick start

```bash
docker build -t p100-bench-kit .          # 30–90 min: compiles two llama.cpp builds
docker run -d --name bench --gpus all -p 8090:8090 \
  -v /path/to/your/models:/models \
  -v bench-data:/data \
  -v /sys/firmware/dmi/tables:/host-dmi:ro \
  p100-bench-kit
docker logs bench                          # shows where the login token is
docker exec bench cat /data/token          # the token itself
```

Open `http://<this-machine>:8090`, paste the token, then:

1. **Models**: download a standard model, or verify one you already have.
   Existing files are found by size and checksum even if you renamed them.
2. **New run**: pick model, engine, GPUs and modules, set the number of runs, **Review run**, **Start run**.
3. Watch it on the live page. Close the browser if you like; the run keeps going.
4. On the finished run, **Download result file** to share it (see `results/README.md`).

Nothing else may use the selected GPUs during a run; the preflight check
blocks the run if another process holds them.

## Engines

| Engine | Source | Multi-GPU | Notes |
|---|---|---|---|
| `upstream` | ggml-org/llama.cpp, pinned commit | layer split | the fair baseline across hardware |
| `kmic` | Kmic-68/llama.cpp, pinned commit | tensor split | Pascal-tuned fork; uses its QUICKSTART env (P2P, graphs on pre-Volta). Built for P100 (sm_60) only, as the fork's own build guide does |

Commits are pinned in the `Dockerfile` (`UPSTREAM_COMMIT`, `KMIC_COMMIT`).
On a P40 or V100, use the `upstream` engine.
Builds use CUDA 12.9, the last toolkit line that still targets sm_60/sm_70.
The CPU backend is built for every x86 variant and picked at runtime, so old
Xeons without AVX2 still work.

## Making results comparable

Numbers from two machines can only be compared when the model file, engine
commit and settings match. The kit helps:

- Standard models are pinned by file size and SHA256 in `models/manifest.yaml`.
  Any other `.gguf` in your models folder works but is marked **not comparable**.
- The result file records the engine, its commit, every module parameter
  and a hardware summary.
- Server requests run at temperature 0 with a fixed seed and a fixed number
  of generated tokens (`ignore_eos`). MTP acceptance still varies a little
  run to run, which is why each test repeats and reports the median.
- Agent pass rates mostly measure the model, not the hardware; wall time is
  the hardware number.

## Configuration

All optional, as environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BENCHKIT_TOKEN` | random, saved in `/data/token` | login token |
| `BENCHKIT_PORT` | `8090` | web port |
| `BENCHKIT_COOLDOWN_C` | `48` | cool-down target before each test |
| `BENCHKIT_LOW_CLOCK_MHZ` | `1000` | low-clock guardrail |
| `BENCHKIT_THERMAL_WARN_C` | `80` | thermal guardrail |
| `BENCHKIT_GUARDRAIL_SUSTAIN_S` | `10` | how long a condition must hold before it fires |
| `BENCHKIT_SAMPLE_INTERVAL_S` | `1` | telemetry interval |

## Security

- The app always requires the token. Log in once per browser; the session
  cookie is HTTP-only and same-site. API clients can send
  `Authorization: Bearer <token>`.
- Keep it on your own network. For access from outside, put it behind a
  reverse proxy with HTTPS (Caddy: `reverse_proxy 127.0.0.1:8090`) or a VPN.
- Agent tasks run shell commands chosen by the model. They run as an
  unprivileged user inside the container, in a scratch folder under `/data/agent`.
- RAM type and speed come from `dmidecode`, which reads the firmware tables.
  Docker hides those from containers, so the run command mounts the host's
  copy read-only at `/host-dmi`. Leave that line out and the RAM fields show
  as unknown; everything else still works.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/pytest -q                        # fake nvidia-smi/llama tools, no GPU needed
BENCHKIT_TEST_HERMES_SOURCE=/path/to/hermes-agent .venv/bin/pytest -q   # also the agent flow
```

`tests/fakes/` has stand-ins for `nvidia-smi`, `llama-bench` and
`llama-server`, so the whole run flow works on a machine without a GPU.

## Credits

- llama.cpp (MIT), the Kmic-68 fork, hermes-agent and its toolperf eval (MIT, Nous Research), uPlot (MIT).
- The guardrails and the cool-down and interleaving rules come from many P100 test runs.
