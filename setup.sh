#!/usr/bin/env bash
# Paste this on a fresh GPU instance. Installs vLLM, reports the card and its
# configuration, optionally pre-downloads the model, then runs a short smoke test.
#
#   bash setup.sh                       install, report, smoke test
#   bash setup.sh --prefetch            also download the model now (see note below)
#   bash setup.sh --skip-install        report and smoke test only
#
# Note on --prefetch: the model download is billed GPU time on every provider. Pull
# it before the timed run so download bandwidth does not land inside the result.
set -uo pipefail

INSTALL=1; PREFETCH=0
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
for a in "$@"; do
  case "$a" in
    --skip-install) INSTALL=0 ;;
    --prefetch) PREFETCH=1 ;;
    *) echo "unknown flag: $a"; exit 2 ;;
  esac
done

PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "FAIL: no python on PATH"; exit 1; }
cd "$(dirname "$0")"

echo "=== 1. GPU ==="
command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi not found"; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,power.limit,power.default_limit,power.max_limit,clocks.max.sm,ecc.mode.current,persistence_mode,memory.total \
  --format=csv || { echo "FAIL: nvidia-smi could not query the GPU"; exit 1; }
# A provider that has lowered power.limit below power.default_limit has capped the
# card. Record it: it explains sustained numbers that look wrong for the SKU.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
  --format=csv

# Only now, once a GPU is confirmed, record when this box started costing money.
# Writing it before the check would leave a stale stamp on a machine that never ran
# a benchmark, and a later run in this directory would quietly price against it.
STAMP=".gpubench_instance_start"
if [ ! -f "$STAMP" ]; then
  date -u +%s > "$STAMP"
  echo "Recorded instance start in $STAMP. This is a LOWER BOUND on overhead: it starts when"
  echo "you got a shell, not when the instance started billing. Pass --instance-start to run"
  echo "if you know when you clicked deploy."
fi

echo
echo "=== 2. Software ==="
if [ "$INSTALL" = 1 ]; then
  # vLLM first: it pins a compatible torch. Installing torch first then vllm
  # usually replaces the torch you just installed.
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q vllm || echo "WARN: vllm install failed, GEMM and bandwidth will still run"
fi
"$PY" - <<'PYEOF'
import platform
try:
    import torch
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"devices {torch.cuda.device_count()}  available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"device 0: {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.0f} GiB")
except Exception as e:
    print(f"FAIL: torch unusable: {type(e).__name__}: {e}")
    raise SystemExit(1)
try:
    import vllm
    print(f"vllm {vllm.__version__}")
except Exception as e:
    print(f"WARN: vllm unusable: {type(e).__name__}: {e}")
print(f"python {platform.python_version()}  kernel {platform.release()}")
PYEOF
[ $? -eq 0 ] || exit 1

if [ "$PREFETCH" = 1 ]; then
  echo
  echo "=== 2b. Pre-downloading $MODEL ==="
  "$PY" -m pip install -q huggingface_hub
  ("$PY" -m huggingface_hub.commands.huggingface_cli download "$MODEL" \
     || hf download "$MODEL" \
     || huggingface-cli download "$MODEL") || echo "WARN: prefetch failed, the run will download it"
fi

echo
echo "=== 3. Smoke test (about 3 minutes, writes to results-smoke/) ==="
# Deliberately a separate --out: a smoke test must never land in the real results.
"$PY" gpubench.py run --provider smoke --repeats 2 --gemm-seconds 60 \
  --membw-gib 2 --skip-inference --out results-smoke || { echo "FAIL: smoke run"; exit 1; }
"$PY" gpubench.py analyze --dir results-smoke --warmup 10 >/dev/null || { echo "FAIL: analyze"; exit 1; }
echo
sed -n '1,30p' results-smoke/summary.md

echo
echo "=== Ready ==="
echo "Real run:  $PY gpubench.py run --provider <name> --price-per-hr <price>"
echo "Delete the smoke data before copying results back:  rm -rf results-smoke"
