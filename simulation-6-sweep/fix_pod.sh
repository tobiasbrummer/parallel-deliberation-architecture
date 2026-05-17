#!/usr/bin/env bash
# Quick-fix für den laufenden Pod ohne Image-Rebuild.
# Pinnt alle Versionen auf eine kompatible Combo (Unsloth allowlist + torch 2.6).
#
# Ursache des Bruchs: transformers 4.57.0 referenziert beim Modul-Import
# torchao.quantization.Float8WeightOnlyConfig — eine API, die erst in
# torchao >= 0.14 existiert; aber torchao >= 0.14 braucht torch >= 2.9
# bzw. torch >= 2.10 (register_constant). Das Image hat torch 2.6 gepinnt,
# also bleibt nur transformers downgrade auf 4.56.2.

set -euo pipefail

# Constraint umgehen, sonst überschreibt das Image-Constraint nichts
unset PIP_CONSTRAINT

echo "=== Pinning compatible versions ==="
pip install --force-reinstall --no-deps \
    "transformers==4.56.2" \
    "peft==0.18.0" \
    "trl==0.23.1" \
    "torchao==0.13.0" \
    "bitsandbytes==0.47.0" \
    "accelerate==1.10.1"

echo
echo "=== Verifying imports ==="
python <<'PY'
import torch, transformers, peft, trl, torchao, bitsandbytes, accelerate
from importlib.metadata import version as _v
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"peft         {_v('peft')}")
print(f"trl          {_v('trl')}")
print(f"torchao      {torchao.__version__}")
print(f"bitsandbytes {_v('bitsandbytes')}")
print(f"accelerate   {_v('accelerate')}")

# This is the actual smoke test — the broken combo died here
from transformers import BloomPreTrainedModel  # via lazy import
print("BloomPreTrainedModel import ok")

import unsloth
print(f"unsloth      {_v('unsloth')} ok")
PY

echo
echo "=== Done. Run sim6.py now. ==="
