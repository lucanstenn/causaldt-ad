#!/usr/bin/env bash
set -euo pipefail
python -m causaldt_ad.headworks --config "${1:-main}" gauge
python -m causaldt_ad.headworks --config "${1:-main}" divert
