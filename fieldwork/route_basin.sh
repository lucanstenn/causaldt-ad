#!/usr/bin/env bash
set -euo pipefail
python -m causaldt_ad.headworks --config "${1:-main}" route
python -m causaldt_ad.headworks --config "${1:-main}" regulate
