#!/bin/bash
set -euo pipefail

OPTIM_RUNNER="${OPTIM_RUNNER:-streaming}"
OUTPUT_BASE="${OUTPUT_BASE:-$(cd "$(dirname "$0")/../../../.." && pwd)/outputs/stream/${METHOD_NAME:?}}"
PARAMS_LIST="${PARAMS_LIST:-$(cd "$(dirname "$0")" && pwd)/default_params.yaml}"

source "$(dirname "$0")/../optim/_run_optim_method.sh"
