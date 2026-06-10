#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")"

DEVICES=${1-}
if [ -z "$DEVICES" ]; then
    echo "Usage: bash $0 <devices>"
    echo "Example: bash $0 0,1,2,3"
    exit 1
fi

for mode in --random --none --highdeg; do
    bash run_node_fill_ablation.sh "${DEVICES}" --arxiv --GT "${mode}"
    bash run_node_fill_ablation.sh "${DEVICES}" --arxiv --GPH_Slim "${mode}"
    bash run_node_fill_ablation.sh "${DEVICES}" --arxiv --GPH_Large "${mode}"
    bash run_node_fill_ablation.sh "${DEVICES}" --products --GT "${mode}"
    bash run_node_fill_ablation.sh "${DEVICES}" --products --GPH_Slim "${mode}"
    bash run_node_fill_ablation.sh "${DEVICES}" --products --GPH_Large "${mode}"
done
