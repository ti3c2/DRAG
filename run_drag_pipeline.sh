#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model> <dataset> [extra args...]"
  echo "Example: $0 gpt squad --multithread"
  exit 1
fi

model="$1"
dataset="$2"
shift 2
extra_args=("$@")

echo "Running step 1: generate evidences"
python 1_generate_evidences.py "$model" "$dataset" "${extra_args[@]}"

echo "Running step 2: generate evidence rankings"
python 2_generate_evidence_rankings.py "$model" "$dataset" "${extra_args[@]}"

echo "Running step 3: generate graph"
python 3_generate_graph.py $model $dataset ${extra_args[@]}

echo "Running step 4: generate graph rankings"
python 4_generate_graph_rankings.py $model $dataset ${extra_args[@]}

echo "Running step 5: generate responses no context"
python 5_generate_responses_no_context.py $model $dataset ${extra_args[@]}

echo "Running step 6: generate responses"
python 6_generate_responses.py $model $dataset ${extra_args[@]}
