#!/usr/bin/env bash
# Report 04 feedback campaign on the GPU (run from the repo root, inside tmux, env qaoa-gpu).
#   bash scripts/campaign.sh smoke   # ~5 min check with 12 qubits (run this first)
#   bash scripts/campaign.sh all     # experiments 1 -> 2 -> 3 -> 4 (~52 h), then the report
#   bash scripts/campaign.sh exp2    # a single experiment: exp1 | exp2 | exp3 | exp4 | report
# Everything resumes: finished runs and grid rows are skipped, so re-running after a crash continues.
set -u
export PYTHONPATH=src
PY="python -u"
GPU="--method statevector --device GPU"
mkdir -p logs

step() {  # step <log name> <command...>: runs, logs, and keeps going if it fails
  local name=$1; shift
  echo "=== $(date '+%F %T') START $name"
  "$@" 2>&1 | tee -a "logs/$name.log"
  echo "=== $(date '+%F %T') END $name (exit ${PIPESTATUS[0]})"
}

smoke() {
  step smoke_grid $PY src/experiments/grid_search_ext.py $GPU --nodes 12 --steps 6 --zoom-steps 0 --max-p 2 --tag smoke
  step smoke_camp $PY src/experiments/run_campaign.py $GPU --name smoke --grid n12_statevector_GPU_smoke --nodes 12 --depths 2,3 --seeds 2 --final-shots 2000
  step smoke_report $PY src/experiments/campaign_report.py --only smoke
}

exp1() {  # depth: p = 3..6, 20 seeds, 50x50 exact grid (layers 1-3), extra layers = 0
  step exp1 $PY src/experiments/run_campaign.py $GPU --name exp1_depth --depths 3,4,5,6 --seeds 20
}

exp2() {  # grid effects at p = 4: resolution (20 vs 50), exact vs shots energy, init depth (2 vs 3 layers)
  step exp2_grid20_exact $PY src/experiments/grid_search_ext.py $GPU --steps 20 --zoom-steps 0 --max-p 3 --energy exact --tag s20_exact
  step exp2_grid20_shots $PY src/experiments/grid_search_ext.py $GPU --steps 20 --zoom-steps 0 --max-p 3 --energy shots --tag s20_shots
  step exp2_camp_g20_exact $PY src/experiments/run_campaign.py $GPU --name exp2_grid20_exact --grid n30_statevector_GPU_s20_exact --depths 4 --seeds 20
  step exp2_camp_g20_shots $PY src/experiments/run_campaign.py $GPU --name exp2_grid20_shots --grid n30_statevector_GPU_s20_shots --depths 4 --seeds 20
  step exp2_camp_g50_2layers $PY src/experiments/run_campaign.py $GPU --name exp2_grid50_2layers --grid-max-layer 2 --depths 4 --seeds 20
}

exp3() {  # extra layers at p = 5, 6: perturb and interp (zeros = exp1, seeds 1-10)
  step exp3_perturb $PY src/experiments/run_campaign.py $GPU --name exp3_perturb --depths 5,6 --seeds 10 --extend perturb
  step exp3_interp $PY src/experiments/run_campaign.py $GPU --name exp3_interp --depths 5,6 --seeds 10 --extend interp
}

exp4() {  # generalization: p = 4 angles transferred to other graphs (start node 60), 5 seeds
  for n in 20 24 28 30; do
    for tol in 2.0 3.0; do
      step "exp4_n${n}_t${tol}" $PY src/experiments/run_campaign.py $GPU --name "exp4_n${n}_tol${tol}" \
        --nodes $n --start-node 60 --tolerance $tol --depths 4 --seeds 5
    done
  done
}

report() { step report $PY src/experiments/campaign_report.py; }

case "${1:-}" in
  smoke) smoke ;;
  exp1) exp1 ;;
  exp2) exp2 ;;
  exp3) exp3 ;;
  exp4) exp4 ;;
  report) report ;;
  all) exp1; exp2; exp3; exp4; report ;;
  *) echo "usage: bash scripts/campaign.sh smoke|all|exp1|exp2|exp3|exp4|report"; exit 1 ;;
esac
