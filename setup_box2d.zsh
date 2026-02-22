# Source this script to set up the box2d environment:
#   source ./setup_box2d.zsh

cd "$(dirname "${(%):-%x}")/mappo-test" || return
source .venv/bin/activate
cd env
echo "[OK] box2d env ready. Run: python simulate_model.py --env box2d"
