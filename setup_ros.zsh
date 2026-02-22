# Source this script to set up the ROS environment:
#   source ./setup_ros.zsh

cd "$(dirname "${(%):-%x}")/mappo-test" || return
source .venv2/bin/activate
cd env
echo "[OK] ROS env ready. Run: python simulate_model.py --env ros"
