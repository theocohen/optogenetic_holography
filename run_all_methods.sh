source venv/bin/activate
export PYTHONPATH="$PWD"
python optogenetic_holography/main.py --config_path=./input/configs/bin_amp_phase_mgsa.txt
python optogenetic_holography/main.py --config_path=./input/configs/bin_amp_amp_mgsa.txt
python optogenetic_holography/main.py --config_path=./input/configs/bin_amp_phase_sgd.txt
python optogenetic_holography/main.py --config_path=./input/configs/bin_amp_amp_sgd.txt
python optogenetic_holography/main.py --config_path=./input/configs/bin_amp_amp_sig_sgd.txt
