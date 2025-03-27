
apt update && apt install -y nano
apt update && apt install -y jq
python -m pip install --upgrade pip
python -m pip install uv

cd /home/kyle/code/sglang
git pull
git checkout implement-toploc
git pull
uv venv .sglang --python 3.12 --seed

source .sglang/bin/activate
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install transformers==4.48.3
pip install datasets
pip install pre-commit
pre-commit install
deactivate
