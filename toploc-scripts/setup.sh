
apt update && apt install -y nano
apt update && apt install -y jq

python3 --version

cd /home/kyle/code/sglang
python3 -m venv .sglang
source .sglang/bin/activate
python3 -m pip install --upgrade pip

git pull
git checkout implement-toploc
git pull


pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install transformers==4.48.3
pip install datasets
pip install pre-commit
pip install dotenv
pre-commit install
deactivate
