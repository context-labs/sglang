source .venv/bin/activate
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install transformers==4.48.3
pip install datasets

pip install dotenv
pip install huggingface-hub
pip install tabulate
pip install sentence-transformers
pip install scikit-learn
pip install matplotlib
pip install seaborn
deactivate
