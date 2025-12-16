echo "Installing Python venv..."
sudo apt install -y python3-venv

echo "Creating virtual environment (.venv)..."
python3.10 -m venv .venv

printf "\nexport LLMWORKDIR=$(dirname \"$PWD\")" >> .venv/bin/activate

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip, wheel, and setuptools..."
pip install --upgrade pip wheel setuptools

echo "Installing requirements..."
pip install -r llm_requirements.txt

# deactivate
