echo "Installing Python venv..."
sudo apt update
sudo apt install -y python3.10-venv

echo "Creating virtual environment (.venv)..."
python3.10 -m venv .venv

echo "Appending LLMWORKDIR to activate..."
cat >> .venv/bin/activate <<'EOF'

# Project-specific env var
export LLMWORKDIR="$(dirname "$PWD")"
EOF

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip, wheel, and setuptools..."
pip install --upgrade pip wheel setuptools

echo "Installing requirements..."
pip install -r llm_requirements.txt