# Run training inside the project's .venv
# Usage (PowerShell): .\scripts\run_training.ps1

$venv = '..\.venv\Scripts\Activate.ps1'
if (Test-Path -Path $venv) {
    Write-Output "Activating .venv"
    & $venv
} else {
    Write-Output ".venv activation script not found. Ensure .venv exists."
}

# Install requirements if needed
Write-Output "Installing requirements (if not installed)"
. .venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt

# Run the training script
. .venv\Scripts\Activate.ps1; python -m src.churn_prediction.train
