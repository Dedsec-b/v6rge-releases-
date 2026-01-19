@echo off
echo ============================================================
echo V6RGE DESKTOP APP - ONE-TIME SETUP
echo ============================================================
echo.
echo This will install all required AI models and dependencies.
echo This may take 10-30 minutes depending on your internet speed.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.10 or 3.11 from https://python.org
    pause
    exit /b 1
)

echo [1/5] Python found. Checking version...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or higher is required!
    pause
    exit /b 1
)

echo [2/5] Installing PyTorch with CUDA support...
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo WARNING: CUDA installation failed. Trying CPU version...
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
)

echo [3/5] Installing AI model dependencies...
pip install flask flask-cors transformers accelerate diffusers safetensors sentencepiece
pip install qwen-vl-utils Pillow numpy scipy opencv-python soundfile librosa

echo [4/5] Installing image/audio tools...
pip install rembg realesrgan basicsr gfpgan facexlib
pip install audio-separator trimesh pygltflib

echo [5/5] Installing remaining packages...
pip install omegaconf einops peft huggingface-hub autoawq

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo You can now run V6rge by double-clicking: start_v6rge.bat
echo Or by running: npm start
echo.
pause
