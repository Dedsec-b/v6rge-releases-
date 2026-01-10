@echo off
echo ================================================
echo V6rge Backend Build Script (PyInstaller)
echo ================================================

cd ../backend

echo [1/3] Cleaning previous builds...
rmdir /s /q build dist
del *.spec

echo [2/3] Building Backend Executable...
echo This may take a while as it packages Torch and CUDA libs.
echo.

:: PyInstaller Command
:: --onedir: Easier to debug than --onefile, faster startup
:: --name: Output name
:: --hidden-import: Vital for dynamic libraries
python -m PyInstaller ^
    --name "v6rge_backend" ^
    --onedir ^
    --windowed ^
    --icon "../app/icon.ico" ^
    --clean ^
    --hidden-import="sklearn.utils._cython_blas" ^
    --hidden-import="sklearn.neighbors.typedefs" ^
    --hidden-import="sklearn.neighbors.quad_tree" ^
    --hidden-import="sklearn.tree._utils" ^
    --hidden-import="scipy.signal" ^
    --hidden-import="scipy.special.cython_special" ^
    --hidden-import="scikit_image" ^
    --hidden-import="watchfiles" ^
    --hidden-import="sqlite3" ^
    --hidden-import="llama_cpp" ^
    --hidden-import="qwen_vl_utils" ^
    --hidden-import="chatterbox" ^
    --hidden-import="soundfile" ^
    --hidden-import="rembg" ^
    --hidden-import="realesrgan" ^
    --hidden-import="basicsr" ^
    --hidden-import="gfpgan" ^
    --hidden-import="facexlib" ^
    --hidden-import="audio_separator" ^
    --hidden-import="pygltflib" ^
    --hidden-import="trimesh" ^
    --hidden-import="hy3dgen" ^
    --hidden-import="perth" ^
    --collect-all="gradio_client" ^
    --collect-all="tqdm" ^
    --collect-all="outetts" ^
    --collect-all="perth" ^
    --collect-all="watchfiles" ^
    --collect-all="llama_cpp" ^
    --collect-all="llama_cpp_python" ^
    --collect-all="scipy" ^
    --collect-all="sklearn" ^
    --collect-all="chatterbox" ^
    --collect-all="rembg" ^
    --collect-all="audio_separator" ^
    --collect-all="basicsr" ^
    --collect-all="realesrgan" ^
    --collect-all="onnxruntime" ^
    --collect-all="hy3dgen" ^
    --collect-all="soundfile" ^
    --collect-all="qwen_vl_utils" ^
    server_modular.py

echo.
echo [3/3] Build Complete!
echo Backend executable is located in: desktop-app/backend/dist/v6rge_backend/
echo.
echo IMPORTANT: Copy this 'v6rge_backend' folder to your Electron 'resources' folder when packaging the frontend.
echo Build Finished.
