V6RGE BACKEND BUILD INSTRUCTIONS
==================================

1. Install Dependencies
   Open a terminal in the 'backend' folder and run:
   pip install -r requirements.txt

   *Note: Ensure you have PyTorch installed with CUDA support first!*
   Visit https://pytorch.org/get-started/locally/ for the command.

2. Build the Executable
   Open this 'packaging' folder in a terminal and run:
   build_backend.bat

3. Locate the Output
   The standalone backend will be in:
   backend/dist/v6rge_backend/

4. Integration
   When building your Electron app installer, include this 'v6rge_backend' folder
   in your resources/ directory.
