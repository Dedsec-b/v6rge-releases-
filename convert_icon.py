from PIL import Image
import os

png_path = r"C:\Users\Administrator\.gemini\antigravity\brain\fb13c3d8-562f-4f52-8192-1a62f5cca1d3\v6rge_logo_new_1768405618130.png"
ico_path = r"C:\Users\Administrator\Desktop\V6rge\desktop-app\app\icon.ico"

try:
    img = Image.open(png_path)
    # Resize to standard icon sizes if needed, but save usually handles layers if specified
    # For simple conversion:
    img.save(ico_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"Successfully converted PNG to ICO: {ico_path}")
except ImportError:
    print("Pillow not installed. Please run: pip install Pillow")
except Exception as e:
    print(f"Error converting: {e}")
