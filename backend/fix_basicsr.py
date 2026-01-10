import os
import site
import fileinput
from pathlib import Path

def fix_basicsr():
    # Find site-packages
    site_packages = site.getsitepackages()
    target_file = None
    
    for sp in site_packages:
        possible_path = Path(sp) / "basicsr" / "data" / "degradations.py"
        if possible_path.exists():
            target_file = possible_path
            break
            
    if not target_file:
        # Try local user install
        possible_path = Path(site.getusersitepackages()) / "basicsr" / "data" / "degradations.py"
        if possible_path.exists():
            target_file = possible_path

    if not target_file:
        print("Could not find basicsr installation to fix.")
        return

    print(f"Fixing basicsr at: {target_file}")
    
    # Read and replace
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in content:
            new_content = content.replace(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                "from torchvision.transforms.functional import rgb_to_grayscale"
            )
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("Successfully patched basicsr!")
        else:
            print("basicsr already patched or different version.")
            
    except Exception as e:
        print(f"Failed to patch: {e}")

if __name__ == "__main__":
    fix_basicsr()
