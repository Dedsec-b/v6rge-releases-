import subprocess
import os

class TerminalService:
    def __init__(self):
        pass

    def execute(self, command):
        """
        Executes a shell command and returns the output.
        """
        try:
            print(f"[Terminal] Executing: {command}")
            
            if os.name == 'nt':
                # On Windows, properly invoke PowerShell with -Command flag
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", command],
                    capture_output=True, 
                    text=True,
                    cwd=os.path.expanduser('~')
                )
            else:
                # Unix-like systems
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    cwd=os.path.expanduser('~')
                )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]: {result.stderr}"
                
            return output.strip()
            
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def read_file(self, path):
        try:
            if not os.path.isabs(path):
                # Default to Desktop if relative, or just fail? 
                # Let's assume absolute paths or relative to CWD (usually desktop-app)
                # Better: resolve relative to User Desktop for "God Mode" feel
                path = os.path.join(os.path.expanduser('~'), 'Desktop', path)
            
            if not os.path.exists(path):
                return f"Error: File not found at {path}"
                
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, path, content):
        try:
            if not os.path.isabs(path):
                path = os.path.join(os.path.expanduser('~'), 'Desktop', path)
                
            # Create dirs if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
            
    def list_dir(self, path):
        try:
            if not path or path == '.':
                path = os.path.join(os.path.expanduser('~'), 'Desktop')
            elif not os.path.isabs(path):
                path = os.path.join(os.path.expanduser('~'), 'Desktop', path)
                
            if not os.path.exists(path):
                return f"Error: Path not found {path}"
                
            files = []
            for item in os.listdir(path):
                full = os.path.join(path, item)
                type_ = "DIR" if os.path.isdir(full) else "FILE"
                files.append(f"[{type_}] {item}")
            return "\n".join(files)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

# Singleton
terminal_service = TerminalService()
