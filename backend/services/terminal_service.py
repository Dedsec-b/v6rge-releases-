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
            # Run command, capture output
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=os.path.expanduser('~') # Default to User Home, safer than CWD of the exe
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]: {result.stderr}"
                
            return output.strip()
            
        except Exception as e:
            return f"Error executing command: {str(e)}"

# Singleton
terminal_service = TerminalService()
