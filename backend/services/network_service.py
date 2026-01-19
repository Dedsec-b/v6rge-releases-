import os
import subprocess
import threading
import time
import requests
import re
import sys
from pathlib import Path
from config import BASE_DIR

class NetworkService:
    def __init__(self):
        self.tunnel_process = None
        self.tunnel_url = None
        self.is_running = False
        
        # Tools directory for binaries
        self.bin_dir = BASE_DIR / 'bin'
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        
        self.cloudflared_path = self.bin_dir / 'cloudflared.exe'
        self.download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

    def ensure_cloudflared(self):
        """Check if cloudflared is installed, download if not"""
        if self.cloudflared_path.exists():
            return True
            
        print(f"[Network] Downloading cloudflared from {self.download_url}...")
        try:
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            with open(self.cloudflared_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[Network] cloudflared downloaded to {self.cloudflared_path}")
            return True
        except Exception as e:
            print(f"[Network] Failed to download cloudflared: {e}")
            return False

    def start_hotspot(self, port=5000):
        """Start the Cloudflare tunnel"""
        if self.is_running:
            return {"status": "success", "url": self.tunnel_url, "message": "Already running"}
            
        if not self.ensure_cloudflared():
            return {"status": "error", "message": "Failed to download cloudflared binary"}

        print(f"[Network] Starting Hotspot on port {port}...")
        
        # Kill any existing instances first
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'cloudflared.exe'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass

        try:
            # properly detach process to avoid blocking
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            self.tunnel_process = subprocess.Popen(
                [str(self.cloudflared_path), 'tunnel', '--url', f'http://localhost:{port}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                startupinfo=startupinfo
            )
            
            self.is_running = True
            
            # Read stderr in a separate thread to find the URL (cloudflared outputs link to stderr)
            threading.Thread(target=self._monitor_tunnel, daemon=True).start()
            
            # Wait a bit for URL to appear
            for _ in range(20):
                if self.tunnel_url:
                    break
                time.sleep(0.5)
                
            if self.tunnel_url:
                return {"status": "success", "url": self.tunnel_url}
            else:
                return {"status": "pending", "message": "Tunnel starting, check back in a few seconds"}
                
        except Exception as e:
            print(f"[Network] Error starting tunnel: {e}")
            self.is_running = False
            return {"status": "error", "message": str(e)}

    def _monitor_tunnel(self):
        """Background thread to read tunnel output and find URL"""
        print("[Network] Monitoring tunnel output...")
        while self.is_running and self.tunnel_process:
            line = self.tunnel_process.stderr.readline()
            if not line:
                break
                
            if 'trycloudflare.com' in line:
                # regex to extract url
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match:
                    self.tunnel_url = match.group(0)
                    print(f"[Network] 🌍 Hotspot Live at: {self.tunnel_url}")
            
    def stop_hotspot(self):
        """Stop the tunnel"""
        print("[Network] Stopping Hotspot...")
        self.is_running = False
        self.tunnel_url = None
        
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process = None
            except:
                pass
                
        # Force kill to be safe
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'cloudflared.exe'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
            
        return {"status": "success"}

    def get_status(self):
        return {
            "running": self.is_running,
            "url": self.tunnel_url
        }

network_service = NetworkService()
