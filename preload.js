// Preload script - runs before page loads
// Used to securely expose APIs to the renderer process

const { contextBridge, ipcRenderer } = require('electron');
const os = require('os');

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('v6rge', {
    // Platform info
    platform: process.platform,
    version: '0.1.0',
    homeDir: os.homedir(),

    // Backend Communication
    getGpuInfo: () => ipcRenderer.invoke('get-gpu-info'),
    selectFolder: () => ipcRenderer.invoke('select-folder'),
    reload: () => ipcRenderer.invoke('app:reload'),
});

console.log('V6rge Preload Ready - Platform:', process.platform);
