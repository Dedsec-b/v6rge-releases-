import { api } from './api.js';

export class SettingsManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupBackendSettings();
        this.setupGPUSettings();
        this.setupStorageSettings();
    }

    setupBackendSettings() {
        const urlInput = document.getElementById('backend-url-input');
        const resetBtn = document.getElementById('reset-backend-btn');
        const testBtn = document.getElementById('test-connection-btn');
        const saveBtn = document.getElementById('save-backend-btn');
        const statusEl = document.getElementById('connection-status');

        if (urlInput) {
            urlInput.value = localStorage.getItem('v6rge_backend_url') || 'http://localhost:5000';
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (urlInput) urlInput.value = 'http://localhost:5000';
                statusEl.textContent = '';
            });
        }

        if (testBtn) {
            testBtn.addEventListener('click', async () => {
                const url = urlInput.value.trim();
                if (!url) return;

                testBtn.disabled = true;
                testBtn.textContent = 'Testing...';
                statusEl.textContent = 'Checking connection...';
                statusEl.className = 'connection-status';

                const success = await api.checkConnection(url);

                if (success) {
                    statusEl.textContent = '✅ Connection Successful!';
                    statusEl.classList.add('success');
                } else {
                    statusEl.textContent = '❌ Connection Failed';
                    statusEl.classList.add('error');
                }

                testBtn.disabled = false;
                testBtn.textContent = 'Test Connection';
            });
        }

        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                const url = urlInput.value.trim();
                if (url) {
                    api.setBaseUrl(url);
                    statusEl.textContent = 'Settings Saved';
                    setTimeout(() => statusEl.textContent = '', 2000);
                }
            });
        }
    }

    setupGPUSettings() {
        const saveBtn = document.getElementById('save-gpu-btn');
        const nameInput = document.getElementById('gpu-name');
        const vramInput = document.getElementById('gpu-vram');

        // Load saved values
        if (nameInput) nameInput.value = localStorage.getItem('gpu_name') || '';
        if (vramInput) vramInput.value = localStorage.getItem('gpu_vram') || '';

        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                localStorage.setItem('gpu_name', nameInput.value);
                localStorage.setItem('gpu_vram', vramInput.value);
                saveBtn.textContent = 'Saved!';
                setTimeout(() => saveBtn.textContent = 'Save GPU Info', 2000);
            });
        }
    }

    setupStorageSettings() {
        const pathInput = document.getElementById('model-dir-input');
        const selectBtn = document.getElementById('select-model-dir-btn');

        // Fetch current config from backend 
        if (pathInput) {
            api.get('/config')
                .then(config => {
                    if (config.models_dir) {
                        pathInput.value = config.models_dir;
                    }
                })
                .catch(err => console.error('Failed to load config:', err));
        }

        if (selectBtn) {
            selectBtn.addEventListener('click', async () => {
                // Use Electron API via preload
                // v6rge is exposed in preload.js
                if (window.v6rge && window.v6rge.selectFolder) {
                    try {
                        const path = await window.v6rge.selectFolder();
                        if (path) {
                            pathInput.value = path;

                            // Send to backend
                            try {
                                const response = await api.post('/config/update', { models_dir: path }, { 'Content-Type': 'application/json' });
                                if (response.status === 'success') {
                                    alert('Settings saved. Please restart the application for changes to take effect.');
                                }
                            } catch (e) {
                                console.error('Failed to save config:', e);
                                alert('Failed to save settings: ' + e.message);
                            }
                        }
                    } catch (err) {
                        console.error('Folder selection failed:', err);
                    }
                } else {
                    alert('Folder selection only available in desktop app mode.');
                }
            });
        }
    }
}
