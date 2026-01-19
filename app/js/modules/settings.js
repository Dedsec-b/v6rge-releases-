import { api } from './api.js';

export class SettingsManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupBackendSettings();
        this.setupHotspotSettings();
        this.setupAppearanceSettings();
        this.setupGPUSettings();
        this.setupStorageSettings();
    }

    setupAppearanceSettings() {
        const toggle = document.getElementById('dark-mode-toggle');
        if (!toggle) return;

        // Load saved preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
            toggle.checked = true;
        }

        toggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
            }
        });
    }

    setupHotspotSettings() {
        const toggle = document.getElementById('hotspot-toggle');
        const detailsPanel = document.getElementById('hotspot-details');
        const urlDisplay = document.getElementById('hotspot-url-display');
        const copyBtn = document.getElementById('copy-hotspot-url');

        if (!toggle) return;

        // Poll status on load
        this.checkHotspotStatus(toggle, detailsPanel, urlDisplay);

        toggle.addEventListener('change', async (e) => {
            if (e.target.checked) {
                // START
                detailsPanel.style.display = 'block';
                urlDisplay.textContent = 'Starting tunnel (downloading cloudflared if needed)...';
                urlDisplay.style.color = 'var(--text-muted)';
                toggle.disabled = true;

                try {
                    const res = await api.post('/network/hotspot/start');
                    if (res.status === 'success' || res.status === 'pending') {
                        this.pollHotspotUrl(urlDisplay, toggle);
                    } else {
                        urlDisplay.textContent = 'Error: ' + res.message;
                        urlDisplay.style.color = 'var(--status-error)';
                        toggle.checked = false;
                        toggle.disabled = false;
                    }
                } catch (err) {
                    urlDisplay.textContent = 'API Error: ' + err.message;
                    toggle.checked = false;
                    toggle.disabled = false;
                }
            } else {
                // STOP
                toggle.disabled = true;
                try {
                    await api.post('/network/hotspot/stop');
                    detailsPanel.style.display = 'none';
                    urlDisplay.textContent = '';
                } catch (err) {
                    console.error(err);
                } finally {
                    toggle.disabled = false;
                }
            }
        });

        // Copy button
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const url = urlDisplay.textContent;
                if (url && url.startsWith('http')) {
                    navigator.clipboard.writeText(url);
                    const originalText = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => copyBtn.textContent = originalText, 2000);
                }
            });
        }
    }

    async checkHotspotStatus(toggle, details, display) {
        try {
            const res = await api.get('/network/hotspot/status');
            if (res.running) {
                toggle.checked = true;
                details.style.display = 'block';
                if (res.url) {
                    display.textContent = res.url;
                    display.style.color = 'var(--accent-primary)';
                } else {
                    this.pollHotspotUrl(display, toggle);
                }
            } else {
                toggle.checked = false;
                details.style.display = 'none';
            }
        } catch (e) {
            console.warn('Failed to check hotspot status', e);
        }
    }

    pollHotspotUrl(displayElement, toggleElement) {
        const interval = setInterval(async () => {
            if (!toggleElement.checked) {
                clearInterval(interval);
                return;
            }
            try {
                const res = await api.get('/network/hotspot/status');
                if (res.url) {
                    displayElement.textContent = res.url;
                    displayElement.style.color = 'var(--accent-primary)';
                    toggleElement.disabled = false;
                    clearInterval(interval);
                } else if (!res.running) {
                    displayElement.textContent = 'Tunnel stopped unexpectedly.';
                    displayElement.style.color = 'var(--status-error)';
                    toggleElement.disabled = false;
                    clearInterval(interval);
                }
            } catch (e) {
                // ignore
            }
        }, 2000);
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
                                const response = await api.post('/config', { models_dir: path }, { 'Content-Type': 'application/json' });
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
