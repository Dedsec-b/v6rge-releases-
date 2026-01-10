const { app, BrowserWindow, Menu, shell } = require('electron');
const { autoUpdater } = require('electron-updater');
const path = require('path');
const { spawn } = require('child_process');
const fetch = require('node-fetch');

// Keep a global reference to prevent garbage collection
let mainWindow = null;
let pythonProcess = null;

// Python server configuration
const PYTHON_SERVER_PORT = 5000;
const PYTHON_SERVER_URL = `http://127.0.0.1:${PYTHON_SERVER_PORT}`;

async function checkPythonInstalled() {
    return new Promise((resolve) => {
        const python = spawn('python', ['--version']);

        python.on('error', () => {
            const py3 = spawn('python3', ['--version']);

            py3.on('error', () => {
                console.log('Python not found');
                resolve(false);
            });
            py3.on('close', () => resolve(true));
        });

        python.on('close', () => resolve(true));
    });
}

async function waitForServer(maxAttempts = 120) {
    console.log('Waiting for Python server to start...');
    const http = require('http');

    for (let i = 0; i < maxAttempts; i++) {
        const isReady = await new Promise((resolve) => {
            const req = http.get(`${PYTHON_SERVER_URL}/health`, (res) => {
                resolve(res.statusCode === 200);
            });
            req.on('error', () => resolve(false));
            req.setTimeout(2000, () => {
                req.destroy();
                resolve(false);
            });
        });

        if (isReady) {
            console.log('Python server ready!');
            return true;
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    console.log('Python server failed to start');
    return false;
}

async function startPythonServer() {
    console.log('Checking for existing Python server...');
    const http = require('http');
    const isRunning = await new Promise((resolve) => {
        const req = http.get(`${PYTHON_SERVER_URL}/health`, (res) => {
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.setTimeout(1000, () => {
            req.destroy();
            resolve(false);
        });
    });

    if (isRunning) {
        console.log('Existing Python server detected (Manual Mode). Connecting...');
        return true;
    }

    console.log('Starting Python backend server...');

    // Check if Python is installed
    const hasPython = await checkPythonInstalled();
    if (!hasPython) {
        console.error('Python is not installed!');
        // Show error dialog to user
        const { dialog } = require('electron');
        dialog.showErrorBox(
            'Python Required',
            'V6rge requires Python 3.8+ to be installed.\\n\\nPlease install Python from python.org and restart V6rge.'
        );
        app.quit();
        return false;
    }

    // Spawn Python Backend
    // Using the new modular server entry point
    const scriptPath = path.join(__dirname, 'backend', 'server_modular.py');

    // Check if we are in dev or prod (packaged)
    let pythonCmd;
    let spawnArgs;
    let spawnOptions;

    if (app.isPackaged) {
        // PRODUCTION: Run the compiled EXE
        const exePath = path.join(process.resourcesPath, 'v6rge_backend', 'v6rge_backend.exe');
        console.log(`Production Mode : Launching Backend from ${exePath}`);

        pythonCmd = exePath;
        spawnArgs = [];
        spawnOptions = {
            stdio: ['ignore', 'pipe', 'pipe'],
            windowsHide: true // Important for "windowed" mode
        };
    } else {
        // DEVELOPMENT: Run the python script
        const scriptPath = path.join(__dirname, 'backend', 'server_modular.py');
        pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        spawnArgs = [scriptPath];
        spawnOptions = {
            cwd: path.join(__dirname, 'backend'),
            stdio: ['ignore', 'pipe', 'pipe']
        };
    }

    pythonProcess = spawn(pythonCmd, spawnArgs, spawnOptions);

    // Log Python output
    pythonProcess.stdout.on('data', (data) => {
        console.log(`[Python] ${data.toString().trim()}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Python Error] ${data.toString().trim()}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        pythonProcess = null;
    });

    // Wait for server to be ready
    const serverReady = await waitForServer();
    return serverReady;
}

async function createWindow() {
    // Start Python server first
    const serverStarted = await startPythonServer();

    if (!serverStarted) {
        console.error('Failed to start Python server');
        // Show error and quit
        const { dialog } = require('electron');
        dialog.showErrorBox(
            'Backend Error',
            'Failed to start V6rge backend server.\\n\\nPlease check that all dependencies are installed.'
        );
        app.quit();
        return;
    }

    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1000,
        minHeight: 700,
        title: 'V6rge',
        backgroundColor: '#0a0a0f',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false, // Ensure we have access to Node APIs in preload
            nodeIntegrationInWorker: true
        }
    });


    // Auto Update Logic (GitHub Releases)
    // Checks for update. If found, downloads automatically and notifies user to restart.
    autoUpdater.checkForUpdatesAndNotify();

    // Load the frontend
    mainWindow.loadFile('app/index.html');

    // Open DevTools in development (remove for production)
    // mainWindow.webContents.openDevTools();

    // Handle external links - open in default browser
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Create application menu
    createMenu();

    // IPC Handlers
    const { ipcMain, dialog } = require('electron');

    ipcMain.handle('select-folder', async () => {
        const result = await dialog.showOpenDialog(mainWindow, {
            properties: ['openDirectory']
        });
        if (result.canceled) {
            return null;
        } else {
            return result.filePaths[0];
        }
    });

    // Hard Reload Handler
    ipcMain.handle('app:reload', () => {
        if (mainWindow) {
            mainWindow.webContents.reloadIgnoringCache();
        }
    });
}

function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                { role: 'quit', label: 'Exit V6rge' }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Documentation',
                    click: () => shell.openExternal('https://v6rge.com/docs')
                },
                {
                    label: 'Report Issue',
                    click: () => shell.openExternal('https://github.com/v6rge/v6rge-desktop/issues')
                },
                { type: 'separator' },
                {
                    label: 'About V6rge',
                    click: () => {
                        const { dialog } = require('electron');
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About V6rge',
                            message: 'V6rge AI Suite',
                            detail: 'Version 0.1.0 Beta\\n\\nYour personal AI toolkit.\\nRuns 100% locally on your GPU.\\n\\nPowered by open-source models.'
                        });
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    // Kill Python server when app closes
    if (pythonProcess) {
        console.log('🛑 Stopping Python server...');
        pythonProcess.kill();
    }

    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

app.on('before-quit', () => {
    // Ensure Python server is killed
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

// Security: Prevent navigation to external URLs
app.on('web-contents-created', (event, contents) => {
    contents.on('will-navigate', (event, navigationUrl) => {
        const parsedUrl = new URL(navigationUrl);
        if (parsedUrl.origin !== 'file://') {
            event.preventDefault();
            shell.openExternal(navigationUrl);
        }
    });
});

// IPC: Get GPU Info
const { ipcMain } = require('electron');

ipcMain.handle('get-gpu-info', async () => {
    return new Promise((resolve) => {
        // 1. Try nvidia-smi first (Best for AI / VRAM)
        const checkNvidia = () => {
            const cmd = 'nvidia-smi';
            const args = ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'];

            const smi = spawn(cmd, args);
            let output = '';

            smi.stdout.on('data', (data) => output += data.toString());

            smi.on('close', (code) => {
                if (code === 0) {
                    const parts = output.trim().split(',');
                    if (parts.length >= 2) {
                        resolve({
                            success: true,
                            name: parts[0].trim(),
                            vram: parseFloat(parts[1].trim()) / 1024
                        });
                        return;
                    }
                }
                // If nvidia-smi failed, try fallback
                checkFallback();
            });

            smi.on('error', () => {
                checkFallback();
            });
        };

        // 2. Fallback: WMIC (Windows only)
        const checkFallback = () => {
            if (process.platform === 'win32') {
                const wmic = spawn('wmic', ['path', 'win32_VideoController', 'get', 'Name,AdapterRAM', '/format:csv']);
                let output = '';

                wmic.stdout.on('data', (data) => output += data.toString());

                wmic.on('close', (code) => {
                    if (code === 0) {
                        // Skip header (Node,AdapterRAM,Name) -> Get last non-empty line
                        const lines = output.trim().split('\n').filter(l => l.trim());
                        if (lines.length > 1) {
                            // Format: Node,AdapterRAM,Name
                            // Note: wmic CSV often has blank lines or multiple GPUs. We take the first valid one.
                            const lastLine = lines[lines.length - 1];
                            const parts = lastLine.split(',');

                            if (parts.length >= 3) {
                                // parts[1] is AdapterRAM (bytes), parts[2] is Name
                                const ramBytes = parseInt(parts[1]);
                                const name = parts.slice(2).join(',').trim(); // Join rest in case name has commas

                                resolve({
                                    success: true,
                                    name: name || 'Unknown GPU',
                                    vram: isNaN(ramBytes) ? 0 : ramBytes / (1024 * 1024 * 1024) // Bytes to GB
                                });
                                return;
                            }
                        }
                    }
                    resolve({ success: false, error: 'GPU detection failed (WMIC)' });
                });

                wmic.on('error', (err) => resolve({ success: false, error: err.message }));
            } else {
                resolve({ success: false, error: 'GPU detection failed (nvidia-smi not found)' });
            }
        };

        checkNvidia();
    });
});

console.log('V6rge Desktop App Starting...');
