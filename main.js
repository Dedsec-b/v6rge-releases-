const { app, BrowserWindow, Menu, shell, ipcMain, dialog } = require('electron');
const { autoUpdater } = require('electron-updater');
const path = require('path');
const { spawn } = require('child_process');
const fetch = require('node-fetch');

// Keep a global reference to prevent garbage collection
let mainWindow = null;
let pythonProcess = null;
let isSwitchingWindows = false; // Prevent quit during setup->main transition

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

        // Check if backend died
        if (pythonProcess && pythonProcess.exitCode !== null) {
            console.error(`Python server process exited early with code ${pythonProcess.exitCode}`);
            return false;
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

        // ==========================================
        // KEY FIX: Extraction Logic BEFORE Spawn
        // ==========================================
        const zipPath = path.join(process.resourcesPath, 'backend.zip');
        const fs = require('fs');

        if (!fs.existsSync(exePath)) {
            console.log("Backend not found, checking for zip...");
            if (fs.existsSync(zipPath)) {
                console.log("Found backend.zip. Launching Setup Wizard...");

                let setupWindow = new BrowserWindow({
                    width: 450,
                    height: 350,
                    frame: false,
                    resizable: false,
                    alwaysOnTop: true,
                    webPreferences: {
                        nodeIntegration: true,
                        contextIsolation: false
                    },
                    icon: path.join(__dirname, 'app/icon.ico')
                });

                setupWindow.loadFile(path.join(__dirname, 'app/setup.html'));

                try {
                    const destPath = process.resourcesPath;
                    console.log("Starting silent extraction...");
                    const { spawn } = require('child_process');

                    let extractionComplete = false;
                    let extractionChild = null;

                    // If user closes window, intercept to ensure cleanup runs
                    setupWindow.on('close', (e) => {
                        if (!extractionComplete) {
                            e.preventDefault(); // Prevent close so app stays alive for cleanup
                            setupWindow.hide(); // Hide so it looks closed to user

                            console.log("Setup canceled by user. Cleaning up...");
                            if (extractionChild) extractionChild.kill();
                            // The kill triggers 'close' -> reject -> catch block -> cleanup -> setupWindow.destroy()
                        }
                    });

                    await new Promise((resolve, reject) => {
                        // Use LiteralPath and single quotes to handle spaces/special chars correctly
                        // We use spawn to avoid cmd.exe parsing issues
                        // $ProgressPreference = 'SilentlyContinue' speeds up Expand-Archive significantly
                        const psCommand = `$ProgressPreference = 'SilentlyContinue'; Expand-Archive -LiteralPath '${zipPath}' -DestinationPath '${destPath}' -Force`;

                        extractionChild = spawn('powershell', ['-NoProfile', '-Command', psCommand], {
                            windowsHide: true,
                            stdio: 'ignore' // We don't need output, just exit code
                        });

                        extractionChild.on('close', (code) => {
                            if (code === 0) {
                                extractionComplete = true; // Mark as success so closing window doesn't double-kill
                                resolve();
                            } else {
                                // If killed by user, this triggers and rejects, going to Catch -> Cleanup
                                reject(new Error(`PowerShell process exited with code ${code}`));
                            }
                        });

                        extractionChild.on('error', (err) => reject(err));
                    });

                    console.log("Extraction complete. Closing setup...");
                    isSwitchingWindows = true; // Signal we are transitioning
                    setupWindow.close();

                    if (!fs.existsSync(exePath)) return false; // Verify success

                } catch (e) {
                    console.error("Setup Failed:", e);
                    if (setupWindow && !setupWindow.isDestroyed()) {
                        setupWindow.destroy(); // Force close, skipping 'close' event
                    }

                    // Cleanup on failure
                    try {
                        const backendDir = path.join(process.resourcesPath, 'v6rge_backend');
                        if (fs.existsSync(backendDir)) fs.rmdirSync(backendDir, { recursive: true, force: true });
                    } catch (ign) { }

                    const { dialog } = require('electron');
                    dialog.showErrorBox("Setup Error", "Failed to setup V6rge.\n" + e.message);
                    return false;
                }
            } else {
                console.error("Backend executable AND zip missing!");
                const { dialog } = require('electron');
                dialog.showErrorBox("Corrupt Installation", "V6rge backend files are missing.");
                return false;
            }
        }

        pythonCmd = exePath;
        spawnArgs = [];
        spawnOptions = {
            stdio: ['ignore', 'pipe', 'pipe'],
            windowsHide: true
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

    // NOW it is safe to spawn
    pythonProcess = spawn(pythonCmd, spawnArgs, spawnOptions);

    // Log Python output
    pythonProcess.stdout.on('data', (data) => console.log(`[Python] ${data.toString().trim()}`));
    pythonProcess.stderr.on('data', (data) => console.error(`[Python Error] ${data.toString().trim()}`));
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
    isSwitchingWindows = false; // Reset flag


    // Auto Update Logic (GitHub Releases)
    // Detailed Debugging Mode
    const fsLogSync = require('fs');
    const logPath = path.join(app.getPath('desktop'), 'v6rge_update_log.txt');

    function log(msg) {
        const timestamp = new Date().toISOString();
        const line = `[${timestamp}] ${msg}\n`;
        try { fsLogSync.appendFileSync(logPath, line); } catch (e) { }
        console.log(msg);
    }

    log('Initializing Auto Updater...');

    autoUpdater.on('checking-for-update', () => log('Checking for updates...'));

    // Disable auto downloading
    autoUpdater.autoDownload = false;
    autoUpdater.autoInstallOnAppQuit = false;

    // 1. Update Available -> Show Popup
    autoUpdater.on('update-available', (info) => {
        log(`Update available: ${info.version}`);
        if (mainWindow) {
            mainWindow.webContents.send('show-update-popup', info.version);
        }
    });

    autoUpdater.on('update-not-available', () => log('No update available.'));
    autoUpdater.on('error', (err) => log(`Updater Error: ${err.message}`));

    autoUpdater.on('download-progress', (progressObj) => {
        log(`Download speed: ${progressObj.bytesPerSecond} - ${progressObj.percent}%`);
    });

    // 3. Update Downloaded -> Install
    autoUpdater.on('update-downloaded', () => {
        log('Update downloaded. Installing...');
        autoUpdater.quitAndInstall();
    });

    // 2. User Clicked "Update" -> Start Download
    ipcMain.on('start-update', () => {
        log('User requested update. Starting download...');
        autoUpdater.downloadUpdate();
    });

    // Check for updates
    autoUpdater.checkForUpdates();

    // Load the frontend using absolute path to avoid relative path issues
    mainWindow.loadFile(path.join(__dirname, 'app/index.html'));

    // Open DevTools to debug black screen (Disabled    // Open the DevTools.
    mainWindow.webContents.openDevTools();

    // Handle external links - open in default browser
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // ================================
    // HYBRID AGENT EXECUTION
    // ================================
    // const { ipcMain, dialog } = require('electron'); // MOVED TO TOP
    const util = require('util');
    const exec = util.promisify(require('child_process').exec);
    const fs = require('fs/promises');

    ipcMain.handle('execute-local-tool', async (event, { tool, command }) => {
        console.log(`[Hybrid Agent] Executing Local Tool: ${tool} -> ${command}`);

        try {
            if (tool === 'terminal') {
                // Execute Shell Command
                const { stdout, stderr } = await exec(command, {
                    shell: 'powershell.exe',
                    maxBuffer: 1024 * 1024 * 5 // 5MB buffer
                });
                let output = stdout.trim();
                if (stderr) output += `\n[STDERR]\n${stderr.trim()}`;
                return { status: 'success', output: output || "(No output)" };

            } else if (tool === 'read_file') {
                // Read File
                // Sanitize / Resolve Path
                let targetPath = command;
                // Basic resolving relative to User Home if not absolute
                if (!path.isAbsolute(targetPath)) {
                    targetPath = path.join(require('os').homedir(), 'Desktop', targetPath);
                }
                const content = await fs.readFile(targetPath, 'utf-8');
                return { status: 'success', output: content };

            } else if (tool === 'list_dir') {
                // List Dir
                let targetPath = command;
                if (!targetPath || targetPath === '.') targetPath = require('os').homedir() + "\\Desktop";
                if (!path.isAbsolute(targetPath)) {
                    targetPath = path.join(require('os').homedir(), 'Desktop', targetPath);
                }

                const files = await fs.readdir(targetPath, { withFileTypes: true });
                const output = files.map(f => `[${f.isDirectory() ? 'DIR' : 'FILE'}] ${f.name}`).join('\n');
                return { status: 'success', output: output };

            } else if (tool === 'write_file') {
                // Write File
                // Command format: "path|content" or just path and we need content separately?
                // The Cloud Agent sends [TOOL:write_file:path]. Wait, where is the content?
                // Standard Agent protocols usually send content in the tool call.
                // Qwen Tool Call format: [TOOL:write_file:path] ... <content>?
                // User prompt says: [TOOL:write_file:path] -> Reads local file. Wait, that's read.

                // If the agent sends [TOOL:write_file:path], it implies the content is separate.
                // We don't have a content argument in the current regex!
                // The regex is `\[TOOL:(\w+):(.*?)\]`. It captures the args inside the brackets.
                // If the agent writes `[TOOL:write_file:path|content]`, it works.
                // But writing a whole file in one line is hard.

                // Alternative: `terminal` tool with `echo "content" > file`.
                // Qwen might be trying to use `terminal`.
                // In the screenshot, it did: `New-Item ...`. That is `terminal` tool!
                // The user said "I don't see any file".
                // That means `terminal` execution failed silently or path was wrong.

                // BUT in the last screenshot, it shows `<TOOL:write_file:...>`.
                // This confirms it tried `write_file`.

                // Implementation:
                // I will assume the args are `path|content`. If not, I'll error.
                // Or better: Let's support `terminal` better.

                // But for `write_file`, I'll implement a simple split.
                const parts = command.split('|');
                let targetPath = parts[0];
                let content = parts.slice(1).join('|'); // Rejoin rest

                if (!path.isAbsolute(targetPath)) {
                    targetPath = path.join(require('os').homedir(), 'Desktop', targetPath);
                }

                await fs.writeFile(targetPath, content, 'utf-8');
                return { status: 'success', output: `File written to ${targetPath}` };
            }

            return { status: 'error', error: 'Unknown tool type' };

        } catch (err) {
            console.error(`Local Execution Failed: ${err.message}`);
            return { status: 'error', error: err.message };
        }
    });

    // Create application menu
    createMenu();

    // IPC Handlers
    // const { ipcMain, dialog } = require('electron'); // This line is now handled above

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
    // If we are just switching windows (Setup -> Main), don't quit
    if (isSwitchingWindows) return;

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
// const { ipcMain } = require('electron'); // MOVED TO TOP

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
