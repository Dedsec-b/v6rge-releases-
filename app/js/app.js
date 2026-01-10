import { api } from './modules/api.js';
import { ModelManager } from './modules/model-manager.js';
import { ChatView } from './modules/chat.js';
import { ToolManager } from './modules/tools.js';
import { SettingsManager } from './modules/settings.js';

class App {
    constructor() {
        this.api = api;
        this.modelManager = null;
        this.chatView = null;
        this.toolManager = null;

        // State
        this.currentView = 'dashboard';
        this.isCloudMode = localStorage.getItem('cloud-mode') === 'true';

        this.init();
    }

    async init() {
        console.log('Initializing V6rge Desktop...');

        // Initialize Modules
        this.modelManager = new ModelManager();
        this.toolManager = new ToolManager();
        this.settingsManager = new SettingsManager();
        this.chatView = new ChatView();

        // Setup UI
        this.setupNavigation();
        this.setupCloudMode();
        this.setupSidebar();

        // Check backend connection
        const isConnected = await this.api.checkConnection();
        this.updateConnectionStatus(isConnected);

        // Load initial data
        if (isConnected) {
            await this.modelManager.refresh();
        }

        // Restore Scroll Memory (if reload occurred)
        const restoreView = sessionStorage.getItem('restore_view');
        if (restoreView) {
            this.switchView(restoreView);
            const restoreScroll = sessionStorage.getItem('restore_scroll');
            if (restoreScroll) {
                setTimeout(() => {
                    const viewEl = document.getElementById(`${restoreView}-view`);
                    if (viewEl) {
                        viewEl.scrollTo({
                            top: parseInt(restoreScroll),
                            behavior: 'smooth'
                        });
                    }
                }, 500); // Smooth delay for cinematic restore
            }
            // Cleanup one-time restore
            sessionStorage.removeItem('restore_view');
            sessionStorage.removeItem('restore_scroll');
        }
    }

    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item[data-view]');
        const toolCards = document.querySelectorAll('.tool-card[data-tool]');

        // Navigation Handler
        const handleNav = (viewId) => {
            this.switchView(viewId);
        };

        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                handleNav(item.dataset.view);
            });
        });

        toolCards.forEach(card => {
            card.addEventListener('click', () => {
                handleNav(card.dataset.tool);
            });
        });
    }

    switchView(viewId) {
        // Update State
        this.currentView = viewId;

        // Update UI Classes
        document.querySelectorAll('.nav-item').forEach(el => {
            el.classList.toggle('active', el.dataset.view === viewId);
        });

        document.querySelectorAll('.view').forEach(el => {
            el.classList.toggle('active', el.id === `${viewId}-view`);
        });

        // Update Breadcrumb
        const breadcrumb = document.querySelector('.breadcrumb');
        if (breadcrumb) {
            breadcrumb.textContent = this.getReadableName(viewId);
        }

        // Trigger generic view events if needed
        window.dispatchEvent(new CustomEvent('viewChanged', { detail: { view: viewId } }));
    }

    getReadableName(viewId) {
        const names = {
            'dashboard': 'Dashboard',
            'chat': 'Chat',
            'image-gen': 'Image Generation',
            'music-gen': 'Music Generation',
            'video-gen': 'Video Generation',
            'settings': 'Settings'
        };
        return names[viewId] || viewId.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    setupSidebar() {
        const sidebar = document.getElementById('sidebar');
        const toggle = document.getElementById('collapse-toggle');

        if (localStorage.getItem('sidebar-collapsed') === 'true') {
            sidebar.classList.add('collapsed');
        }

        if (toggle) {
            toggle.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
            });
        }
    }

    setupCloudMode() {
        const toggle = document.getElementById('cloud-mode-toggle');
        if (!toggle) return;

        toggle.checked = this.isCloudMode;
        this.toggleCloudUI(this.isCloudMode);

        toggle.addEventListener('change', (e) => {
            this.isCloudMode = e.target.checked;
            localStorage.setItem('cloud-mode', this.isCloudMode);
            this.toggleCloudUI(this.isCloudMode);

            if (!this.isCloudMode) {
                this.api.setBaseUrl('http://localhost:5000');
                this.modelManager.refresh();
            }
        });

        // Connect Button Logic
        const connectBtn = document.getElementById('cloud-connect-btn');
        const urlInput = document.getElementById('cloud-backend-url');

        if (connectBtn && urlInput) {
            // Restore saved URL
            urlInput.value = localStorage.getItem('cloud-backend-url') || '';

            connectBtn.addEventListener('click', async () => {
                const url = urlInput.value.trim();
                if (!url) return;

                connectBtn.textContent = 'Testing...';
                connectBtn.disabled = true;

                const success = await this.api.checkConnection(url);
                if (success) {
                    this.api.setBaseUrl(url);
                    localStorage.setItem('cloud-backend-url', url);
                    connectBtn.textContent = 'Connected';
                    connectBtn.classList.add('btn-success');
                    this.modelManager.refresh();
                } else {
                    connectBtn.textContent = 'Failed';
                    connectBtn.classList.add('btn-error');
                    setTimeout(() => {
                        connectBtn.textContent = 'Connect';
                        connectBtn.classList.remove('btn-error');
                    }, 2000);
                }
                connectBtn.disabled = false;
            });
        }
    }

    toggleCloudUI(isCloud) {
        const section = document.getElementById('cloud-url-section');
        const manager = document.querySelector('.model-manager');
        if (section) section.style.display = isCloud ? 'flex' : 'none';
        if (manager) manager.classList.toggle('cloud-active', isCloud);
    }

    updateConnectionStatus(isConnected) {
        // Optional: Add a visual indicator in the UI
        console.log(`Backend Status: ${isConnected ? 'Online' : 'Offline'}`);
    }
}

// Global instance
window.app = new App();
