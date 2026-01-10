// V6rge Desktop - Frontend with Backend Integration

// Configurable API URL (default to localhost, can be changed to ngrok in Settings)
let API_URL = localStorage.getItem('backend-url') || 'http://localhost:5000';

// Track if cloud mode is active (non-localhost backend)
window.isCloudMode = !API_URL.includes('localhost') && !API_URL.includes('127.0.0.1');

document.addEventListener('DOMContentLoaded', () => {
    // ================================
    // VIEW NAVIGATION
    // ================================
    const navItems = document.querySelectorAll('.nav-item[data-view]');
    const views = document.querySelectorAll('.view');
    const breadcrumb = document.querySelector('.breadcrumb');
    const toolCards = document.querySelectorAll('.tool-card[data-tool]');

    // View name mapping
    const viewNames = {
        'dashboard': 'Dashboard',
        'chat': 'Chat',
        'image-gen': 'Image Generation',
        'music-gen': 'Music Generation',
        'tts': 'Text to Speech',
        '3d-gen': '3D Generation',
        'video-gen': 'Video Generation',
        'upscale': 'Image Upscaler',
        'bg-remove': 'Background Removal',
        'vocal-sep': 'Vocal Separator',
        'settings': 'Settings'
    };

    // ================================
    // SIDEBAR COLLAPSE
    // ================================
    const sidebar = document.getElementById('sidebar');
    const collapseToggle = document.getElementById('collapse-toggle');

    // Restore collapsed state from localStorage
    if (localStorage.getItem('sidebar-collapsed') === 'true') {
        sidebar.classList.add('collapsed');
    }

    if (collapseToggle) {
        collapseToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
        });
    }

    // ================================
    // CLOUD MODE TOGGLE
    // ================================
    const cloudModeToggle = document.getElementById('cloud-mode-toggle');
    const cloudUrlSection = document.getElementById('cloud-url-section');
    const cloudUrlInput = document.getElementById('cloud-backend-url');
    const cloudConnectBtn = document.getElementById('cloud-connect-btn');
    const cloudStatus = document.getElementById('cloud-status');
    const modelManager = document.querySelector('.model-manager');

    // Track cloud mode state globally
    window.isCloudMode = localStorage.getItem('cloud-mode') === 'true';

    // Restore cloud mode state on load
    if (cloudModeToggle) {
        cloudModeToggle.checked = window.isCloudMode;
        if (window.isCloudMode) {
            cloudUrlSection.style.display = 'flex';
            if (modelManager) modelManager.classList.add('cloud-active');
            // Restore saved URL
            const savedUrl = localStorage.getItem('cloud-backend-url');
            if (savedUrl) {
                cloudUrlInput.value = savedUrl;
                API_URL = savedUrl;
            }
        }

        cloudModeToggle.addEventListener('change', function () {
            window.isCloudMode = this.checked;
            localStorage.setItem('cloud-mode', this.checked);

            if (this.checked) {
                cloudUrlSection.style.display = 'flex';
                if (modelManager) modelManager.classList.add('cloud-active');
            } else {
                cloudUrlSection.style.display = 'none';
                if (modelManager) modelManager.classList.remove('cloud-active');
                // Reset to local backend
                API_URL = 'http://localhost:5000';
                localStorage.setItem('backend-url', API_URL);
                cloudStatus.textContent = '';
                cloudStatus.className = 'cloud-status';
            }
        });
    }

    if (cloudConnectBtn) {
        cloudConnectBtn.addEventListener('click', async () => {
            const url = cloudUrlInput.value.trim().replace(/\/$/, ''); // Remove trailing slash
            if (!url) {
                cloudStatus.textContent = 'Enter a URL';
                cloudStatus.className = 'cloud-status error';
                return;
            }

            cloudConnectBtn.disabled = true;
            cloudConnectBtn.textContent = 'Testing...';
            cloudStatus.textContent = '';

            try {
                const response = await fetch(`${url}/health`, { method: 'GET', mode: 'cors' });
                if (response.ok) {
                    API_URL = url;
                    localStorage.setItem('backend-url', url);
                    localStorage.setItem('cloud-backend-url', url);
                    window.isCloudMode = !url.includes('localhost') && !url.includes('127.0.0.1');
                    cloudStatus.textContent = '✓ Connected';
                    cloudStatus.className = 'cloud-status connected';
                } else {
                    throw new Error('Server returned error');
                }
            } catch (err) {
                cloudStatus.textContent = '✗ Failed to connect';
                cloudStatus.className = 'cloud-status error';
            }

            cloudConnectBtn.disabled = false;
            cloudConnectBtn.textContent = 'Connect';
        });
    }

    // ================================
    // MODEL MANAGER
    // ================================

    async function initModelManager() {
        console.log('Initializing Model Manager...');

        try {
            const response = await fetch(`${API_URL}/models/status`);
            const data = await response.json();

            // Update total downloaded
            document.getElementById('total-downloaded').textContent = `${data.total_downloaded_gb} GB`;

            // Update each model status
            Object.keys(data.models).forEach(modelId => {
                const modelData = data.models[modelId];
                const modelItem = document.querySelector(`.model-item[data-model-id="${modelId}"]`);

                if (modelItem) {
                    const statusSpan = modelItem.querySelector('.model-status');
                    const downloadBtn = modelItem.querySelector('.btn-download');

                    if (modelData.downloaded) {
                        statusSpan.textContent = 'Ready';
                        statusSpan.setAttribute('data-status', 'ready');
                        downloadBtn.textContent = 'Delete';
                        downloadBtn.classList.remove('btn-download');
                        downloadBtn.classList.add('btn-delete');
                    } else {
                        statusSpan.textContent = 'Not Downloaded';
                        statusSpan.setAttribute('data-status', 'not-downloaded');

                        // Fix: Reset button to Download state
                        if (downloadBtn) {
                            downloadBtn.textContent = 'Download';
                            downloadBtn.classList.remove('btn-delete', 'btn-secondary');
                            downloadBtn.classList.add('btn-download', 'btn-primary');
                            downloadBtn.disabled = false;
                        } else {
                            // Handle case where it might be a delete button now
                            const deleteBtn = modelItem.querySelector('.btn-delete');
                            if (deleteBtn) {
                                deleteBtn.textContent = 'Download';
                                deleteBtn.classList.remove('btn-delete', 'btn-secondary');
                                deleteBtn.classList.add('btn-download', 'btn-primary');
                                deleteBtn.disabled = false;
                            }
                        }
                    }
                }
            });

            // Populate Dropdowns in Tool Views
            populateModelSelectors(data.models);

        } catch (error) {
            console.error('Error loading model status:', error);
        }
    }

    function populateModelSelectors(models) {
        const selectors = document.querySelectorAll('.model-selector');
        selectors.forEach(select => {
            const category = select.dataset.category;
            const specific = select.dataset.specific;
            const currentVal = select.value;

            // Filter available models
            const availableModels = Object.entries(models).filter(([id, m]) => {
                // Must be downloaded
                if (!m.downloaded) return false;

                // Must match category
                if (m.type !== category) return false;

                // If specific ID required (for tools like upscaler/rembg)
                if (specific && id !== specific) return false;

                return true;
            });

            // Save previous selection if valid
            const previousSelection = select.value;

            // Clear existing options
            select.innerHTML = '';

            // Add default placeholder
            const placeholder = document.createElement('option');
            placeholder.text = "Select Model...";
            placeholder.value = "";
            placeholder.disabled = true;
            placeholder.selected = true;
            select.appendChild(placeholder);

            if (availableModels.length === 0) {
                placeholder.text = "No models downloaded";
            } else {
                availableModels.forEach(([id, m]) => {
                    const opt = document.createElement('option');
                    opt.value = id;
                    opt.text = m.name || id;
                    select.appendChild(opt);
                });

                // Restore selection if valid, otherwise keep placeholder
                if (previousSelection && Array.from(select.options).some(o => o.value === previousSelection)) {
                    select.value = previousSelection;
                } else {
                    select.value = ""; // Ensure placeholder is selected
                }
            }

            // Custom UI - DISABLED: Now handled by tools.js ToolManager
            // setupCustomSelect(select);
        });
    }

    function setupAdvancedControls() {
        // Helper to link slider to value display
        const linkSlider = (sliderId, displayId) => {
            const slider = document.getElementById(sliderId);
            const display = document.getElementById(displayId);
            if (slider && display) {
                slider.addEventListener('input', () => {
                    display.textContent = slider.value;
                });
            }
        };

        linkSlider('image-steps', 'val-image-steps');
        linkSlider('image-guidance', 'val-image-guidance');
        linkSlider('music-guidance', 'val-music-guidance');
        linkSlider('music-temp', 'val-music-temp');
        linkSlider('video-frames', 'val-video-frames');
        linkSlider('video-steps', 'val-video-steps');
        linkSlider('3d-steps', 'val-3d-steps');
    }

    function setupCustomSelect(select) {
        // Wrapper check
        if (select.nextElementSibling && select.nextElementSibling.classList.contains('custom-select-wrapper')) {
            select.nextElementSibling.remove();
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'custom-select-wrapper';

        // Create custom select UI
        const trigger = document.createElement('div');
        trigger.className = 'custom-select-trigger';
        const selected = select.options[select.selectedIndex];
        trigger.innerHTML = `<span>${selected ? selected.text : 'Select'}</span>`;

        const optionsPanel = document.createElement('div');
        optionsPanel.className = 'custom-options';

        Array.from(select.options).forEach(opt => {
            const div = document.createElement('div');
            div.className = 'custom-option ' + (opt.disabled ? 'disabled' : '') + (opt.selected ? 'selected' : '');
            div.textContent = opt.text;
            div.dataset.value = opt.value;

            if (!opt.disabled) {
                div.addEventListener('click', () => {
                    select.value = opt.value;
                    trigger.querySelector('span').textContent = opt.text;
                    optionsPanel.querySelectorAll('.custom-option').forEach(o => o.classList.remove('selected'));
                    div.classList.add('selected');
                    wrapper.classList.remove('open');
                    select.dispatchEvent(new Event('change'));
                });
            }
            optionsPanel.appendChild(div);
        });

        trigger.addEventListener('click', (e) => {
            document.querySelectorAll('.custom-select-wrapper.open').forEach(w => {
                if (w !== wrapper) w.classList.remove('open');
            });
            wrapper.classList.toggle('open');
            e.stopPropagation();
        });

        wrapper.appendChild(trigger);
        wrapper.appendChild(optionsPanel);

        select.style.display = 'none';
        select.parentNode.insertBefore(wrapper, select.nextSibling);
    }

    document.addEventListener('click', () => {
        document.querySelectorAll('.custom-select-wrapper.open').forEach(w => w.classList.remove('open'));
    });

    // Handle download button clicks
    document.addEventListener('click', async (e) => {
        // Handle Cancel Button
        if (e.target.closest('.btn-cancel-download')) {
            const btn = e.target.closest('.btn-cancel-download');
            const modelId = btn.dataset.model;
            btn.disabled = true; // Prevent double clicks

            try {
                const response = await fetch(`${API_URL}/models/cancel/${modelId}`, { method: 'POST' });
                const data = await response.json();
                console.log('Cancel response:', data);
            } catch (err) {
                console.error('Failed to cancel:', err);
            }
            return;
        }

        if (e.target.classList.contains('btn-download')) {
            const modelId = e.target.getAttribute('data-model');
            const modelItem = e.target.closest('.model-item');
            const statusSpan = modelItem.querySelector('.model-status');

            e.target.disabled = true;
            e.target.textContent = 'Starting...';
            statusSpan.textContent = 'Downloading';
            statusSpan.setAttribute('data-status', 'downloading');

            // Start download in background
            fetch(`${API_URL}/models/download/${modelId}`, {
                method: 'POST'
            }).then(async response => {
                const data = await response.json();
                if (data.status === 'success') {
                    statusSpan.textContent = 'Ready';
                    statusSpan.setAttribute('data-status', 'ready');
                    e.target.textContent = 'Delete';
                    e.target.classList.remove('btn-download');
                    e.target.classList.add('btn-delete');
                    initModelManager();
                } else {
                    throw new Error(data.error || 'Download failed');
                }
                e.target.disabled = false;
            }).catch(error => {
                alert(`Download failed: ${error.message}`);
                statusSpan.textContent = 'Not Downloaded';
                statusSpan.setAttribute('data-status', 'not-downloaded');
                e.target.textContent = 'Download';
                e.target.disabled = false;
            });

            // Poll for progress with detailed UI
            const pollProgress = async () => {
                try {
                    const progressRes = await fetch(`${API_URL}/models/progress/${modelId}`);
                    const progress = await progressRes.json();

                    if (progress.status === 'downloading') {
                        // Create or update progress UI
                        let progressContainer = modelItem.querySelector('.progress-container');
                        if (!progressContainer) {
                            progressContainer = document.createElement('div');
                            progressContainer.className = 'progress-container';
                            progressContainer.style.cssText = `
                                margin-top: 10px;
                                background: var(--bg-secondary);
                                border-radius: 4px;
                                padding: 8px;
                                border: 1px solid var(--border-subtle);
                            `;
                            // Insert before the actions div or append to item
                            modelItem.appendChild(progressContainer);
                        }

                        const percent = progress.progress ? progress.progress.toFixed(1) : 0;
                        const speed = progress.speed ? (progress.speed / 1024 / 1024).toFixed(1) : '0.0';
                        const eta = progress.eta ? Math.ceil(progress.eta) : 0;
                        const downloaded = progress.downloaded ? (progress.downloaded / 1024 / 1024).toFixed(0) : '0';
                        const total = progress.total ? (progress.total / 1024 / 1024).toFixed(0) : '?';

                        // Format ETA
                        let etaText = `${eta}s`;
                        if (eta > 60) etaText = `${Math.floor(eta / 60)}m ${eta % 60}s`;

                        progressContainer.innerHTML = `
                            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: var(--text-primary); align-items: center;">
                                <span>${progress.message || 'Downloading...'}</span>
                                <div style="display: flex; gap: 8px; align-items: center;">
                                    <span>${percent}%</span>
                                    <button class="btn-cancel-download" data-model="${modelId}" style="
                                        background: none; 
                                        border: none; 
                                        color: #ef4444; 
                                        cursor: pointer; 
                                        padding: 0 4px; 
                                        font-size: 14px; 
                                        line-height: 1;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                    " title="Cancel Download">✕</button>
                                </div>
                            </div>
                            <div style="height: 6px; background: var(--bg-tertiary); border-radius: 3px; overflow: hidden; margin-bottom: 6px;">
                                <div style="width: ${percent}%; height: 100%; background: var(--accent-primary); transition: width 0.3s ease;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted);">
                                <span>📦 ${downloaded}MB / ${total}MB</span>
                                <span>⚡ ${speed} MB/s</span>
                                <span>⏳ ${etaText} left</span>
                            </div>
                        `;

                        e.target.textContent = 'Downloading...';
                        e.target.disabled = true;
                        statusSpan.style.display = 'none'; // Hide default status

                        setTimeout(pollProgress, 500); // Poll every 500ms
                    } else if (progress.status === 'init') {
                        setTimeout(pollProgress, 500);
                    } else if (progress.status === 'loading') {
                        const container = modelItem.querySelector('.progress-container');
                        if (container) container.innerHTML = '<div style="color: var(--accent-primary); font-size: 12px;">Finalizing installation...</div>';
                        setTimeout(pollProgress, 500);
                    } else if (progress.status === 'complete') {
                        // Cleanup
                        const container = modelItem.querySelector('.progress-container');
                        if (container) container.remove();
                        statusSpan.style.display = ''; // Show status again

                        e.target.textContent = 'Delete';
                        statusSpan.textContent = 'Ready';
                        statusSpan.setAttribute('data-status', 'ready');
                        e.target.classList.remove('btn-download');
                        e.target.classList.add('btn-delete');
                        e.target.disabled = false;
                    } else if (progress.status === 'error') {
                        const container = modelItem.querySelector('.progress-container');
                        if (container) container.innerHTML = `<div style="color: #ef4444; font-size: 12px;">Error: ${progress.message}</div>`;
                        e.target.textContent = 'Retry';
                        e.target.disabled = false;
                    }
                } catch (err) {
                    console.error('Poll error', err);
                    setTimeout(pollProgress, 1000);
                }
            };

            setTimeout(pollProgress, 200);
        }

        if (e.target.classList.contains('btn-delete')) {
            const modelId = e.target.getAttribute('data-model');

            if (!confirm('Delete this model? You can re-download it later.')) {
                return;
            }

            try {
                const response = await fetch(`${API_URL}/models/delete/${modelId}`, {
                    method: 'DELETE'
                });

                const data = await response.json();

                if (data.status === 'success') {
                    // Refresh status
                    initModelManager();
                } else {
                    throw new Error(data.error || 'Delete failed');
                }
            } catch (error) {
                alert(`Delete failed: ${error.message}`);
            }
        }

    });

    // Initialize model manager on dashboard view
    initModelManager();

    function switchView(viewId) {
        // Update nav items
        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.view === viewId) {
                item.classList.add('active');
            }
        });

        // Update views
        views.forEach(view => {
            view.classList.remove('active');
            if (view.id === `${viewId}-view`) {
                view.classList.add('active');
            }
        });

        // Update breadcrumb
        breadcrumb.textContent = viewNames[viewId] || viewId;
    }

    // Nav item clicks
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            switchView(item.dataset.view);
        });
    });

    // Tool card clicks (from dashboard)
    toolCards.forEach(card => {
        card.addEventListener('click', () => {
            const tool = card.dataset.tool;
            switchView(tool);
        });
    });

    // ================================
    // CHAT FUNCTIONALITY
    // ================================
    const chatInput = document.querySelector('.chat-input');
    const sendBtn = document.querySelector('.send-btn');
    const chatMessages = document.getElementById('chat-messages');

    if (chatInput && sendBtn) {
        // Auto-resize textarea
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
        });

        // Send on Enter (not Shift+Enter)
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendBtn.addEventListener('click', sendMessage);
    }

    // Local chat history
    let chatHistory = [];

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Clear empty state
        const emptyState = chatMessages.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        // Add user message to UI
        addMessage('user', message);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add user message to history
        chatHistory.push({ role: 'user', content: message });

        // Show loading
        const loadingDiv = addMessage('assistant', 'Thinking');
        loadingDiv.classList.add('loading');

        try {
            const modelSelect = document.getElementById('chat-model-selector');
            const modelId = modelSelect ? modelSelect.value : '';

            // Skip model check when in cloud mode (cloud backend handles model selection)
            if (!window.isCloudMode && !modelId && modelSelect) {
                loadingDiv.remove();
                addMessage('assistant', '⚠️ Please select a model from the dropdown above to start chatting.');
                return;
            }

            const response = await fetch(`${API_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: new URLSearchParams({
                    message,
                    // Send accumulated history
                    history: JSON.stringify(chatHistory),
                    model_id: modelId
                })
            });

            const data = await response.json();

            // Remove loading
            loadingDiv.remove();

            // Add response with typewriter effect
            await typewriterMessage(data.response);

            // Add assistant response to history
            chatHistory.push({ role: 'assistant', content: data.response });

            // Handle tool results
            if (data.tool_result) {
                handleToolResult(data.tool_result);
            }

        } catch (error) {
            loadingDiv.remove();
            addMessage('assistant', `Error: ${error.message}`);
        }
    }

    // Typewriter effect for smooth text animation
    async function typewriterMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        const isShortMessage = content.length < 50 && !content.includes('\n');
        const bodyClass = !isShortMessage ? 'message-content markdown-body' : 'message-content';

        const contentDiv = document.createElement('div');
        contentDiv.className = bodyClass;
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);

        // For very short messages, just show instantly
        if (isShortMessage) {
            contentDiv.innerHTML = escapeHtml(content);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return messageDiv;
        }

        // Typewriter animation - show characters progressively
        let displayedText = '';
        const speed = 5; // milliseconds per character (very fast)
        const chunkSize = 3; // characters per frame for faster rendering

        for (let i = 0; i < content.length; i += chunkSize) {
            displayedText = content.slice(0, i + chunkSize);

            // Parse markdown as we go for proper formatting
            if (typeof marked !== 'undefined') {
                marked.setOptions({ breaks: true, gfm: true, headerIds: false });
                contentDiv.innerHTML = marked.parse(displayedText);
            } else {
                contentDiv.innerHTML = escapeHtml(displayedText);
            }

            chatMessages.scrollTop = chatMessages.scrollHeight;
            await new Promise(r => setTimeout(r, speed));
        }


        return messageDiv;
    }

    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        // For assistant messages, parse markdown; for user/loading, escape HTML
        let formattedContent;
        const isShortMessage = content.length < 50 && !content.includes('\n');

        if (role === 'assistant' && typeof marked !== 'undefined' && !isShortMessage) {
            // Configure marked for better rendering
            marked.setOptions({
                breaks: true,  // Convert \n to <br>
                gfm: true,     // GitHub Flavored Markdown
                headerIds: false
            });
            formattedContent = marked.parse(content);
        } else {
            formattedContent = escapeHtml(content);
        }

        // Only add markdown-body class for actual markdown content
        const bodyClass = (role === 'assistant' && !isShortMessage) ? 'message-content markdown-body' : 'message-content';

        messageDiv.innerHTML = `
            <div class="${bodyClass}">${formattedContent}</div>
        `;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }

    function handleToolResult(result) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'message assistant';

        let contentHtml = '';

        // Improved URL construction - ensure no double slashes and proper format
        let mediaUrl;
        if (result.url) {
            if (result.url.startsWith('http://') || result.url.startsWith('https://')) {
                mediaUrl = result.url;
            } else {
                // Normalize URL path
                const cleanUrl = result.url.replace(/^\/+/, ''); // Remove leading slashes
                mediaUrl = `${API_URL}/${cleanUrl}`;
            }
        }

        if (result.type === 'image') {
            // Generate unique ID for retry logic
            const imgId = `img-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

            contentHtml = `
                <div class="generated-content">
                    <div class="content-label">Generated Image</div>
                    <img 
                        id="${imgId}"
                        src="${mediaUrl}" 
                        alt="Generated image" 
                        style="max-width: 100%; border-radius: 8px; display: block;"
                        onerror="
                            console.error('Image load failed:', this.src);
                            const retryCount = parseInt(this.dataset.retry || 0);
                            if (retryCount < 3) {
                                setTimeout(() => {
                                    this.dataset.retry = retryCount + 1;
                                    this.src = this.src.split('?')[0] + '?retry=' + retryCount;
                                }, 1000 * (retryCount + 1));
                            } else {
                                this.alt = '❌ Image failed to load after 3 retries';
                                this.style.display = 'none';
                                this.parentElement.innerHTML += '<div style=\\'color: #ef4444; padding: 12px; text-align: center;\\'>Failed to load image. URL: ' + this.src.split('?')[0] + '</div>';
                            }
                        "
                        onload="console.log('Image loaded successfully:', this.src);"
                    >
                </div>
            `;
        } else if (result.type === 'audio') {
            contentHtml = `
                <div class="generated-content">
                    <div class="content-label">Generated Audio</div>
                    <audio controls src="${mediaUrl}" style="width: 100%; max-width: 500px;">
                        Your browser does not support audio playback.
                    </audio>
                </div>
            `;
        } else if (result.type === 'video') {
            contentHtml = `
                <div class="generated-content">
                    <div class="content-label">Generated Video</div>
                    <video controls src="${mediaUrl}" style="max-width: 100%; border-radius: 8px;">
                        Your browser does not support video playback.
                    </video>
                </div>
            `;
        } else if (result.type === 'model') {
            contentHtml = `
                <div class="generated-content">
                    <div class="content-label">3D Model Ready</div>
                    <a href="${mediaUrl}" download class="download-link">
                        <span>📦</span> Download ${result.format.toUpperCase()} Model
                    </a>
                </div>
            `;
        }

        resultDiv.innerHTML = `
            <div class="message-content">${contentHtml}</div>
        `;

        chatMessages.appendChild(resultDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ================================
    // SETTINGS
    // ================================
    const toggles = document.querySelectorAll('.toggle input');
    toggles.forEach(toggle => {
        toggle.addEventListener('change', () => {
            console.log(`Toggle ${toggle.closest('.settings-row').querySelector('.settings-label').textContent}: ${toggle.checked}`);
        });
    });

    // ================================
    // KEYBOARD SHORTCUTS
    // ================================
    document.addEventListener('keydown', (e) => {
        // Cmd/Ctrl + K - Focus chat
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            switchView('chat');
            chatInput?.focus();
        }

        // Cmd/Ctrl + , - Settings
        if ((e.metaKey || e.ctrlKey) && e.key === ',') {
            e.preventDefault();
            switchView('settings');
        }

        // Escape - Back to dashboard
        if (e.key === 'Escape') {
            switchView('dashboard');
        }
    });

    // ================================
    // SETTINGS MANAGEMENT
    // ================================

    // Load settings from localStorage
    async function loadSettings() {
        let gpuName = localStorage.getItem('gpu-name');
        let gpuVram = localStorage.getItem('gpu-vram');

        // Auto-detect if missing
        if (!gpuName || !gpuVram) {
            console.log('Auto-detecting GPU...');
            try {
                if (window.v6rge && window.v6rge.getGpuInfo) {
                    const info = await window.v6rge.getGpuInfo();
                    if (info.success) {
                        gpuName = info.name;
                        gpuVram = Math.round(info.vram); // Round to nearest GB

                        // Save detected values
                        localStorage.setItem('gpu-name', gpuName);
                        localStorage.setItem('gpu-vram', gpuVram);
                        console.log('GPU Detected:', gpuName, gpuVram + 'GB');
                    } else {
                        console.warn('GPU Detection failed:', info.error);
                    }
                }
            } catch (err) {
                console.error('Error detecting GPU:', err);
            }
        }

        if (gpuName) {
            const gpuNameInput = document.getElementById('gpu-name');
            if (gpuNameInput) gpuNameInput.value = gpuName;
        }

        if (gpuVram) {
            const gpuVramInput = document.getElementById('gpu-vram');
            if (gpuVramInput) gpuVramInput.value = gpuVram;
        }

        // Update header display
        updateGPUDisplay(gpuName, gpuVram);
    }

    function updateGPUDisplay(name, vram) {
        const statusDisplay = document.getElementById('gpu-status-display');
        const nameDisplay = document.getElementById('gpu-display-name');
        const vramDisplay = document.getElementById('gpu-display-vram');

        if (name && vram && nameDisplay && vramDisplay && statusDisplay) {
            nameDisplay.textContent = name;
            vramDisplay.textContent = `${vram}GB`;
            statusDisplay.style.display = 'flex';
        } else if (statusDisplay) {
            statusDisplay.style.display = 'none';
        }
    }

    async function checkBackendStatus() {
        const statusDiv = document.getElementById('connection-status');
        const statusDisplay = document.getElementById('backend-status-display');

        try {
            const response = await fetch(`${API_URL}/health`);
            if (response.ok) {
                if (statusDiv) {
                    statusDiv.innerHTML = '<span style="color: #10b981;">✓ Connected to Backend</span>';
                }
                if (statusDisplay) {
                    statusDisplay.style.display = 'flex';
                    statusDisplay.querySelector('.status-dot').style.background = '#10b981';
                    statusDisplay.querySelector('.status-text').textContent = 'Backend Connected';
                }
                // Initialize models if connected
                initModelManager();
            } else {
                throw new Error('Backend not ready');
            }
        } catch (error) {
            console.log('Backend status check failed:', error);
            if (statusDiv) {
                statusDiv.innerHTML = '<span style="color: #ef4444;">✕ Not Connected (Is server running?)</span>';
            }
            if (statusDisplay) {
                statusDisplay.style.display = 'flex';
                statusDisplay.querySelector('.status-dot').style.background = '#ef4444';
                statusDisplay.querySelector('.status-text').textContent = 'Backend Offline';
            }
        }
    }

    // Load backend URL into settings input
    const backendUrlInput = document.getElementById('backend-url-input');
    if (backendUrlInput) {
        backendUrlInput.value = API_URL;
    }

    // Check backend status
    checkBackendStatus();

    // Setup Advanced Controls
    setupAdvancedControls();

    // Save backend URL
    const saveBackendBtn = document.getElementById('save-backend-btn');
    if (saveBackendBtn) {
        saveBackendBtn.addEventListener('click', () => {
            const newUrl = document.getElementById('backend-url-input').value.trim();
            if (newUrl) {
                localStorage.setItem('backend-url', newUrl);
                API_URL = newUrl;

                const statusDiv = document.getElementById('connection-status');
                if (statusDiv) {
                    statusDiv.innerHTML = '<span style="color: #10b981;">✓ Saved! Using: ' + newUrl + '</span>';
                }

                // Refresh model status with new URL
                initModelManager();
            }
        });
    }

    // Reset to Localhost
    const resetBackendBtn = document.getElementById('reset-backend-btn');
    if (resetBackendBtn) {
        resetBackendBtn.addEventListener('click', () => {
            const defaultUrl = 'http://localhost:5000';
            const input = document.getElementById('backend-url-input');
            if (input) input.value = defaultUrl;

            // Auto-save
            localStorage.setItem('backend-url', defaultUrl);
            API_URL = defaultUrl;

            const statusDiv = document.getElementById('connection-status');
            if (statusDiv) {
                statusDiv.innerHTML = '<span style="color: #10b981;">✓ Reset to Localhost</span>';
            }

            // Refresh model status
            initModelManager();
        });
    }

    // Test connection - check if backend is running
    const testConnectionBtn = document.getElementById('test-connection-btn');
    if (testConnectionBtn) {
        testConnectionBtn.addEventListener('click', async () => {
            const input = document.getElementById('backend-url-input');
            const testUrl = input ? input.value.trim() : API_URL;
            const statusDiv = document.getElementById('connection-status');

            if (statusDiv) {
                statusDiv.innerHTML = '<span style="color: #f59e0b;">⏳ Testing connection...</span>';
            }

            try {
                const response = await fetch(`${testUrl}/health`, {
                    mode: 'cors',
                    method: 'GET'
                });
                if (response.ok) {
                    statusDiv.innerHTML = '<span style="color: #22c55e;">✓ Connected successfully!</span>';
                } else {
                    statusDiv.innerHTML = '<span style="color: #ef4444;">✗ Backend returned error</span>';
                }
            } catch (error) {
                statusDiv.innerHTML = '<span style="color: #ef4444;">✗ Cannot connect: ' + error.message + '</span>';
            }
        });
    }

    // Save GPU info
    const saveGpuBtn = document.getElementById('save-gpu-btn');
    if (saveGpuBtn) {
        saveGpuBtn.addEventListener('click', () => {
            const gpuName = document.getElementById('gpu-name').value;
            const gpuVram = document.getElementById('gpu-vram').value;

            if (gpuName) localStorage.setItem('gpu-name', gpuName);
            if (gpuVram) localStorage.setItem('gpu-vram', gpuVram);

            updateGPUDisplay(gpuName, gpuVram);
            alert('GPU settings saved!');
        });
    }

    // ================================
    // MODEL STORAGE PATH
    // ================================

    // Get default storage path - C:\Users\username\.v6rge
    function getDefaultStoragePath() {
        const homeDir = window.v6rge?.homeDir || 'C:\\Users\\User';
        const separator = window.v6rge?.platform === 'win32' ? '\\' : '/';
        return `${homeDir}${separator}.v6rge`;
    }

    // Initialize storage path display
    function initStoragePath() {
        const modelDirInput = document.getElementById('model-dir-input');
        const storageStatus = document.getElementById('storage-status');

        if (modelDirInput) {
            // Load saved path or use default
            let storagePath = localStorage.getItem('model-storage-path');

            // Fix: If path is the generic fallback 'User' but we have real home dir, update it
            if (storagePath && storagePath.includes('Users\\User') && window.v6rge?.homeDir && !window.v6rge.homeDir.includes('Users\\User')) {
                // Determine if we should append models
                storagePath = getDefaultStoragePath();
                console.log('Correcting generic storage path to:', storagePath);
                localStorage.setItem('model-storage-path', storagePath);
            }

            if (!storagePath) {
                storagePath = getDefaultStoragePath();
                localStorage.setItem('model-storage-path', storagePath);
            }
            modelDirInput.value = storagePath;

            // Update backend with storage path
            updateBackendStoragePath(storagePath);
        }
    }

    // Update backend with new storage path
    async function updateBackendStoragePath(path) {
        const storageStatus = document.getElementById('storage-status');
        try {
            const response = await fetch(`${API_URL}/config/storage-path`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.path) {
                    // Update input and local storage with the actual path (with /models appended)
                    const modelDirInput = document.getElementById('model-dir-input');
                    if (modelDirInput) modelDirInput.value = data.path;
                    localStorage.setItem('model-storage-path', data.path);
                }

                if (storageStatus) {
                    storageStatus.textContent = '✓ Storage path configured';
                    storageStatus.classList.add('success');
                }

                // Refresh model status as path changed
                initModelManager();
            }
        } catch (error) {
            console.log('Backend storage path update skipped (backend may not support it yet)');
        }
    }

    // Folder selector button
    const selectModelDirBtn = document.getElementById('select-model-dir-btn');
    if (selectModelDirBtn) {
        // Use onclick to prevent duplicate event listeners if script re-runs
        selectModelDirBtn.onclick = async () => {
            const modelDirInput = document.getElementById('model-dir-input');
            const storageStatus = document.getElementById('storage-status');

            if (window.v6rge && window.v6rge.selectFolder) {
                // Prevent multiple clicks
                selectModelDirBtn.disabled = true;

                try {
                    const selectedPath = await window.v6rge.selectFolder();
                    if (selectedPath) {
                        modelDirInput.value = selectedPath;
                        localStorage.setItem('model-storage-path', selectedPath);

                        // Auto-save to backend
                        await updateBackendStoragePath(selectedPath);

                        if (storageStatus) {
                            storageStatus.textContent = '✓ Path saved';
                            storageStatus.classList.add('success');
                        }
                    }
                } finally {
                    selectModelDirBtn.disabled = false;
                }
            } else {
                // Fallback for non-Electron environment
                if (storageStatus) {
                    storageStatus.textContent = 'Folder selection only available in desktop app';
                }
            }
        };
    }

    // Initialize storage path
    initStoragePath();

    loadSettings();

    // ================================
    // TOOL INTERACTIONS
    // ================================

    // Emotion tag buttons for TTS
    const tagBtns = document.querySelectorAll('.tag-btn');
    tagBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const ttsText = document.getElementById('tts-text');
            if (ttsText) {
                const tag = btn.dataset.tag;
                const cursorPos = ttsText.selectionStart;
                const textBefore = ttsText.value.substring(0, cursorPos);
                const textAfter = ttsText.value.substring(cursorPos);
                ttsText.value = textBefore + tag + ' ' + textAfter;
                ttsText.focus();
                ttsText.setSelectionRange(cursorPos + tag.length + 1, cursorPos + tag.length + 1);
            }
        });
    });

    // File upload handlers
    setupFileUpload('upscale-drop-zone', 'upscale-file-input', 'upscale-btn');
    setupFileUpload('bg-drop-zone', 'bg-file-input', 'remove-bg-btn');
    setupFileUpload('vocal-drop-zone', 'vocal-file-input', 'separate-btn');
    setupFileUpload('3d-drop-zone', '3d-file-input', 'generate-3d-btn');

    function setupFileUpload(dropZoneId, fileInputId, buttonId) {
        const dropZone = document.getElementById(dropZoneId);
        const fileInput = document.getElementById(fileInputId);
        const button = document.getElementById(buttonId);

        if (!dropZone || !fileInput) return;

        // Click to browse
        dropZone.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const fileName = e.target.files[0].name;
                dropZone.querySelector('p').textContent = fileName;
                if (button) button.disabled = false;
            }
        });

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                dropZone.querySelector('p').textContent = files[0].name;
                if (button) button.disabled = false;
            }
        });
    }

    // ================================
    // TOOL BUTTON HANDLERS (BACKEND INTEGRATION)
    // ================================

    // IMAGE GENERATION
    document.getElementById('generate-image-btn')?.addEventListener('click', async function () {
        const prompt = document.getElementById('image-prompt').value;
        const width = document.getElementById('image-width').value;
        const height = document.getElementById('image-height').value;
        const output = document.getElementById('image-output');
        const modelSelect = document.getElementById('image-model-select');

        if (!modelSelect || !modelSelect.value) {
            alert('Please select a model first');
            return;
        }

        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        this.disabled = true;
        this.textContent = 'Generating...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Generating image...</div>';

        try {
            const formData = new FormData();
            formData.append('prompt', prompt);
            formData.append('width', width);
            formData.append('height', height);

            // Advanced Params
            formData.append('steps', document.getElementById('image-steps').value);
            formData.append('guidance', document.getElementById('image-guidance').value);
            formData.append('seed', document.getElementById('image-seed').value);
            formData.append('negative_prompt', document.getElementById('image-negative').value);

            const modelSelect = document.getElementById('image-model-select');
            if (modelSelect && modelSelect.value) {
                formData.append('model', modelSelect.value);
            }

            const response = await fetch(`${API_URL}/generate_image`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                const mediaUrl = data.image_url.startsWith('http') ? data.image_url : `${API_URL}${data.image_url.startsWith('/') ? '' : '/'}${data.image_url}`;
                output.innerHTML = `<img src="${mediaUrl}" style="max-width: 100%; border-radius: 8px;" onerror="this.src=''; this.alt='Image failed to load';">`;
            } else {
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Generate Image';
        }
    });

    // MUSIC GENERATION
    document.getElementById('generate-music-btn')?.addEventListener('click', async function () {
        const prompt = document.getElementById('music-prompt').value;
        const duration = document.getElementById('music-duration').value;
        const output = document.getElementById('music-output');
        const modelSelect = document.getElementById('music-model-select');

        if (!modelSelect || !modelSelect.value) {
            alert('Please select a model first');
            return;
        }

        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        this.disabled = true;
        this.textContent = 'Generating...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Generating music...</div>';

        try {
            const formData = new FormData();
            formData.append('prompt', prompt);
            formData.append('duration', duration);

            // Advanced Params
            formData.append('guidance', document.getElementById('music-guidance').value);
            formData.append('temperature', document.getElementById('music-temp').value);

            const modelSelect = document.getElementById('music-model-select');
            if (modelSelect && modelSelect.value) {
                formData.append('model', modelSelect.value);
            }

            const response = await fetch(`${API_URL}/generate_music`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                const mediaUrl = data.audio_url.startsWith('http') ? data.audio_url : `${API_URL}${data.audio_url.startsWith('/') ? '' : '/'}${data.audio_url}`;
                output.innerHTML = `
                <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; width: 100%;">
                    <div style="margin-bottom: 8px; font-weight: 500;">Generated Track</div>
                    <audio controls src="${mediaUrl}" style="width: 100%;"></audio>
                </div>`;
            } else {
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Generate Music';
        }
    });

    // TEXT-TO-SPEECH
    document.getElementById('synthesize-btn')?.addEventListener('click', async function () {
        const text = document.getElementById('tts-text').value;
        const output = document.getElementById('tts-output');
        const modelSelect = document.getElementById('tts-model-select');

        if (!modelSelect || !modelSelect.value) {
            alert('Please select a model first');
            return;
        }

        if (!text) {
            alert('Please enter text');
            return;
        }

        this.disabled = true;
        this.textContent = 'Synthesizing...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Generating speech...</div>';

        try {
            const formData = new FormData();
            formData.append('text', text);

            const modelSelect = document.getElementById('tts-model-select');
            if (modelSelect && modelSelect.value) {
                formData.append('model', modelSelect.value);
            }

            const response = await fetch(`${API_URL}/synthesize`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                output.innerHTML = `<audio controls src="${url}" style="width: 100%;"></audio>`;
            } else {
                const data = await response.json();
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Synthesize Speech';
        }
    });

    // VIDEO GENERATION
    document.getElementById('generate-video-btn')?.addEventListener('click', async function () {
        const prompt = document.getElementById('video-prompt').value;
        const duration = document.getElementById('video-duration').value; // This might be unused in backend based on previous context but kept for consistency
        const output = document.getElementById('video-output');
        const modelSelect = document.getElementById('video-model-select');

        if (!modelSelect || !modelSelect.value) {
            alert('Please select a model first');
            return;
        }

        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        const frames = document.getElementById('video-frames').value;
        const steps = document.getElementById('video-steps').value;
        const fps = document.getElementById('video-fps').value;

        this.disabled = true;
        this.textContent = 'Generating Video...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Generating video (this will take a while)...</div>';

        try {
            const formData = new FormData();
            formData.append('prompt', prompt);
            if (duration) formData.append('duration', duration); // Conditionally append if it exists/is supported

            // Advanced Params
            formData.append('frames', frames);
            formData.append('steps', steps);
            formData.append('fps', fps);

            const modelSelect = document.getElementById('video-model-select');
            if (modelSelect && modelSelect.value) {
                formData.append('model', modelSelect.value);
            }

            const response = await fetch(`${API_URL}/generate_video`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                const mediaUrl = data.video_url.startsWith('http') ? data.video_url : `${API_URL}${data.video_url.startsWith('/') ? '' : '/'}${data.video_url}`;
                output.innerHTML = `<video controls src="${mediaUrl}" style="max-width: 100%; border-radius: 8px;"></video>`;
            } else {
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Generate Video';
        }
    });

    // IMAGE UPSCALER
    document.getElementById('upscale-btn')?.addEventListener('click', async function () {
        const fileInput = document.getElementById('upscale-file-input');
        const scale = document.getElementById('upscale-scale').value;
        const output = document.getElementById('upscale-output');

        if (!fileInput.files[0]) {
            alert('Please select an image');
            return;
        }

        this.disabled = true;
        this.textContent = 'Upscaling...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Upscaling image...</div>';

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('scale', scale);

            const response = await fetch(`${API_URL}/upscale`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                const imageUrl = `${API_URL}${data.upscaled_url}`;
                output.innerHTML = `
                    <div class="result-card">
                        <div class="preview-container" style="height: 250px; background: rgba(0,0,0,0.3); border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; overflow: hidden; border: 1px solid var(--border-subtle);">
                            <img src="${imageUrl}" style="max-height: 100%; max-width: 100%; object-fit: contain; cursor: pointer;" id="upscale-preview-img">
                        </div>
                        <div class="actions-row" style="display: flex; gap: 8px;">
                            <button class="btn-secondary" id="upscale-fullscreen-btn" style="flex: 1;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px;"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
                                Full Screen
                            </button>
                            <a href="${imageUrl}" download="upscaled_image.png" class="btn-primary" style="flex: 1; display: flex; align-items: center; justify-content: center; text-decoration: none;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                                Save Image
                            </a>
                        </div>
                    </div>
                `;

                // Add fullscreen handler
                const fsBtn = document.getElementById('upscale-fullscreen-btn');
                const img = document.getElementById('upscale-preview-img');

                const openFs = () => {
                    const modal = document.createElement('div');
                    modal.style.cssText = 'position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.9); z-index: 9999; display: flex; align-items: center; justify-content: center; cursor: zoom-out;';
                    modal.innerHTML = `<img src="${imageUrl}" style="max-width: 95%; max-height: 95%; object-fit: contain;">`;
                    modal.onclick = () => modal.remove();
                    document.body.appendChild(modal);
                };

                fsBtn.onclick = openFs;
                img.onclick = openFs;

            } else {
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Upscale Image';
        }
    });

    // BACKGROUND REMOVER
    document.getElementById('remove-bg-btn')?.addEventListener('click', async function () {
        const fileInput = document.getElementById('bg-file-input');
        const output = document.getElementById('bg-output');

        if (!fileInput.files[0]) {
            alert('Please select an image');
            return;
        }

        this.disabled = true;
        this.textContent = 'Removing...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Removing background...</div>';

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const response = await fetch(`${API_URL}/remove_background`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                output.innerHTML = `<img src="${API_URL}${data.image_url}" style="max-width: 100%; border-radius: 8px;">`;
            } else {
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Remove Background';
        }
    });

    // VOCAL SEPARATOR
    document.getElementById('separate-btn')?.addEventListener('click', async function () {
        const fileInput = document.getElementById('vocal-file-input');
        const output = document.getElementById('vocal-output');

        if (!fileInput.files[0]) {
            alert('Please select an audio file');
            return;
        }

        this.disabled = true;
        this.textContent = 'Separating...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Separating vocals...</div>';

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const response = await fetch(`${API_URL}/process`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                output.innerHTML = `
                    <div style="display: flex; flex-direction: column; gap: 12px;">
                        <div>
                            <strong>Vocals:</strong>
                            <audio controls src="${API_URL}${data.vocals_url}" style="width: 100%; margin-top: 4px;"></audio>
                        </div>
                        <div>
                            <strong>Instrumental:</strong>
                            <audio controls src="${API_URL}${data.instrumental_url}" style="width: 100%; margin-top: 4px;"></audio>
                        </div>
                    </div>
                `;
            } else {
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Separate Vocals';
        }
    });


    // 3D GENERATION
    document.getElementById('generate-3d-btn')?.addEventListener('click', async function () {
        const fileInput = document.getElementById('3d-file-input');
        const formatSelect = document.getElementById('3d-format');
        const output = document.getElementById('3d-output');
        const modelSelect = document.getElementById('3d-model-select');

        // Model Validation
        if (!modelSelect || !modelSelect.value) {
            alert('Please select a model first');
            return;
        }

        // File Validation
        if (!fileInput.files[0]) {
            alert('Please select an image file');
            return;
        }

        this.disabled = true;
        this.textContent = 'Generating 3D Model...';
        output.classList.add('active');
        output.innerHTML = '<div class="loading"><div class="spinner"></div> Generating 3D model (this may take a while)...</div>';

        try {
            const formData = new FormData();
            formData.append('images', fileInput.files[0]);
            formData.append('format', formatSelect.value);
            formData.append('model', modelSelect.value);

            // Advanced Params
            formData.append('steps', document.getElementById('3d-steps').value);
            formData.append('texture_res', document.getElementById('3d-texture-res').value);
            formData.append('seed', document.getElementById('3d-seed').value);

            const response = await fetch(`${API_URL}/generate-3d`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                // data.model_url e.g. /download-3d/filename.glb
                const downloadUrl = data.model_url.startsWith('http') ? data.model_url : `${API_URL}${data.model_url.startsWith('/') ? '' : '/'}${data.model_url}`;

                output.innerHTML = `
                    <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 10px;">📦</div>
                        <h3>Model Generated!</h3>
                        <p>Format: ${data.format.toUpperCase()}</p>
                        <a href="${downloadUrl}" class="btn-secondary" download style="display: inline-block; margin-top: 10px; text-decoration: none;">Download 3D Model</a>
                    </div>`;
            } else {
                output.innerHTML = `<div style="color: #ef4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            output.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
        } finally {
            this.disabled = false;
            this.textContent = 'Generate 3D Model';
        }
    });

    // 3D Drag and Drop
    const dropZone3d = document.getElementById('3d-drop-zone');
    const fileInput3d = document.getElementById('3d-file-input');

    if (dropZone3d && fileInput3d) {
        dropZone3d.addEventListener('click', () => fileInput3d.click());

        dropZone3d.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone3d.style.borderColor = 'var(--accent-primary)';
        });

        dropZone3d.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone3d.style.borderColor = 'var(--border-color)';
        });

        dropZone3d.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone3d.style.borderColor = 'var(--border-color)';
            if (e.dataTransfer.files.length) {
                fileInput3d.files = e.dataTransfer.files;
                dropZone3d.innerHTML = `<div class="upload-icon">✅</div><p>${e.dataTransfer.files[0].name}</p>`;
            }
        });

        fileInput3d.addEventListener('change', () => {
            if (fileInput3d.files.length) {
                dropZone3d.innerHTML = `<div class="upload-icon">✅</div><p>${fileInput3d.files[0].name}</p>`;
            }
        });
    }


    // ================================
    // QUICK ACTION BUTTONS
    // ================================
    const quickActionBtns = document.querySelectorAll('.capsule-btn[data-action]');

    quickActionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            let promptText = '';

            // Update active state
            quickActionBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            switch (action) {
                case 'generate-image':
                    promptText = 'Generate an image of ';
                    break;
                case 'generate-music':
                    promptText = 'Create music for ';
                    break;
                case 'text-to-speech':
                    promptText = 'Read this aloud: ';
                    break;
                case 'summarize':
                    promptText = 'Summarize this: ';
                    break;
            }

            if (chatInput) {
                chatInput.value = promptText;
                chatInput.focus();
                // Set cursor to end
                if (chatInput.setSelectionRange) {
                    chatInput.setSelectionRange(promptText.length, promptText.length);
                }
            }
        });
    });


    // ================================
    // SETTINGS & CONFIGURATION
    // ================================

    // Load initial config
    async function loadConfig() {
        console.log('Loading configuration...');
        try {
            const response = await fetch(`${API_URL}/config`);
            if (response.ok) {
                const config = await response.json();
                const dirInput = document.getElementById('model-dir-input');
                if (dirInput && config.models_dir) {
                    dirInput.value = config.models_dir;
                }
            }
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }
    loadConfig();

    // Select Folder Handler
    const selectFolderBtn = document.getElementById('select-model-dir-btn');
    if (selectFolderBtn) {
        selectFolderBtn.addEventListener('click', async () => {
            if (window.v6rge && window.v6rge.selectFolder) {
                const path = await window.v6rge.selectFolder();
                if (path) {
                    document.getElementById('model-dir-input').value = path;
                }
            } else {
                alert('Folder selection is not supported in this environment (requires Electron).');
            }
        });
    }

    // Save Config Handler
    const saveConfigBtn = document.getElementById('save-model-dir-btn');
    if (saveConfigBtn) {
        saveConfigBtn.addEventListener('click', async function () {
            const newPath = document.getElementById('model-dir-input').value;
            if (!newPath) return;

            this.disabled = true;
            this.textContent = 'Saving...';

            try {
                const response = await fetch(`${API_URL}/config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ models_dir: newPath })
                });

                if (response.ok) {
                    alert('Model directory updated! Please restart the app for full effect.');
                } else {
                    alert('Failed to save configuration');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                this.disabled = false;
                this.textContent = 'Update Directory';
            }
        });
    }

    console.log('V6rge Desktop UI Ready - Connected to localhost:5000');

});

// Add message styles dynamically
const messageStyles = document.createElement('style');
messageStyles.textContent = `
    .message {
        padding: 8px 0;
        max-width: 100%;
        width: 100%;
    }
    
    .message.user {
        display: flex;
        justify-content: flex-end;
    }
    
    .message.user .message-content {
        background: var(--bg-tertiary);
        padding: 10px 14px;
        border-radius: 12px;
        border-bottom-right-radius: 4px;
        max-width: 70%;
        word-wrap: break-word;
    }
    
    .message.assistant .message-content {
        color: var(--text-secondary);
        padding: 12px 16px;
        background: white;
        border-radius: 12px;
        max-width: 100%;
        width: 100%;
    }
    
    .message.loading .message-content {
        opacity: 0.6;
        font-style: italic;
    }
`;
document.head.appendChild(messageStyles);
