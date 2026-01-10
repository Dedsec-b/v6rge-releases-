import { api } from './api.js';

export class ChatView {
    constructor() {
        this.messages = [];
        this.setupUI();
    }

    setupUI() {
        this.container = document.getElementById('chat-messages');
        this.input = document.querySelector('.chat-input');
        this.sendBtn = document.querySelector('.send-btn');
        this.attachBtn = document.querySelector('.attach-btn');
        this.modelSelect = document.getElementById('chat-model-selector');
        this.currentFile = null;

        // Hidden File Input
        this.fileInput = document.createElement('input');
        this.fileInput.type = 'file';
        this.fileInput.style.display = 'none';
        document.body.appendChild(this.fileInput); // Append to body so it works

        // --- Event Listeners ---

        // Send Button
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // Enter Key
        if (this.input) {
            this.input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // Attach Button
        if (this.attachBtn) {
            this.attachBtn.addEventListener('click', () => {
                this.fileInput.click();
            });
        }

        // File Input Change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Presets (Capsule Buttons)
        document.querySelectorAll('.capsule-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.getAttribute('data-action');
                this.handlePreset(action);
            });
        });
    }

    handlePreset(action) {
        if (!this.input) return;

        switch (action) {
            case 'generate-image':
                this.input.value = "Generate an image of ";
                break;
            case 'generate-music':
                this.input.value = "Generate music about ";
                break;
            case 'text-to-speech':
                this.input.value = "Read this aloud: ";
                break;
            case 'summarize':
                if (this.currentFile) {
                    this.input.value = "Summarize the attached file.";
                } else {
                    this.input.value = "Summarize this: ";
                }
                break;
        }
        this.input.focus();
    }

    handleFileSelect(file) {
        this.currentFile = file;

        let preview = document.querySelector('.file-preview-pill');
        if (!preview) {
            preview = document.createElement('div');
            preview.className = 'file-preview-pill';
            // Gemini-style: Just the image/file container, styled cleanly
            preview.style.cssText = `
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: var(--bg-secondary);
                padding: 6px;
                border-radius: 8px;
                margin: 0 12px 8px 12px;
                width: fit-content;
                border: 1px solid var(--border-subtle);
                position: relative;
            `;
            const wrapper = document.querySelector('.chat-input-wrapper');
            if (wrapper) wrapper.insertBefore(preview, wrapper.firstChild);
        }

        const isImage = file.type.startsWith('image/');

        if (isImage) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.innerHTML = `
                    <div style="position: relative; display: flex;">
                        <img src="${e.target.result}" style="width: 48px; height: 48px; object-fit: cover; border-radius: 6px;">
                        <span style="font-size: 0.8rem; margin-left: 8px; align-self: center; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${file.name}</span>
                        <button class="remove-file-btn" style="
                            position: absolute; 
                            top: -6px; 
                            right: -6px; 
                            background: var(--bg-tertiary); 
                            border: 1px solid var(--border-subtle); 
                            border-radius: 50%; 
                            width: 18px; 
                            height: 18px; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            cursor: pointer; 
                            color: var(--text-primary);
                            padding: 0;
                        ">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                `;
                this._attachRemoveListener(preview);
            };
            reader.readAsDataURL(file);
        } else {
            // Document/Other file style
            preview.innerHTML = `
                <div style="display: flex; align-items: center; padding: 0 4px;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: var(--text-secondary);">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <span style="margin-left: 8px; font-size: 0.85rem; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${file.name}</span>
                    <button class="remove-file-btn" style="background:none; border:none; cursor:pointer; color:var(--text-secondary); padding:0; margin-left:8px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
            `;
            this._attachRemoveListener(preview);
        }
    }

    _attachRemoveListener(preview) {
        const btn = preview.querySelector('.remove-file-btn');
        if (btn) {
            btn.addEventListener('click', () => {
                this.currentFile = null;
                this.fileInput.value = '';
                preview.remove();
            });
        }
    }

    async sendMessage() {
        const text = this.input.value.trim();
        if (!text && !this.currentFile) return; // Allow sending if just file? Usually need text too or backend fails. Let's assume text required for now or just allow it.

        // Construct display text with file info
        let displayText = text;
        if (this.currentFile) {
            displayText += `\n\n[Attached: ${this.currentFile.name}]`;
        }
        if (!displayText) displayText = "...";

        // Add User Message
        this.addMessage('user', displayText);
        this.input.value = '';
        this.messages.push({ role: 'user', content: displayText });

        // Add Loading - store the element directly
        const loadingDiv = this.addMessageElement('assistant', 'Thinking');
        loadingDiv.classList.add('loading');

        try {
            const modelId = this.modelSelect ? this.modelSelect.value : 'qwen-3b';

            // Prepare Payload (FormData if file, else JSON)
            let res;
            if (this.currentFile) {
                const formData = new FormData();
                formData.append('message', text); // Send clean text to backend
                formData.append('model_id', modelId);
                formData.append('history', JSON.stringify(this.messages));
                formData.append('file', this.currentFile);
                
                res = await api.post('/chat', formData); // Header auto-handled by browser
            } else {
                res = await api.post('/chat', {
                    message: text,
                    model_id: modelId,
                    history: JSON.stringify(this.messages)
                });
            }

            // Clear File
            this.currentFile = null;
            this.fileInput.value = '';
            const preview = document.querySelector('.file-preview-pill');
            if (preview) preview.remove();

            // Remove Loading
            loadingDiv.remove();

            // Add response
            await this.typewriterMessage(res.response);
            this.messages.push({ role: 'assistant', content: res.response });

            // Handle tool results
            if (res.tool_result) {
                if (res.tool_result.type === 'approval_required') {
                    this.showApprovalModal(res.tool_result);
                } else {
                    this.handleToolResult(res.tool_result);
                }
            }

        } catch (e) {
            loadingDiv.classList.remove('loading');
            const contentDiv = loadingDiv.querySelector('.message-content');
            if (contentDiv) contentDiv.textContent = `Error: ${e.message}`;
        }
    }

    showApprovalModal(request) {
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'approval-modal';
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8); z-index: 1000;
            display: flex; align-items: center; justify-content: center;
        `;
        
        const card = document.createElement('div');
        card.style.cssText = `
            background: #1a1a1a;
            border: 2px solid #ff0055;
            box-shadow: 0 0 20px rgba(255, 0, 85, 0.3);
            border-radius: 12px;
            padding: 24px;
            width: 400px; max-width: 90%;
            text-align: center; font-family: 'Courier New', monospace;
        `;
        
        card.innerHTML = `
            <h3 style="color: #ff0055; margin-bottom: 16px;">⚠️ SYSTEM ACCESS REQUEST</h3>
            <p style="color: #ddd; margin-bottom: 8px;">V6rge wants to execute:</p>
            <div style="background: #000; padding: 12px; border-radius: 6px; color: #0f0; margin-bottom: 20px; text-align: left; word-break: break-all;">
                > ${request.command}
            </div>
            <div style="display: flex; gap: 12px; justify-content: center;">
                <button id="deny-btn" style="background: transparent; border: 1px solid #666; color: #aaa; padding: 8px 16px; border-radius: 4px; cursor: pointer;">DENY</button>
                <button id="allow-btn" style="background: #ff0055; border: none; color: white; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold;">ALLOW ACCESS</button>
            </div>
        `;
        
        modal.appendChild(card);
        document.body.appendChild(modal);
        
        // Handlers
        modal.querySelector('#deny-btn').addEventListener('click', () => {
             modal.remove();
             this.addMessage('system', '❌ Access Denied by User.');
        });
        
        modal.querySelector('#allow-btn').addEventListener('click', async () => {
             modal.remove();
             await this.executeApprovedTool(request);
        });
    }

    async executeApprovedTool(request) {
        // Show executing status
        const loadingDiv = this.addMessageElement('system', `> Executing: ${request.command}...`);
        
        try {
            // Fix 415: Send JSON header explicitly
            const res = await api.post('/tool/execute', {
                tool: request.tool,
                command: request.command
            }, { 'Content-Type': 'application/json' });
            
            loadingDiv.remove();
            
            if (res.status === 'success') {
                // UX Improvement: Don't just show raw output. 
                // Send the output back to the BOT so it can explain it to the user.
                
                // Show a "Thinking..." indicator
                const thinkingDiv = this.addMessageElement('assistant', 'Analyzing system response...');
                thinkingDiv.classList.add('loading');

                try {
                    // Feed result back to context as SYSTEM role
                    this.messages.push({ role: 'system', content: `Command Output for '${request.command}':\n${res.output}` });
                    
                    const followUpRes = await api.post('/chat', {
                        message: "(System: The command finished. Please explain the result to the user.)", 
                        model_id: this.modelSelect ? this.modelSelect.value : 'qwen-3b',
                        history: JSON.stringify(this.messages)
                    });

                    thinkingDiv.remove();
                    
                    // Add the natural response
                    await this.typewriterMessage(followUpRes.response);
                    this.messages.push({ role: 'assistant', content: followUpRes.response });

                } catch (err) {
                    thinkingDiv.remove();
                    this.addMessage('system', `Error generating follow-up: ${err.message}`);
                }

            } else {
                this.addMessage('system', `Error: ${res.error}`);
            }
            
        } catch (e) {
            loadingDiv.remove();
            this.addMessage('system', `Execution Failed: ${e.message}`);
        }
    }

    handleToolResult(result) {
        const div = document.createElement('div');
        div.className = 'message assistant';

        // Construct the structured content based on ported styles
        let contentHtml = '';
        // Fix relative URLs by prepending backend URL
        let mediaUrl = result.url;
        if (mediaUrl && !mediaUrl.startsWith('http') && !mediaUrl.startsWith('blob')) {
            mediaUrl = `${api.baseUrl}${mediaUrl.startsWith('/') ? mediaUrl : '/' + mediaUrl}`;
        }

        if (result.type === 'image') {
            contentHtml = `
                <div class="generated-content">
                    <div class="content-label">Generated Image</div>
                    <img src="${mediaUrl}" alt="Generated image">
                </div>
            `;
        } else if (result.type === 'audio') {
            contentHtml = `
                <div class="generated-content">
                    <div class="content-label">Generated Audio</div>
                    <audio controls src="${mediaUrl}"></audio>
                </div>
            `;
        } else {
            // Fallback
            contentHtml = `<div class="generated-content">${JSON.stringify(result)}</div>`;
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = contentHtml;

        // Fix images/audio if necessary (using our centralized downloader with ngrok headers)
        if (result.type === 'image') {
            this.fixImages(contentDiv);
        } else if (result.type === 'audio') {
            this.fixAudio(contentDiv);
        }

        div.appendChild(contentDiv);
        this.container.appendChild(div);
        this.scrollToBottom();
    }

    addMessage(role, content) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.id = `msg-${Date.now()}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (role === 'assistant') {
            contentDiv.innerHTML = this.parseMarkdown(content);
            this.fixImages(contentDiv);
        } else {
            contentDiv.textContent = content;
        }

        div.appendChild(contentDiv);
        this.container.appendChild(div);
        this.scrollToBottom();
        return div.id;
    }

    addMessageElement(role, content) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        div.appendChild(contentDiv);
        this.container.appendChild(div);
        this.scrollToBottom();
        return div;
    }

    updateMessage(id, content) {
        const div = document.getElementById(id);
        if (div) {
            const contentDiv = div.querySelector('.message-content');
            contentDiv.innerHTML = this.parseMarkdown(content);
            this.fixImages(contentDiv); // Process images
            this.scrollToBottom();
        }
    }

    // New: Fix images by routing them through api.download
    async fixImages(container) {
        const images = container.querySelectorAll('img');
        for (const img of images) {
            const src = img.getAttribute('src');
            // Check if it looks like a backend path or relative path
            if (src && !src.startsWith('blob:') && !src.startsWith('data:')) {
                // If it's a relative path starting with /, assume it needs API handling
                // Or if it matches our backend URL
                try {
                    const blob = await api.download(src);
                    const blobUrl = URL.createObjectURL(blob);
                    img.src = blobUrl;
                } catch (e) {
                    console.warn("Failed to load chat image via API:", src, e);
                    // Add error styling
                    img.style.border = '2px solid var(--text-error)';
                    img.title = "Failed to load image";
                }
            }
        }
    }

    // New: Fix audio by routing through api.download (for ngrok headers)
    async fixAudio(container) {
        const audios = container.querySelectorAll('audio');
        for (const audio of audios) {
            const src = audio.getAttribute('src');
            // Check if it's a URL that needs our download handler
            if (src && !src.startsWith('blob:') && !src.startsWith('data:')) {
                try {
                    const blob = await api.download(src);
                    const blobUrl = URL.createObjectURL(blob);
                    audio.src = blobUrl;
                } catch (e) {
                    console.warn("Failed to load chat audio via API:", src, e);
                    audio.parentElement.innerHTML = '<p style="color: var(--text-error);">Failed to load audio</p>';
                }
            }
        }
    }

    parseMarkdown(text) {
        // Simple shim if marked isn't available
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        }
        return text.replace(/\n/g, '<br>');
    }

    scrollToBottom() {
        this.container.scrollTop = this.container.scrollHeight;
    }

    async typewriterMessage(content) {
        // Create container logic similar to addMessage but for progressive rendering
        const div = document.createElement('div');
        div.className = `message assistant`;
        div.id = `msg-${Date.now()}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        div.appendChild(contentDiv);
        this.container.appendChild(div);

        // Typewriter animation
        const speed = 5; // fast 
        const chunkSize = 3;

        // Use a temp div to parse markdown progressively or just text if needed
        // For simplicity and performance, we'll just stream text then parse markdown at end?
        // Legacy code did: parse markdown as we go.

        let displayedText = '';
        for (let i = 0; i < content.length; i += chunkSize) {
            displayedText = content.slice(0, i + chunkSize);
            contentDiv.innerHTML = this.parseMarkdown(displayedText);
            this.scrollToBottom();
            await new Promise(r => setTimeout(r, speed));
        }

        // Final pass
        contentDiv.innerHTML = this.parseMarkdown(content);
        this.fixImages(contentDiv);
        this.scrollToBottom();
    }
}
