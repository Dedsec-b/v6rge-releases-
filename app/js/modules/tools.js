import { api } from './api.js';

export class ToolManager {
    constructor() {
        this.initImageGen();
        this.initMusicGen();
        this.initVideoGen();
        this.initTTS();
        this.initUpscaler();
        this.init3DGen();
        this.initBGRemover();
        this.initVocalSeparator();
        this.initFeedback();

        // Universal Model Loader
        this.loadAllModels();
    }

    async loadAllModels() {
        try {
            const status = await api.get('/models/status');
            const selectors = document.querySelectorAll('select.model-selector');

            selectors.forEach(select => {
                const category = select.dataset.category;
                const specificId = select.dataset.specific; // For exact model matches e.g. 'realesrgan'

                // Reset
                select.innerHTML = '<option value="" disabled selected>Select Model...</option>';

                // Use Object.entries to get [id, modelData] pairs - ID comes from the key!
                const availableModels = Object.entries(status.models).filter(([id, m]) => {
                    // Must be downloaded
                    if (!m.downloaded) return false;
                    // Must match category (type)
                    if (m.type !== category) return false;
                    // If specific ID required (for tools like upscaler/rembg)
                    if (specificId && id !== specificId) return false;
                    return true;
                });

                if (availableModels.length === 0) {
                    select.innerHTML = '<option value="" disabled selected>No Models Found (Download in Dashboard)</option>';
                } else {
                    availableModels.forEach(([id, m]) => {
                        const opt = document.createElement('option');
                        opt.value = id;
                        opt.textContent = m.name || id;
                        select.appendChild(opt);
                    });
                    // Auto-select first available model
                    if (select.options.length > 1) {
                        select.selectedIndex = 1;
                    }
                }

                // Apply custom styling
                this.setupCustomSelect(select);
            });

        } catch (e) {
            console.error('Failed to load tool models:', e);
        }
    }

    setupCustomSelect(select) {
        // Remove existing wrapper if present
        const existingWrapper = select.nextElementSibling;
        if (existingWrapper && existingWrapper.classList.contains('custom-select-wrapper')) {
            existingWrapper.remove();
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'custom-select-wrapper';

        // Create trigger (the visible button)
        const trigger = document.createElement('div');
        trigger.className = 'custom-select-trigger';
        const selected = select.options[select.selectedIndex];
        trigger.innerHTML = `<span>${selected ? selected.text : 'Select'}</span><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>`;

        // Create options panel
        const optionsPanel = document.createElement('div');
        optionsPanel.className = 'custom-options';

        Array.from(select.options).forEach((opt, i) => {
            const div = document.createElement('div');
            div.className = 'custom-option' + (opt.disabled ? ' disabled' : '') + (opt.selected ? ' selected' : '');
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

        // Toggle dropdown on click
        trigger.addEventListener('click', (e) => {
            document.querySelectorAll('.custom-select-wrapper.open').forEach(w => {
                if (w !== wrapper) w.classList.remove('open');
            });
            wrapper.classList.toggle('open');
            e.stopPropagation();
        });

        wrapper.appendChild(trigger);
        wrapper.appendChild(optionsPanel);

        // Hide original select, insert custom UI
        select.style.display = 'none';
        select.parentNode.insertBefore(wrapper, select.nextSibling);

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            wrapper.classList.remove('open');
        });
    }

    initImageGen() {
        const btn = document.getElementById('generate-image-btn');
        const modelSelect = document.getElementById('image-model-select');
        const qwenAdvanced = document.getElementById('qwen-advanced-options');

        if (!btn) return;

        // Model selector change handler - toggle Qwen-specific advanced options
        if (modelSelect) {
            modelSelect.addEventListener('change', () => {
                const isQwen = modelSelect.value === 'qwen-image';
                if (qwenAdvanced) qwenAdvanced.style.display = isQwen ? 'block' : 'none';

                // Update steps default based on model
                const stepsSlider = document.getElementById('image-steps');
                if (stepsSlider) {
                    stepsSlider.value = isQwen ? '50' : '4';
                    stepsSlider.max = isQwen ? '50' : '10';
                    const stepsValue = document.getElementById('image-steps-value');
                    if (stepsValue) stepsValue.textContent = stepsSlider.value;
                }
            });
        }

        // Slider value displays
        const imageSteps = document.getElementById('image-steps');
        const imageStepsValue = document.getElementById('image-steps-value');
        if (imageSteps && imageStepsValue) {
            imageSteps.addEventListener('input', () => imageStepsValue.textContent = imageSteps.value);
        }

        const qwenGuidance = document.getElementById('qwen-guidance');
        const qwenGuidanceValue = document.getElementById('qwen-guidance-value');
        if (qwenGuidance && qwenGuidanceValue) {
            qwenGuidance.addEventListener('input', () => qwenGuidanceValue.textContent = qwenGuidance.value);
        }

        // Resolution preset buttons
        const presetButtons = document.querySelectorAll('#resolution-presets .preset-btn');
        presetButtons.forEach(pbtn => {
            pbtn.addEventListener('click', (e) => {
                e.preventDefault();
                presetButtons.forEach(b => b.classList.remove('active'));
                pbtn.classList.add('active');
            });
        });

        // Generate button handler
        btn.addEventListener('click', async () => {
            const prompt = document.getElementById('image-prompt').value;
            const model = modelSelect ? modelSelect.value : 'flux-schnell';

            if (!prompt) return window.toast?.warning('Please enter a prompt');
            if (!window.isCloudMode && model === '') return window.toast?.warning('Please select a model');

            btn.disabled = true;
            btn.textContent = 'Generating...';

            try {
                // Get resolution from active preset button
                const activePreset = document.querySelector('#resolution-presets .preset-btn.active');
                const width = activePreset?.dataset.width || '1024';
                const height = activePreset?.dataset.height || '1024';

                let params = {
                    prompt,
                    model,
                    width,
                    height,
                    steps: document.getElementById('image-steps')?.value || '4',
                    seed: document.getElementById('image-seed')?.value || '0'
                };

                // Add Qwen-specific params if Qwen model
                if (model === 'qwen-image') {
                    params.guidance = document.getElementById('qwen-guidance')?.value || '4';
                    params.negative_prompt = document.getElementById('qwen-negative-prompt')?.value || '';
                    params.prompt_enhance = document.getElementById('qwen-enhance')?.checked ? 'true' : 'false';
                }

                const res = await api.post('/generate_image', params);
                this.showResult('image-output', 'image', res.image_url);
            } catch (e) {
                window.toast?.error(e.message || 'Image generation failed');
            }
            btn.disabled = false;
            btn.textContent = 'Generate';
        });
    }

    initMusicGen() {
        const btn = document.getElementById('generate-music-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const prompt = document.getElementById('music-prompt').value;
            const duration = document.getElementById('music-duration').value;
            const model = document.getElementById('music-model-select').value;

            if (!prompt) return window.toast?.warning('Please enter a prompt');
            if (!window.isCloudMode && model === '') return window.toast?.warning('Please select a model');

            btn.disabled = true;
            btn.textContent = 'Composing...';

            try {
                const res = await api.post('/generate_music', { prompt, duration, model });
                this.showResult('music-output', 'audio', res.audio_url);
            } catch (e) {
                window.toast?.error(e.message || 'Music generation failed');
            }
            btn.disabled = false;
            btn.textContent = 'Generate';
        });
    }

    initVideoGen() {
        const btn = document.getElementById('generate-video-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const prompt = document.getElementById('video-prompt').value;
            const model = document.getElementById('video-model-select').value;

            if (!prompt) return window.toast?.warning('Please enter a prompt');
            if (!window.isCloudMode && model === '') return window.toast?.warning('Please select a model');

            btn.disabled = true;
            btn.textContent = 'Cranking...';

            try {
                const res = await api.post('/generate_video', { prompt, model });
                this.showResult('video-output', 'video', res.video_url);
            } catch (e) {
                window.toast?.error(e.message || 'Video generation failed');
            }
            btn.disabled = false;
            btn.textContent = 'Generate';
        });
    }

    initTTS() {
        const btn = document.getElementById('synthesize-btn');
        if (!btn) return;

        // Emotion Tags Handler
        const tagButtons = document.querySelectorAll('.tag-btn');
        const textArea = document.getElementById('tts-text');

        tagButtons.forEach(tagBtn => {
            tagBtn.addEventListener('click', () => {
                if (!textArea) return;
                const tag = tagBtn.dataset.tag;

                // Insert at cursor position or append
                if (textArea.selectionStart || textArea.selectionStart == '0') {
                    const startPos = textArea.selectionStart;
                    const endPos = textArea.selectionEnd;
                    textArea.value = textArea.value.substring(0, startPos)
                        + " " + tag + " "
                        + textArea.value.substring(endPos, textArea.value.length);
                    textArea.focus();
                    textArea.selectionStart = startPos + tag.length + 2;
                    textArea.selectionEnd = startPos + tag.length + 2;
                } else {
                    textArea.value += " " + tag + " ";
                    textArea.focus();
                }
            });
        });

        btn.addEventListener('click', async () => {
            const text = document.getElementById('tts-text').value;
            if (!text) return window.toast?.warning('Please enter text');

            btn.disabled = true;
            btn.textContent = 'Speaking...';

            try {
                // Pass object (not JSON string) - api.post will convert to URLSearchParams
                const blob = await api.post('/synthesize', { text, voice: 'default', speed: 1.0 });

                // api.post returns blob for non-JSON responses
                const url = URL.createObjectURL(blob);
                this.showResult('tts-output', 'audio', url);
            } catch (e) {
                window.toast?.error(e.message || 'Speech synthesis failed');
            }
            btn.disabled = false;
            btn.textContent = 'Generate';
        });
    }

    async initUpscaler() {
        const dropArea = document.getElementById('upscale-drop-zone');
        const fileInput = document.getElementById('upscale-file-input');
        const modelSelect = document.getElementById('upscale-model-select');
        const btn = document.getElementById('upscale-btn');

        if (!dropArea || !modelSelect) return;

        const handleFile = (file) => {
            if (!file.type.startsWith('image/')) return window.toast?.warning('Please upload an image file');
            const reader = new FileReader();
            reader.onload = (e) => {
                dropArea.innerHTML = `<img src="${e.target.result}" style="max-height: 150px; border-radius: 8px;"><p style="margin-top:8px;">${file.name}</p>`;
                // Store actual file object
                dropArea.fileObj = file;
                btn.disabled = false;
            };
            reader.readAsDataURL(file);
        };

        dropArea.onclick = () => fileInput.click();
        fileInput.onchange = (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); };

        // Handle Drag & Drop
        dropArea.addEventListener('dragover', (e) => { e.preventDefault(); dropArea.style.borderColor = 'var(--accent-primary)'; });
        dropArea.addEventListener('dragleave', (e) => { e.preventDefault(); dropArea.style.borderColor = 'var(--border-subtle)'; });
        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.style.borderColor = 'var(--border-subtle)';
            if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        });

        if (btn) {
            btn.addEventListener('click', async () => {
                const file = dropArea.fileObj;
                const modelId = modelSelect.value;
                const scale = document.getElementById('upscale-factor')?.value || 4;

                if (!file) return window.toast?.warning('Please upload an image');
                if (!window.isCloudMode && !modelId) return window.toast?.warning('Please select a model');

                btn.disabled = true;
                btn.textContent = 'Upscaling...';

                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('scale', scale);
                    // modelId not used by backend currently (uses loaded model) but could be passed if needed

                    const res = await api.post('/upscale', formData);
                    this.showResult('upscale-output', 'image', res.upscaled_url);
                } catch (e) {
                    window.toast?.error(`Upscale Error: ${e.message}`);
                }
                btn.disabled = false;
                btn.textContent = 'Upscale Image';
            });
        }
    }

    async showResult(containerId, type, url) {
        // Full URL for potential direct access/debugging
        // Don't modify blob: URLs or http(s): URLs - only prepend baseUrl for relative paths
        let fullUrl = (url.startsWith('http') || url.startsWith('blob:')) ? url : `${api.baseUrl}${url}`;
        let displayUrl = fullUrl;

        const container = document.getElementById(containerId);
        if (!container) return;

        container.style.display = 'block';
        container.innerHTML = '<div class="loading-spinner">Loading content...</div>';
        container.classList.add('active');

        try {
            // For images, use the API downloader to get a Blob (bypasses Auth/CORS/Browser issues)
            if (type === 'image' && !url.startsWith('blob:')) {
                const blob = await api.download(url);
                displayUrl = URL.createObjectURL(blob);
            }

            if (type === 'image') {
                container.innerHTML = `
                    <div class="generated-image-container" style="position: relative; display: inline-block; text-align: center; width: 100%;">
                        <img src="${displayUrl}" alt="Generated image" 
                             style="max-width: 100%; max-height: 400px; width: auto; border-radius: 8px; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                        <div class="image-actions" style="margin-top: 10px; display: flex; gap: 10px; justify-content: center;">
                            <a href="${displayUrl}" download="generated_${Date.now()}.png" class="btn-secondary btn-sm" title="Download">
                                Download
                            </a>
                            <button class="btn-secondary btn-sm" onclick="window.open('${displayUrl}', '_blank')" title="Open full size">
                                Open Full Size
                            </button>
                        </div>
                    </div>
                `;
            } else if (type === 'audio') {
                // Audio elements handle src fairly well, but could also blobify if needed.
                // For now, sticking to standard URL for streaming support, unless user reports issue.
                container.innerHTML = `
                    <div class="generated-audio-container">
                        <audio controls src="${fullUrl}" style="width:100%"></audio>
                        <div style="margin-top: 8px;">
                            <a href="${fullUrl}" download class="btn-secondary btn-sm">Download Audio</a>
                        </div>
                    </div>
                `;
            } else if (type === 'video') {
                container.innerHTML = `
                    <div class="generated-video-container">
                        <video controls src="${fullUrl}" style="max-width: 100%; border-radius: 8px;"></video>
                        <div style="margin-top: 8px;">
                            <a href="${fullUrl}" download class="btn-secondary btn-sm">Download Video</a>
                        </div>
                    </div>
                `;
            } else if (type === 'model') {
                container.innerHTML = `<a href="${fullUrl}" class="btn-secondary" download>Download 3D Model</a>`;
            }

        } catch (error) {
            console.error("Error loading result:", error);
            container.innerHTML = `
                <div class="image-error" style="padding: 20px; text-align: center; color: var(--text-error); background: var(--bg-secondary); border-radius: 8px;">
                     <p>Failed to load content</p>
                     <p style="font-size: 0.8em; opacity: 0.7;">${error.message}</p>
                </div>
            `;
        }
    }

    async init3DGen() {
        const dropArea = document.getElementById('3d-drop-zone');
        const fileInput = document.getElementById('3d-file-input');
        const modelSelect = document.getElementById('3d-model-select');
        const btn = document.getElementById('generate-3d-btn');

        if (!dropArea || !modelSelect) return;

        modelSelect.innerHTML = '<option value="sf3d">Stable Fast 3D</option>';

        const handleFile = (file) => {
            if (!file.type.startsWith('image/')) return window.toast?.warning('Please upload an image file');
            dropArea.innerHTML = `<p>Selected: ${file.name}</p>`;
            dropArea.fileObj = file;
            btn.disabled = false;
        };

        dropArea.onclick = () => fileInput.click();
        fileInput.onchange = (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); };

        if (btn) {
            btn.addEventListener('click', async () => {
                const file = dropArea.fileObj;
                if (!file) return window.toast?.warning('Please upload an image first');

                btn.disabled = true;
                btn.textContent = 'Generating 3D...';

                try {
                    const formData = new FormData();
                    formData.append('images', file);
                    // formData.append('steps', 50);

                    const res = await api.post('/generate-3d', formData);
                    this.showResult('3d-output', 'model', res.model_url);
                } catch (e) {
                    window.toast?.error(`3D Generation Failed: ${e.message}`);
                }
                btn.disabled = false;
                btn.textContent = 'Generate 3D Model';
            });
        }
    }

    async initBGRemover() {
        const dropArea = document.getElementById('bg-drop-zone');
        const fileInput = document.getElementById('bg-file-input');
        const modelSelect = document.getElementById('bg-model-select');
        const btn = document.getElementById('remove-bg-btn');

        if (!dropArea || !modelSelect) return;

        const handleFile = (file) => {
            if (!file.type.startsWith('image/')) return window.toast?.warning('Please upload an image file');
            dropArea.innerHTML = `<p>Selected: ${file.name}</p>`;
            dropArea.fileObj = file;
            btn.disabled = false;
        };

        dropArea.onclick = () => fileInput.click();
        fileInput.onchange = (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); };

        if (btn) {
            btn.addEventListener('click', async () => {
                const file = dropArea.fileObj;
                const modelId = modelSelect.value;
                if (!file || (!window.isCloudMode && !modelId)) return window.toast?.warning('Please upload a file and select a model');

                btn.disabled = true;
                btn.textContent = 'Removing Background...';

                try {
                    const formData = new FormData();
                    formData.append('file', file);

                    const res = await api.post('/remove_background', formData);
                    this.showResult('bg-output', 'image', res.image_url);
                } catch (e) { window.toast?.error(e.message || 'Background removal failed'); }

                btn.disabled = false;
                btn.textContent = 'Remove Background';
            });
        }
    }

    async initFeedback() {
        const btn = document.getElementById('submit-feedback-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const type = document.getElementById('feedback-type').value;
            const email = document.getElementById('feedback-email').value;
            const desc = document.getElementById('feedback-desc').value;

            if (!desc.trim()) {
                window.toast?.warning('Please enter a description');
                return;
            }

            if (email && !email.includes('@')) {
                window.toast?.warning('Please enter a valid email address');
                return;
            }

            const originalText = btn.textContent;
            btn.disabled = true;
            btn.textContent = 'Sending...';

            try {
                // Use api.post with JSON headers
                await api.post('/send_feedback',
                    { type, email, description: desc },
                    { 'Content-Type': 'application/json' }
                );

                btn.textContent = 'Success ✅';
                // Clear form
                document.getElementById('feedback-desc').value = '';

                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.disabled = false;
                }, 2000);

            } catch (error) {
                console.error('Feedback Error:', error);
                btn.textContent = 'Failed ❌';
                window.toast?.error('Feedback failed: ' + error.message);

                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.disabled = false;
                }, 2000);
            }
        });
    }

    async initVocalSeparator() {
        const dropArea = document.getElementById('vocal-drop-zone');
        const fileInput = document.getElementById('vocal-file-input');
        const modelSelect = document.getElementById('vocal-model-select');
        const btn = document.getElementById('separate-btn');

        if (!dropArea || !modelSelect) return;

        const handleFile = (file) => {
            if (!file.type.startsWith('audio/')) return window.toast?.warning('Please upload an audio file');
            dropArea.innerHTML = `<p>Selected: ${file.name}</p>`;
            dropArea.fileObj = file;
            btn.disabled = false;
        };

        dropArea.onclick = () => fileInput.click();
        fileInput.onchange = (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); };

        if (btn) {
            btn.addEventListener('click', async () => {
                const file = dropArea.fileObj;
                const modelId = modelSelect.value;
                if (!file || (!window.isCloudMode && !modelId)) return window.toast?.warning('Please upload a file and select a model');

                btn.disabled = true;
                btn.textContent = 'Separating...';

                try {
                    const formData = new FormData();
                    formData.append('file', file);

                    const res = await api.post('/process', formData);
                    // Result: res.vocals_url, res.instrumental_url
                    const container = document.getElementById('vocal-output');
                    container.style.display = 'block';
                    container.innerHTML = `
                        <div style="margin-bottom:12px;">
                            <h4 style="margin-bottom:8px; color:var(--text-secondary);">Vocals</h4>
                            <audio controls src="${api.baseURL}${res.vocals_url}" style="width:100%"></audio>
                        </div>
                        <div>
                            <h4 style="margin-bottom:8px; color:var(--text-secondary);">Instrumental</h4>
                            <audio controls src="${api.baseURL}${res.instrumental_url}" style="width:100%"></audio>
                        </div>
                     `;
                } catch (e) { window.toast?.error(e.message || 'Vocal separation failed'); }

                btn.disabled = false;
                btn.textContent = 'Separate Vocals';
            });
        }
    }
}
