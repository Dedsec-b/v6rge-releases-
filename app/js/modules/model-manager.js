import { api } from './api.js';

export class ModelManager {
    constructor() {
        this.models = {};
        this.pollingInterval = null;
    }

    async refresh() {
        try {
            const data = await api.get('/models/status');
            this.models = data.models;
            this.updateUI(data);
            this.updateSelectors();
        } catch (e) {
            console.error('Failed to refresh models:', e);
        }
    }

    updateUI(data) {
        // Update Total Downloaded
        const totalEl = document.getElementById('total-downloaded');
        if (totalEl) totalEl.textContent = `${data.total_downloaded_gb} GB`;

        // Update Cards
        Object.entries(data.models).forEach(([id, info]) => {
            const card = document.querySelector(`.model-item[data-model-id="${id}"]`);
            if (!card) return;

            const statusEl = card.querySelector('.model-status');
            const btn = card.querySelector('button');

            if (info.downloaded) {
                statusEl.textContent = 'Ready';
                statusEl.dataset.status = 'ready';
                btn.textContent = 'Delete';
                btn.className = 'btn-delete';
                btn.onclick = () => this.deleteModel(id);
            } else {
                statusEl.textContent = 'Not Downloaded';
                statusEl.dataset.status = 'not-downloaded';
                btn.textContent = 'Download';
                btn.className = 'btn-download btn-primary';
                btn.onclick = () => this.downloadModel(id);
            }
        });
    }

    updateSelectors() {
        const selectors = document.querySelectorAll('.model-selector');
        selectors.forEach(select => {
            const category = select.dataset.category;
            const currentVal = select.value;

            // Clear
            select.innerHTML = '<option value="" disabled selected>Select Model...</option>';

            Object.entries(this.models).forEach(([id, info]) => {
                if (info.type === category && info.downloaded) {
                    const opt = document.createElement('option');
                    opt.value = id;
                    opt.textContent = info.name;
                    select.appendChild(opt);
                }
            });

            // Restore selection if valid
            if (currentVal && select.querySelector(`option[value="${currentVal}"]`)) {
                select.value = currentVal;
            }
        });
    }

    async downloadModel(modelId) {
        const card = document.querySelector(`.model-item[data-model-id="${modelId}"]`);
        const btn = card.querySelector('button');
        const statusEl = card.querySelector('.model-status');

        btn.disabled = true;
        btn.textContent = 'Starting...';

        try {
            await api.post(`/models/download/${modelId}`);
            statusEl.textContent = 'Downloading...';
            statusEl.dataset.status = 'downloading';

            // Start Polling
            this.pollProgress(modelId, card);
        } catch (e) {
            alert(`Download failed: ${e.message}`);
            btn.disabled = false;
            btn.textContent = 'Download';
        }
    }

    pollProgress(modelId, card) {
        const poll = async () => {
            try {
                const prog = await api.get(`/models/progress/${modelId}`);

                if (prog.status === 'downloading') {
                    // Render Progress Bar
                    let bar = card.querySelector('.progress-container');
                    if (!bar) {
                        bar = document.createElement('div');
                        bar.className = 'progress-container';
                        // Add basic structure
                        bar.innerHTML = `
                            <div class="progress-bar-bg"><div class="progress-bar-fill"></div></div>
                            <div class="progress-info">
                                <span class="p-msg"></span>
                                <span class="p-percent"></span>
                            </div>
                            <div class="progress-sub" style="font-size: 0.8em; opacity: 0.7; margin-top: 2px;"></div>
                        `;
                        const actions = card.querySelector('.model-actions') || card;
                        actions.appendChild(bar);
                    }

                    bar.querySelector('.progress-bar-fill').style.width = `${prog.progress}%`;
                    bar.querySelector('.p-msg').textContent = `Current File: ${(prog.downloaded / 1e6).toFixed(1)}MB / ${(prog.total / 1e6).toFixed(1)}MB`;
                    bar.querySelector('.p-percent').textContent = `${prog.progress.toFixed(1)}%`;
                    
                    if (this.models[modelId]) {
                        bar.querySelector('.progress-sub').textContent = `Installing ${this.models[modelId].name} (~${this.models[modelId].size_gb} GB Total)`;
                    }

                    setTimeout(poll, 500);
                } else if (prog.status === 'complete' || prog.status === 'ready') {
                    // Cleanup
                    const bar = card.querySelector('.progress-container');
                    if (bar) bar.remove();

                    // Save State for Scroll Memory
                    const activeView = document.querySelector('.view.active');
                    const activeNav = document.querySelector('.nav-item.active');
                    if (activeView && activeNav) {
                        sessionStorage.setItem('restore_scroll', activeView.scrollTop);
                        sessionStorage.setItem('restore_view', activeNav.dataset.view);
                    }

                    // Use Electron safe reload if available, else standard
                    if (window.v6rge && window.v6rge.reload) {
                        window.v6rge.reload();
                    } else {
                        window.location.reload();
                    }

                } else if (prog.status === 'error') {
                    alert(`Error: ${prog.message}`);
                    const bar = card.querySelector('.progress-container');
                    if (bar) bar.remove();
                    this.refresh();
                } else {
                    setTimeout(poll, 500);
                }
            } catch (e) {
                console.warn('Poll failed', e);
            }
        };
        poll();
    }

    async deleteModel(modelId) {
        if (!confirm('Are you sure you want to delete this model?')) return;

        try {
            await api.delete(`/models/delete/${modelId}`);

            // Save State for Scroll Memory
            const activeView = document.querySelector('.view.active');
            const activeNav = document.querySelector('.nav-item.active');
            if (activeView && activeNav) {
                sessionStorage.setItem('restore_scroll', activeView.scrollTop);
                sessionStorage.setItem('restore_view', activeNav.dataset.view);
            }

            if (window.v6rge && window.v6rge.reload) {
                window.v6rge.reload();
            } else {
                window.location.reload();
            }

        } catch (e) {
            alert(`Delete failed: ${e.message}`);
        }
    }
}
