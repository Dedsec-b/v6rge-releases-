import { api } from './api.js';

export class ModelManager {
    constructor() {
        this.models = {};
        this.pollingInterval = null;
        this.vramInfo = { vram_total_gb: 0, cuda_available: false };
    }

    async refresh() {
        try {
            // Fetch VRAM info once
            const vram = await api.get('/system/vram').catch(() => ({ vram_total_gb: 0, cuda_available: false }));
            this.vramInfo = vram;

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
            const nameEl = card.querySelector('.model-name');

            // Add VRAM badge if model has vram_requirement
            let vramBadge = card.querySelector('.vram-badge');
            if (info.vram_gb && this.vramInfo.cuda_available) {
                if (!vramBadge) {
                    vramBadge = document.createElement('span');
                    vramBadge.className = 'vram-badge';
                    nameEl?.appendChild(vramBadge);
                }

                const userVram = this.vramInfo.vram_total_gb;
                const reqVram = info.vram_gb;

                if (userVram >= reqVram) {
                    vramBadge.className = 'vram-badge compatible';
                    vramBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>${reqVram}GB`;
                } else if (userVram >= reqVram * 0.7) {
                    vramBadge.className = 'vram-badge warning';
                    vramBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>${reqVram}GB`;
                    vramBadge.title = `Requires ${reqVram}GB VRAM, you have ${userVram}GB - May run slowly`;
                } else {
                    vramBadge.className = 'vram-badge incompatible';
                    vramBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>${reqVram}GB`;
                    vramBadge.title = `Requires ${reqVram}GB VRAM, you only have ${userVram}GB`;
                }
            }

            // Update status with dot indicator
            if (info.downloaded) {
                statusEl.innerHTML = '<span class="status-dot ready"></span>Ready';
                statusEl.dataset.status = 'ready';
                btn.textContent = 'Delete';
                btn.className = 'btn-delete';
                btn.onclick = () => this.deleteModel(id);
                card.classList.remove('downloading', 'error');
            } else {
                statusEl.innerHTML = '<span class="status-dot not-downloaded"></span>Not Downloaded';
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
            window.toast?.error(`Download failed: ${e.message}`, { duration: 6000 });
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
                    window.toast?.error(`Download Error: ${prog.error || prog.message || 'Unknown error'}`, { duration: 8000 });
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
            window.toast?.error(`Delete failed: ${e.message}`);
        }
    }
}
