// V6rge API Client - Backend Integration Module

class V6rgeAPI {
    constructor() {
        this.baseURL = localStorage.getItem('backend-url') || 'http://127.0.0.1:5000';
    }

    setBaseURL(url) {
        this.baseURL = url.replace(/\/$/, ''); // Remove trailing slash
        localStorage.setItem('backend-url', this.baseURL);
    }

    getBaseURL() {
        return this.baseURL;
    }

    async checkConnection() {
        if (!this.baseURL) throw new Error('Backend URL not configured');
        const response = await fetch(`${this.baseURL}/`, { method: 'GET' });
        if (!response.ok) throw new Error('Backend unreachable');
        return await response.text();
    }

    // Generic Helpers
    async get(endpoint) {
        const response = await fetch(`${this.baseURL}${endpoint}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`GET ${endpoint} failed: ${response.statusText}`);
        return await response.json();
    }

    async post(endpoint, body = {}) {
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!response.ok) throw new Error(`POST ${endpoint} failed: ${response.statusText}`);
        return await response.json();
    }

    async delete(endpoint) {
        const response = await fetch(`${this.baseURL}${endpoint}`, { method: 'DELETE' });
        if (!response.ok) throw new Error(`DELETE ${endpoint} failed: ${response.statusText}`);
        return await response.json();
    }

    // ==================== IMAGE GENERATION ====================
    async generateImage(prompt, width = 1024, height = 1024) {
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('width', width);
        formData.append('height', height);

        const response = await fetch(`${this.baseURL}/generate_image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Image generation failed');
        }

        const data = await response.json();

        // Track
        if (window.telemetry) window.telemetry.track('image_generated', { prompt_length: prompt.length, width, height });

        return `${this.baseURL}${data.image_url}`;
    }

    // ==================== MUSIC GENERATION ====================
    async generateMusic(prompt, duration = 10, melodyFile = null) {
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('duration', duration);
        if (melodyFile) formData.append('melody', melodyFile);

        const response = await fetch(`${this.baseURL}/generate_music`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Music generation failed');
        }

        const data = await response.json();

        if (window.telemetry) window.telemetry.track('music_generated', { prompt_length: prompt.length, duration });

        return `${this.baseURL}${data.audio_url}`;
    }

    // ==================== TEXT-TO-SPEECH ====================
    async synthesizeSpeech(text, options = {}) {
        const formData = new FormData();
        formData.append('text', text);

        if (options.seed) formData.append('seed', options.seed);
        if (options.temperature) formData.append('temperature', options.temperature);
        if (options.audioPrompt) formData.append('audio_prompt', options.audioPrompt);

        const response = await fetch(`${this.baseURL}/synthesize`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'TTS failed');
        }

        if (window.telemetry) window.telemetry.track('tts_synthesized', { text_length: text.length });
        return response.blob();
    }

    // ==================== VIDEO GENERATION ====================
    async generateVideo(prompt) {
        const formData = new FormData();
        formData.append('prompt', prompt);

        const response = await fetch(`${this.baseURL}/generate_video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Video generation failed');
        }

        const data = await response.json();
        if (window.telemetry) window.telemetry.track('video_generated', { prompt_length: prompt.length });
        return `${this.baseURL}${data.video_url}`;
    }

    // ==================== 3D GENERATION ====================
    async generate3D(imageFile, format = 'glb', resolution = 1024, steps = 50) {
        const formData = new FormData();
        formData.append('images', imageFile);
        formData.append('format', format);
        formData.append('resolution', resolution);
        formData.append('steps', steps);

        const response = await fetch(`${this.baseURL}/generate-3d`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '3D generation failed');
        }

        const data = await response.json();
        if (window.telemetry) window.telemetry.track('3d_generated', { format, steps });
        return data.model_url;
    }

    // ==================== IMAGE UPSCALING ====================
    async upscaleImage(imageFile, scale = 4) {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('scale', scale);

        const response = await fetch(`${this.baseURL}/upscale`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upscaling failed');
        }

        const data = await response.json();
        if (window.telemetry) window.telemetry.track('image_upscaled', { scale });
        return `${this.baseURL}${data.upscaled_url}`;
    }

    // ==================== BACKGROUND REMOVAL ====================
    async removeBackground(imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);

        const response = await fetch(`${this.baseURL}/remove_background`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Background removal failed');
        }

        const data = await response.json();
        if (window.telemetry) window.telemetry.track('bg_removed', {});
        return `${this.baseURL}${data.image_url}`;
    }

    // ==================== VOCAL SEPARATION ====================
    async separateVocals(audioFile) {
        const formData = new FormData();
        formData.append('file', audioFile);

        const response = await fetch(`${this.baseURL}/process`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Vocal separation failed');
        }

        const data = await response.json();
        if (window.telemetry) window.telemetry.track('vocals_separated', {});
        return {
            vocals: `${this.baseURL}${data.vocals_url}`,
            instrumental: `${this.baseURL}${data.instrumental_url}`
        };
    }

    // ==================== CHAT ====================
    async chat(message, history = [], files = []) {
        const formData = new FormData();
        formData.append('message', message);
        formData.append('history', JSON.stringify(history));

        files.forEach((file, index) => {
            formData.append(`file_${index}`, file);
        });

        const response = await fetch(`${this.baseURL}/chat`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Chat failed');
        }

        if (window.telemetry) window.telemetry.track('chat_message_sent', { has_files: files.length > 0 });
        return await response.json();
    }
}

// Export singleton instance
export const api = new V6rgeAPI();
window.v6rgeAPI = api;
