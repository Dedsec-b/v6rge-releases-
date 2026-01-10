export class API {
    constructor() {
        this.baseUrl = localStorage.getItem('backend-url') || 'http://localhost:5000';
    }

    setBaseUrl(url) {
        this.baseUrl = url.replace(/\/$/, '');
        localStorage.setItem('backend-url', this.baseUrl);
        window.isCloudMode = !this.baseUrl.includes('localhost') && !this.baseUrl.includes('127.0.0.1');
    }

    // Helper to get standard headers
    getHeaders(extras = {}) {
        return {
            'ngrok-skip-browser-warning': 'true',
            ...extras
        };
    }

    async checkConnection(url = this.baseUrl) {
        try {
            const res = await fetch(`${url}/health`, {
                mode: 'cors',
                headers: this.getHeaders()
            });
            return res.ok;
        } catch (e) {
            return false;
        }
    }

    async get(endpoint) {
        const res = await fetch(`${this.baseUrl}${endpoint}`, {
            headers: this.getHeaders()
        });
        if (!res.ok) throw new Error(`API Error ${res.status}: ${res.statusText}`);
        return await res.json();
    }

    async post(endpoint, body, headers = {}) {
        const isFormData = body instanceof FormData;
        const isJson = headers['Content-Type'] === 'application/json';
        
        const options = {
            method: 'POST',
            headers: { ...this.getHeaders(), ...headers }
        };

        if (isFormData) {
            options.body = body;
        } else if (isJson) {
            options.body = JSON.stringify(body);
        } else {
            // Default to x-www-form-urlencoded for backward compatibility
            options.headers['Content-Type'] = 'application/x-www-form-urlencoded';
            options.body = new URLSearchParams(body);
        }

        const res = await fetch(`${this.baseUrl}${endpoint}`, options);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || `API Error ${res.status}`);
        }

        // Handle Blob responses (Download) or JSON
        const contentType = res.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await res.json();
        }
        return await res.blob();
    }

    async delete(endpoint) {
        const res = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'DELETE',
            headers: this.getHeaders()
        });
        return await res.json();
    }
    async download(endpoint) {
        // Handle full URLs (if already constructed) or relative endpoints
        const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;

        const res = await fetch(url, {
            headers: this.getHeaders()
        });
        if (!res.ok) {
            throw new Error(`Download failed: ${res.status}`);
        }
        return await res.blob();
    }
}

export const api = new API();
