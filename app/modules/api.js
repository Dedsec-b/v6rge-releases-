export class API {
    constructor() {
        this.baseUrl = localStorage.getItem('backend-url') || 'http://localhost:5000';
    }

    setBaseUrl(url) {
        this.baseUrl = url.replace(/\/$/, '');
        localStorage.setItem('backend-url', this.baseUrl);
    }

    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
        return await response.json();
    }

    async post(endpoint, body) {
        const headers = { 'Content-Type': 'application/json' };
        
        // Handle form data automatically
        if (body instanceof FormData) {
            delete headers['Content-Type'];
        } else if (body instanceof URLSearchParams) {
             headers['Content-Type'] = 'application/x-www-form-urlencoded';
        }

        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: body instanceof FormData ? undefined : headers,
            body: body instanceof FormData ? body : (
                (typeof body === 'object' && !(body instanceof URLSearchParams)) 
                ? JSON.stringify(body) 
                : body
            )
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || `API Error: ${response.statusText}`);
        }
        return await response.json();
    }

    async delete(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || `API Error: ${response.statusText}`);
        }
        return await response.json();
    }

    // Helper for checking connection
    async checkConnection(url = this.baseUrl) {
        try {
            const response = await fetch(`${url}/health`, { method: 'GET', mode: 'cors' });
            return response.ok;
        } catch (e) {
            return false;
        }
    }
}

export const api = new API();
