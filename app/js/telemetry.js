// V6rge Telemetry Module (PostHog)
// API Key provided by user

const POSTHOG_KEY = 'phc_mezOgzYDrmB4JNDIfpPphFBeI5kGnHb0JBOjjzuTRX8';
const POSTHOG_HOST = 'https://us.i.posthog.com'; // Standard US host

class TelemetryService {
    constructor() {
        this.initialized = false;
        this.userId = null;
    }

    async init() {
        if (this.initialized) return;

        try {
            // 1. Load PostHog Library (Snippet approach for Electron/Web)
            !function (t, e) { var o, n, p, r; e.__SV || (window.posthog = e, e._i = [], e.init = function (i, s, a) { function g(t, e) { var o = e.split("."); 2 == o.length && (t = t[o[0]], e = o[1]), t[e] = function () { t.push([e].concat(Array.prototype.slice.call(arguments, 0))) } } (p = t.createElement("script")).type = "text/javascript", p.crossOrigin = "anonymous", p.async = !0, p.src = s.api_host + "/static/array.js", (r = t.getElementsByTagName("script")[0]).parentNode.insertBefore(p, r); var u = e; for (void 0 !== a ? u = e[a] = [] : a = "posthog", u.people = u.people || [], u.toString = function (t) { var e = "posthog"; return "posthog" !== a && (e += "." + a), t || (e += " (stub)"), e }, u.people.toString = function () { return u.toString(1) + ".people (stub)" }, o = "init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId".split(" "), n = 0; n < o.length; n++)g(u, o[n]); e._i.push([i, s, a]) }, e.__SV = 1) }(document, window.posthog || []);

            // 2. Fetch User ID from Backend
            const config = await window.v6rgeAPI.get('/config');
            this.userId = config.user_id;

            // 3. Initialize PostHog
            posthog.init(POSTHOG_KEY, {
                api_host: POSTHOG_HOST,
                loaded: (ph) => {
                    // Identify the user immediately
                    if (this.userId) {
                        ph.identify(this.userId);
                    }
                },
                autocapture: false, // Disable autocapture to keep it clean/privacy-focused
                persistence: 'localStorage'
            });
            posthog.debug(true); // Enable debug for now

            this.initialized = true;
            console.log('[Telemetry] Initialized for User:', this.userId);

        } catch (err) {
            console.error('[Telemetry] Failed to initialize:', err);
        }
    }

    track(eventName, properties = {}) {
        if (!this.initialized && window.posthog) {
            // Try to send anyway if posthog global exists but init logic failed? 
            // Better to rely on posthog queue.
        }

        if (window.posthog) {
            window.posthog.capture(eventName, properties);
        }
    }
}

// Export singleton
window.telemetry = new TelemetryService();

// Auto-init on load
// We need v6rgeAPI to be ready, so we wait briefly or hook into app load
// Auto-init on load with retry
window.addEventListener('DOMContentLoaded', () => {
    let attempts = 0;
    const maxAttempts = 10;

    const tryInit = async () => {
        try {
            await window.telemetry.init();
            if (!window.telemetry.initialized && attempts < maxAttempts) {
                attempts++;
                console.log(`[Telemetry] Backend not ready, retrying (${attempts}/${maxAttempts})...`);
                setTimeout(tryInit, 2000); // Retry every 2 seconds
            }
        } catch (e) {
            if (attempts < maxAttempts) {
                attempts++;
                setTimeout(tryInit, 2000);
            }
        }
    };

    // Initial wait
    setTimeout(tryInit, 1000);
});
