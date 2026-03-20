/**
 * SoundShield-AI  --  Report Modal (Alpine.js)
 *
 * Full-featured accessible modal for viewing and downloading analysis reports.
 * Focus trapping, ESC close, backdrop click, aria attributes.
 */

/* ------------------------------------------------------------------ */
/*  Alpine component: modalManager                                     */
/* ------------------------------------------------------------------ */
function modalManager() {
  return {
    /* ---- state ---- */
    isOpen: false,
    reportData: null,
    reportFilename: '',
    isLoading: false,
    loadError: '',

    // Internal
    _previousFocus: null,
    _focusTrapHandler: null,
    _escHandler: null,

    /* ---- open report ---- */
    async openReport(filename) {
      if (!filename) return;

      this.reportFilename = filename;
      this.isLoading = true;
      this.loadError = '';
      this.reportData = null;
      this.isOpen = true;

      // Save current focus to restore later
      this._previousFocus = document.activeElement;

      // Prevent body scroll
      document.body.style.overflow = 'hidden';

      try {
        const encodedFilename = encodeURIComponent(filename);
        const res = await fetch(`/report/${encodedFilename}`);
        if (!res.ok) {
          const errText = await res.text().catch(() => '');
          throw new Error(errText || `HTTP ${res.status}`);
        }
        this.reportData = await res.json();
      } catch (err) {
        console.error('[SoundShield] Failed to load report:', err);
        this.loadError = err.message || 'Failed to load report';
      } finally {
        this.isLoading = false;
      }

      // Set up accessibility
      this.$nextTick(() => {
        this._setupFocusTrap();
        this._setupEscClose();

        // Focus the modal or first focusable element
        const modal = document.getElementById('report-modal');
        if (modal) {
          const firstFocusable = modal.querySelector(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          );
          if (firstFocusable) {
            firstFocusable.focus();
          } else {
            modal.focus();
          }
        }
      });
    },

    /* ---- close report ---- */
    closeReport() {
      this.isOpen = false;
      this.reportData = null;
      this.reportFilename = '';
      this.loadError = '';

      // Restore body scroll
      document.body.style.overflow = '';

      // Restore focus
      if (this._previousFocus && typeof this._previousFocus.focus === 'function') {
        this._previousFocus.focus();
      }
      this._previousFocus = null;

      // Remove event listeners
      this._teardownFocusTrap();
      this._teardownEscClose();
    },

    /* ---- backdrop click ---- */
    onBackdropClick(e) {
      // Only close if clicking the overlay itself, not the modal content
      if (e.target === e.currentTarget) {
        this.closeReport();
      }
    },

    /* ---- download ---- */
    downloadReport(filename, format) {
      const fn = filename || this.reportFilename;
      if (!fn) return;

      const encodedFilename = encodeURIComponent(fn);
      let url = `/download/${encodedFilename}`;
      if (format) {
        url += `?format=${encodeURIComponent(format)}`;
      }

      // Trigger download via hidden link
      const a = document.createElement('a');
      a.href = url;
      a.download = '';
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    },

    /* ---- focus trapping ---- */
    _setupFocusTrap() {
      this._focusTrapHandler = (e) => {
        if (e.key !== 'Tab') return;

        const modal = document.getElementById('report-modal');
        if (!modal) return;

        const focusable = modal.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );

        if (focusable.length === 0) return;

        const first = focusable[0];
        const last = focusable[focusable.length - 1];

        if (e.shiftKey) {
          // Shift+Tab: if focus on first, wrap to last
          if (document.activeElement === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          // Tab: if focus on last, wrap to first
          if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }
      };

      document.addEventListener('keydown', this._focusTrapHandler);
    },

    _teardownFocusTrap() {
      if (this._focusTrapHandler) {
        document.removeEventListener('keydown', this._focusTrapHandler);
        this._focusTrapHandler = null;
      }
    },

    /* ---- ESC to close ---- */
    _setupEscClose() {
      this._escHandler = (e) => {
        if (e.key === 'Escape' && this.isOpen) {
          e.preventDefault();
          this.closeReport();
        }
      };
      document.addEventListener('keydown', this._escHandler);
    },

    _teardownEscClose() {
      if (this._escHandler) {
        document.removeEventListener('keydown', this._escHandler);
        this._escHandler = null;
      }
    },

    /* ---- template helpers ---- */

    /**
     * Get severity badge class for a risk level string.
     */
    severityClass(level) {
      const map = { low: 'low', medium: 'medium', high: 'high', critical: 'critical' };
      return map[(level || '').toLowerCase()] || 'low';
    },

    /**
     * Get incident type CSS class.
     */
    incidentTypeClass(type) {
      const map = {
        violence: 'violence',
        emotion: 'emotion',
        emotions: 'emotion',
        cry: 'cry',
        cries: 'cry',
        neglect: 'neglect',
        language: 'language',
      };
      return map[(type || '').toLowerCase()] || 'emotion';
    },

    /**
     * Format seconds to MM:SS.
     */
    formatTime(seconds) {
      if (!seconds && seconds !== 0) return '0:00';
      const m = Math.floor(seconds / 60);
      const s = Math.floor(seconds % 60);
      return `${m}:${s.toString().padStart(2, '0')}`;
    },

    /**
     * Format ISO date string.
     */
    formatDate(iso) {
      if (!iso) return '--';
      try {
        return new Date(iso).toLocaleDateString('en-US', {
          year: 'numeric', month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        });
      } catch (_) {
        return iso;
      }
    },

    /**
     * Collect all findings organized by category from report data.
     * Returns { violence: [...], emotion: [...], cry: [...], neglect: [...], language: [...] }
     */
    getOrganizedFindings() {
      if (!this.reportData) return {};

      const organized = {
        violence: [],
        emotion: [],
        cry: [],
        neglect: [],
        language: [],
      };

      const categoryMap = {
        violence: 'violence',
        violence_incidents: 'violence',
        emotions: 'emotion',
        emotion: 'emotion',
        emotion_incidents: 'emotion',
        cries: 'cry',
        cry: 'cry',
        cry_incidents: 'cry',
        neglect: 'neglect',
        neglect_incidents: 'neglect',
        language: 'language',
        language_incidents: 'language',
      };

      const data = this.reportData;

      for (const [key, category] of Object.entries(categoryMap)) {
        if (Array.isArray(data[key])) {
          organized[category].push(...data[key]);
        }
      }

      // Handle nested incidents object
      if (data.incidents && typeof data.incidents === 'object' && !Array.isArray(data.incidents)) {
        for (const [key, items] of Object.entries(data.incidents)) {
          const category = categoryMap[key] || key;
          if (Array.isArray(items) && organized[category]) {
            organized[category].push(...items);
          }
        }
      }

      // Also handle flat incidents array
      if (Array.isArray(data.incidents)) {
        data.incidents.forEach((inc) => {
          const type = (inc.type || inc.category || '').toLowerCase();
          const category = categoryMap[type] || type;
          if (organized[category]) {
            organized[category].push(inc);
          }
        });
      }

      return organized;
    },

    /**
     * Get recommendations list.
     */
    getRecommendations() {
      if (!this.reportData) return [];
      return this.reportData.recommendations ||
             this.reportData.suggestions ||
             [];
    },

    /**
     * Get summary text.
     */
    getSummary() {
      if (!this.reportData) return '';
      return this.reportData.summary ||
             this.reportData.description ||
             this.reportData.overview ||
             '';
    },

    /**
     * Build the audio clip URL for an incident.
     */
    getClipUrl(incident) {
      if (incident.clip_url) return incident.clip_url;
      if (incident.audio_clip) return incident.audio_clip;

      // Construct from filename and timestamps
      const fn = this.reportFilename;
      const start = incident.start_time || incident.start || 0;
      const end = incident.end_time || incident.end || (start + 2);
      return `/audio-clip/${encodeURIComponent(fn)}?start=${start}&end=${end}`;
    },

    /**
     * Seek the main waveform to this incident's start time.
     */
    seekToIncident(incident) {
      const start = incident.start_time || incident.start || incident.timestamp || 0;
      const end = incident.end_time || incident.end || (start + 2);
      document.dispatchEvent(new CustomEvent('seek-to-incident', {
        detail: { start, end, incident }
      }));
    },

    /* ---- lifecycle ---- */
    init() {
      // Nothing needed on init
    },

    destroy() {
      if (this.isOpen) {
        this.closeReport();
      }
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Register with Alpine                                               */
/* ------------------------------------------------------------------ */
document.addEventListener('alpine:init', () => {
  Alpine.data('modalManager', modalManager);
});

/* Export */
window.SoundShield = window.SoundShield || {};
window.SoundShield.modalManager = modalManager;
