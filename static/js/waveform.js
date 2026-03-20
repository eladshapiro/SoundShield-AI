/**
 * SoundShield-AI  --  WaveSurfer.js Integration (Alpine.js)
 *
 * Audio waveform visualization with incident regions, playback controls,
 * and dark-mode aware styling.
 */

/* ------------------------------------------------------------------ */
/*  Incident-type color map                                            */
/* ------------------------------------------------------------------ */
const REGION_COLORS = {
  violence: { color: '#EF4444', opacity: 0.2 },
  emotion:  { color: '#F97316', opacity: 0.2 },
  emotions: { color: '#F97316', opacity: 0.2 },
  cry:      { color: '#3B82F6', opacity: 0.2 },
  cries:    { color: '#3B82F6', opacity: 0.2 },
  neglect:  { color: '#6B7280', opacity: 0.2 },
  language: { color: '#8B5CF6', opacity: 0.2 },
};

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ------------------------------------------------------------------ */
/*  Alpine component: waveformManager                                  */
/* ------------------------------------------------------------------ */
function waveformManager() {
  return {
    /* ---- state ---- */
    wavesurfer: null,
    isPlaying: false,
    isReady: false,
    currentTime: 0,
    totalDuration: 0,
    regions: [],
    _regionsPlugin: null,

    /* ---- init waveform ---- */
    initWaveform(containerEl) {
      if (!containerEl) {
        console.error('[SoundShield] waveformManager: no container element provided');
        return;
      }

      // Clean up previous instance
      this.destroy();

      const isDark = document.documentElement.classList.contains('dark');

      // Check for WaveSurfer availability
      if (typeof WaveSurfer === 'undefined') {
        console.error('[SoundShield] WaveSurfer is not loaded');
        return;
      }

      const wsOptions = {
        container: containerEl,
        height: 128,
        waveColor: isDark ? '#64748B' : '#94A3B8',
        progressColor: '#2563EB',
        cursorColor: '#2563EB',
        cursorWidth: 2,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        responsive: true,
        normalize: true,
        partialRender: true,
        hideScrollbar: false,
        fillParent: true,
        minPxPerSec: 50,
        autoCenter: true,
        backend: 'WebAudio',
      };

      // Create Regions plugin if available
      if (typeof WaveSurfer.Regions !== 'undefined') {
        this._regionsPlugin = WaveSurfer.Regions.create();
        wsOptions.plugins = [this._regionsPlugin];
      } else if (WaveSurfer.regions) {
        // Older plugin pattern
        this._regionsPlugin = WaveSurfer.regions.create();
        wsOptions.plugins = [this._regionsPlugin];
      }

      try {
        this.wavesurfer = WaveSurfer.create(wsOptions);
      } catch (err) {
        console.error('[SoundShield] WaveSurfer.create failed:', err);
        return;
      }

      // Events
      this.wavesurfer.on('ready', () => {
        this.isReady = true;
        this.totalDuration = this.wavesurfer.getDuration();
        document.dispatchEvent(new CustomEvent('waveform-ready', {
          detail: { duration: this.totalDuration }
        }));
      });

      this.wavesurfer.on('audioprocess', () => {
        this.currentTime = this.wavesurfer.getCurrentTime();
      });

      this.wavesurfer.on('seeking', () => {
        this.currentTime = this.wavesurfer.getCurrentTime();
      });

      this.wavesurfer.on('play', () => {
        this.isPlaying = true;
      });

      this.wavesurfer.on('pause', () => {
        this.isPlaying = false;
      });

      this.wavesurfer.on('finish', () => {
        this.isPlaying = false;
      });

      // Dark mode listener
      this._darkModeHandler = (e) => {
        if (this.wavesurfer) {
          const dark = e.detail.darkMode;
          this.wavesurfer.setOptions({
            waveColor: dark ? '#64748B' : '#94A3B8',
          });
        }
      };
      document.addEventListener('dark-mode-changed', this._darkModeHandler);
    },

    /* ---- load audio ---- */
    loadAudio(filename) {
      if (!this.wavesurfer) {
        console.warn('[SoundShield] waveformManager: wavesurfer not initialized');
        return;
      }

      this.isReady = false;
      this.isPlaying = false;
      this.currentTime = 0;
      this.totalDuration = 0;
      this.clearRegions();

      const encodedFilename = encodeURIComponent(filename);
      this.wavesurfer.load(`/uploaded-audio/${encodedFilename}`);
    },

    /* ---- incident regions ---- */
    addIncidentRegions(incidents) {
      if (!this.wavesurfer || !incidents || !Array.isArray(incidents)) return;

      this.clearRegions();

      incidents.forEach((incident, index) => {
        const type = (incident.type || incident.category || '').toLowerCase();
        const colorInfo = REGION_COLORS[type] || { color: '#6B7280', opacity: 0.2 };

        const start = incident.start_time || incident.start || incident.timestamp || 0;
        const end = incident.end_time || incident.end || (start + (incident.duration || 2));

        const regionData = {
          id: `incident-${index}`,
          start: start,
          end: end,
          color: hexToRgba(colorInfo.color, colorInfo.opacity),
          drag: false,
          resize: false,
          content: incident.label || incident.description || type,
        };

        let region = null;

        // Try WaveSurfer v7+ Regions plugin API
        if (this._regionsPlugin && typeof this._regionsPlugin.addRegion === 'function') {
          region = this._regionsPlugin.addRegion(regionData);
        } else if (this.wavesurfer.addRegion) {
          // Older WaveSurfer API
          region = this.wavesurfer.addRegion(regionData);
        }

        if (region) {
          this.regions.push(region);

          // Click handler on region
          const regionClickHandler = () => {
            document.dispatchEvent(new CustomEvent('region-clicked', {
              detail: {
                incident: incident,
                index: index,
                start: start,
                end: end,
                type: type,
              }
            }));

            // Scroll to incident finding in the UI
            const findingEl = document.getElementById(`finding-${index}`);
            if (findingEl) {
              findingEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
              findingEl.classList.add('ss-finding--active');
              setTimeout(() => findingEl.classList.remove('ss-finding--active'), 2000);
            }

            // Play that segment
            this.playSegment(start, end);
          };

          // Attach click handler based on API version
          if (region.on) {
            region.on('click', regionClickHandler);
          } else if (region.element) {
            region.element.addEventListener('click', regionClickHandler);
          }
        }
      });
    },

    clearRegions() {
      if (this._regionsPlugin && typeof this._regionsPlugin.clearRegions === 'function') {
        this._regionsPlugin.clearRegions();
      } else if (this.wavesurfer && typeof this.wavesurfer.clearRegions === 'function') {
        this.wavesurfer.clearRegions();
      } else {
        // Manual cleanup
        this.regions.forEach((r) => {
          if (r && typeof r.remove === 'function') r.remove();
        });
      }
      this.regions = [];
    },

    /* ---- playback controls ---- */
    playPause() {
      if (!this.wavesurfer || !this.isReady) return;
      this.wavesurfer.playPause();
    },

    stop() {
      if (!this.wavesurfer) return;
      this.wavesurfer.stop();
      this.isPlaying = false;
      this.currentTime = 0;
    },

    seekTo(time) {
      if (!this.wavesurfer || !this.isReady) return;
      const duration = this.wavesurfer.getDuration();
      if (duration > 0) {
        const progress = Math.max(0, Math.min(1, time / duration));
        this.wavesurfer.seekTo(progress);
        this.currentTime = time;
      }
    },

    playSegment(start, end) {
      if (!this.wavesurfer || !this.isReady) return;
      this.seekTo(start);
      this.wavesurfer.play(start, end);
    },

    /* ---- formatting ---- */
    formatTime(seconds) {
      if (!seconds && seconds !== 0) return '0:00';
      const m = Math.floor(seconds / 60);
      const s = Math.floor(seconds % 60);
      return `${m}:${s.toString().padStart(2, '0')}`;
    },

    /* ---- destroy / cleanup ---- */
    destroy() {
      if (this._darkModeHandler) {
        document.removeEventListener('dark-mode-changed', this._darkModeHandler);
        this._darkModeHandler = null;
      }

      this.clearRegions();

      if (this.wavesurfer) {
        try {
          this.wavesurfer.destroy();
        } catch (_) { /* ignore */ }
        this.wavesurfer = null;
      }

      this._regionsPlugin = null;
      this.isReady = false;
      this.isPlaying = false;
      this.currentTime = 0;
      this.totalDuration = 0;
    },

    /* ---- init ---- */
    init() {
      // Listen for analysis complete to auto-load waveform
      document.addEventListener('analysis-complete', (e) => {
        const data = e.detail;
        if (data && data.filename) {
          // Small delay to ensure the container is rendered
          this.$nextTick(() => {
            const container = document.getElementById('waveform-container');
            if (container) {
              if (!this.wavesurfer) {
                this.initWaveform(container);
              }
              this.loadAudio(data.filename);

              // Once ready, add incident regions
              const onReady = () => {
                const allIncidents = this._collectIncidents(data);
                if (allIncidents.length > 0) {
                  this.addIncidentRegions(allIncidents);
                }
                document.removeEventListener('waveform-ready', onReady);
              };
              document.addEventListener('waveform-ready', onReady);
            }
          });
        }
      });

      // Listen for region-click requests from findings list
      document.addEventListener('seek-to-incident', (e) => {
        const { start, end } = e.detail;
        if (typeof start === 'number') {
          this.playSegment(start, end || start + 2);
        }
      });
    },

    /* ---- collect all incidents from results ---- */
    _collectIncidents(results) {
      const incidents = [];

      // Flatten incidents from various result categories
      const categories = [
        { key: 'violence', type: 'violence' },
        { key: 'violence_incidents', type: 'violence' },
        { key: 'emotions', type: 'emotion' },
        { key: 'emotion_incidents', type: 'emotion' },
        { key: 'cries', type: 'cry' },
        { key: 'cry_incidents', type: 'cry' },
        { key: 'neglect', type: 'neglect' },
        { key: 'neglect_incidents', type: 'neglect' },
        { key: 'language', type: 'language' },
        { key: 'language_incidents', type: 'language' },
        { key: 'incidents', type: 'mixed' },
      ];

      for (const cat of categories) {
        const items = results[cat.key];
        if (Array.isArray(items)) {
          items.forEach((item) => {
            incidents.push({
              ...item,
              type: item.type || item.category || cat.type,
            });
          });
        }
      }

      // Also handle flat "results.incidents" if it's an object with sub-arrays
      if (results.incidents && typeof results.incidents === 'object' && !Array.isArray(results.incidents)) {
        for (const [type, items] of Object.entries(results.incidents)) {
          if (Array.isArray(items)) {
            items.forEach((item) => {
              incidents.push({ ...item, type: item.type || type });
            });
          }
        }
      }

      // Sort by start time
      incidents.sort((a, b) => {
        const aStart = a.start_time || a.start || a.timestamp || 0;
        const bStart = b.start_time || b.start || b.timestamp || 0;
        return aStart - bStart;
      });

      return incidents;
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Register with Alpine                                               */
/* ------------------------------------------------------------------ */
document.addEventListener('alpine:init', () => {
  Alpine.data('waveformManager', waveformManager);
});

/* Export */
window.SoundShield = window.SoundShield || {};
window.SoundShield.waveformManager = waveformManager;
