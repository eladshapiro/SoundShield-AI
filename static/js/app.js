/**
 * SoundShield-AI  --  Main Application State (Alpine.js)
 *
 * Provides global state, language switching (EN/HE), dark-mode, history,
 * and a translation helper.  Register with Alpine.data('appState', appState).
 */

/* ------------------------------------------------------------------ */
/*  Translation dictionary                                             */
/* ------------------------------------------------------------------ */
const translations = {
  /* ---- General / Nav ---- */
  appTitle:            { en: 'SoundShield AI',            he: 'SoundShield AI' },
  appSubtitle:         { en: 'Kindergarten Audio Analysis', he: '\u05E0\u05D9\u05EA\u05D5\u05D7 \u05E9\u05DE\u05E2 \u05D2\u05DF \u05D9\u05DC\u05D3\u05D9\u05DD' },
  dashboard:           { en: 'Dashboard',                  he: '\u05DC\u05D5\u05D7 \u05D1\u05E7\u05E8\u05D4' },
  history:             { en: 'History',                    he: '\u05D4\u05D9\u05E1\u05D8\u05D5\u05E8\u05D9\u05D4' },
  settings:            { en: 'Settings',                   he: '\u05D4\u05D2\u05D3\u05E8\u05D5\u05EA' },
  language:            { en: 'Language',                   he: '\u05E9\u05E4\u05D4' },
  darkMode:            { en: 'Dark Mode',                  he: '\u05DE\u05E6\u05D1 \u05DB\u05D4\u05D4' },
  lightMode:           { en: 'Light Mode',                 he: '\u05DE\u05E6\u05D1 \u05D1\u05D4\u05D9\u05E8' },

  /* ---- Upload ---- */
  uploadTitle:         { en: 'Upload Audio File',          he: '\u05D4\u05E2\u05DC\u05D0\u05EA \u05E7\u05D5\u05D1\u05E5 \u05E9\u05DE\u05E2' },
  uploadHint:          { en: 'Drag & drop an audio file or click to browse', he: '\u05D2\u05E8\u05D5\u05E8 \u05E7\u05D5\u05D1\u05E5 \u05E9\u05DE\u05E2 \u05DC\u05DB\u05D0\u05DF \u05D0\u05D5 \u05DC\u05D7\u05E5 \u05DC\u05D1\u05D7\u05D9\u05E8\u05D4' },
  uploadBtn:           { en: 'Upload & Analyze',           he: '\u05D4\u05E2\u05DC\u05D4 \u05D5\u05E0\u05EA\u05D7' },
  cancelBtn:           { en: 'Cancel',                     he: '\u05D1\u05D9\u05D8\u05D5\u05DC' },
  retryBtn:            { en: 'Retry',                      he: '\u05E0\u05E1\u05D4 \u05E9\u05D5\u05D1' },
  browseFiles:         { en: 'Browse Files',               he: '\u05E2\u05D9\u05D5\u05DF \u05D1\u05E7\u05D1\u05E6\u05D9\u05DD' },
  supportedFormats:    { en: 'Supported: WAV, MP3, M4A, FLAC, AAC, OGG (max 500 MB)', he: '\u05E0\u05EA\u05DE\u05DA: WAV, MP3, M4A, FLAC, AAC, OGG (\u05E2\u05D3 500 MB)' },
  fileTooBig:          { en: 'File exceeds 500 MB limit',  he: '\u05D4\u05E7\u05D5\u05D1\u05E5 \u05D7\u05D5\u05E8\u05D2 \u05DE\u05DE\u05D2\u05D1\u05DC\u05EA 500 MB' },
  fileBigWarning:      { en: 'Large file \u2014 upload may take a while', he: '\u05E7\u05D5\u05D1\u05E5 \u05D2\u05D3\u05D5\u05DC \u2014 \u05D4\u05D4\u05E2\u05DC\u05D0\u05D4 \u05E2\u05E9\u05D5\u05D9\u05D4 \u05DC\u05E7\u05D7\u05EA \u05D6\u05DE\u05DF' },
  unsupportedFormat:   { en: 'Unsupported file format',    he: '\u05E4\u05D5\u05E8\u05DE\u05D8 \u05E7\u05D5\u05D1\u05E5 \u05DC\u05D0 \u05E0\u05EA\u05DE\u05DA' },
  uploadingFile:       { en: 'Uploading file\u2026',           he: '\u05DE\u05E2\u05DC\u05D4 \u05E7\u05D5\u05D1\u05E5\u2026' },
  analyzingAudio:      { en: 'Analyzing audio\u2026',          he: '\u05DE\u05E0\u05EA\u05D7 \u05E9\u05DE\u05E2\u2026' },

  /* ---- Analysis steps ---- */
  step1:               { en: 'Uploading file',             he: '\u05DE\u05E2\u05DC\u05D4 \u05E7\u05D5\u05D1\u05E5' },
  step2:               { en: 'Pre-processing audio',       he: '\u05E2\u05D9\u05D1\u05D5\u05D3 \u05DE\u05E7\u05D3\u05D9\u05DD \u05E9\u05DC \u05E9\u05DE\u05E2' },
  step3:               { en: 'Detecting speech segments',  he: '\u05D6\u05D9\u05D4\u05D5\u05D9 \u05E7\u05D8\u05E2\u05D9 \u05D3\u05D9\u05D1\u05D5\u05E8' },
  step4:               { en: 'Transcribing audio',         he: '\u05DE\u05EA\u05DE\u05DC\u05DC \u05E9\u05DE\u05E2' },
  step5:               { en: 'Analyzing emotions',         he: '\u05DE\u05E0\u05EA\u05D7 \u05E8\u05D2\u05E9\u05D5\u05EA' },
  step6:               { en: 'Running safety classifiers', he: '\u05DE\u05E4\u05E2\u05D9\u05DC \u05DE\u05E1\u05D5\u05D5\u05D2\u05D9 \u05D1\u05D8\u05D9\u05D7\u05D5\u05EA' },
  step7:               { en: 'Generating report',          he: '\u05DE\u05D9\u05D9\u05E6\u05E8 \u05D3\u05D5\u05D7' },

  /* ---- Results ---- */
  resultsTitle:        { en: 'Analysis Results',           he: '\u05EA\u05D5\u05E6\u05D0\u05D5\u05EA \u05E0\u05D9\u05EA\u05D5\u05D7' },
  summary:             { en: 'Summary',                    he: '\u05E1\u05D9\u05DB\u05D5\u05DD' },
  findings:            { en: 'Findings',                   he: '\u05DE\u05DE\u05E6\u05D0\u05D9\u05DD' },
  recommendations:     { en: 'Recommendations',            he: '\u05D4\u05DE\u05DC\u05E6\u05D5\u05EA' },
  noFindings:          { en: 'No concerning findings',     he: '\u05DC\u05D0 \u05E0\u05DE\u05E6\u05D0\u05D5 \u05DE\u05DE\u05E6\u05D0\u05D9\u05DD \u05DE\u05D3\u05D0\u05D9\u05D2\u05D9\u05DD' },
  riskLevel:           { en: 'Risk Level',                 he: '\u05E8\u05DE\u05EA \u05E1\u05D9\u05DB\u05D5\u05DF' },
  totalIncidents:      { en: 'Total Incidents',            he: '\u05E1\u05D4\u05F4\u05DB \u05D0\u05D9\u05E8\u05D5\u05E2\u05D9\u05DD' },
  duration:            { en: 'Duration',                   he: '\u05DE\u05E9\u05DA' },
  analyzedAt:          { en: 'Analyzed At',                he: '\u05E0\u05D5\u05EA\u05D7 \u05D1\u05EA\u05D0\u05E8\u05D9\u05DA' },

  /* ---- Incident types ---- */
  violence:            { en: 'Violence',                   he: '\u05D0\u05DC\u05D9\u05DE\u05D5\u05EA' },
  emotion:             { en: 'Emotion',                    he: '\u05E8\u05D2\u05E9' },
  cry:                 { en: 'Cry Detection',              he: '\u05D6\u05D9\u05D4\u05D5\u05D9 \u05D1\u05DB\u05D9' },
  neglect:             { en: 'Neglect',                    he: '\u05D4\u05D6\u05E0\u05D7\u05D4' },
  languageAnalysis:    { en: 'Language',                   he: '\u05E9\u05E4\u05D4' },
  emotions:            { en: 'Emotions',                   he: '\u05E8\u05D2\u05E9\u05D5\u05EA' },
  cries:               { en: 'Cries',                      he: '\u05D1\u05DB\u05D9\u05D5\u05EA' },

  /* ---- Severity ---- */
  low:                 { en: 'Low',                        he: '\u05E0\u05DE\u05D5\u05DA' },
  medium:              { en: 'Medium',                     he: '\u05D1\u05D9\u05E0\u05D5\u05E0\u05D9' },
  high:                { en: 'High',                       he: '\u05D2\u05D1\u05D5\u05D4' },
  critical:            { en: 'Critical',                   he: '\u05E7\u05E8\u05D9\u05D8\u05D9' },

  /* ---- Waveform ---- */
  play:                { en: 'Play',                       he: '\u05E0\u05D2\u05DF' },
  pause:               { en: 'Pause',                      he: '\u05D4\u05E9\u05D4\u05D4' },
  stop:                { en: 'Stop',                       he: '\u05E2\u05E6\u05D5\u05E8' },

  /* ---- Report / Modal ---- */
  viewReport:          { en: 'View Report',                he: '\u05E6\u05E4\u05D4 \u05D1\u05D3\u05D5\u05D7' },
  downloadReport:      { en: 'Download Report',            he: '\u05D4\u05D5\u05E8\u05D3 \u05D3\u05D5\u05D7' },
  closeReport:         { en: 'Close',                      he: '\u05E1\u05D2\u05D5\u05E8' },
  reportTitle:         { en: 'Analysis Report',            he: '\u05D3\u05D5\u05D7 \u05E0\u05D9\u05EA\u05D5\u05D7' },
  fileInfo:            { en: 'File Information',           he: '\u05DE\u05D9\u05D3\u05E2 \u05E7\u05D5\u05D1\u05E5' },
  detailedFindings:    { en: 'Detailed Findings',          he: '\u05DE\u05DE\u05E6\u05D0\u05D9\u05DD \u05DE\u05E4\u05D5\u05E8\u05D8\u05D9\u05DD' },
  incidentAt:          { en: 'Incident at',                he: '\u05D0\u05D9\u05E8\u05D5\u05E2 \u05D1' },
  confidence:          { en: 'Confidence',                 he: '\u05E8\u05DE\u05EA \u05D1\u05D9\u05D8\u05D7\u05D5\u05DF' },
  listenClip:          { en: 'Listen to clip',             he: '\u05D4\u05D0\u05D6\u05E0\u05D4 \u05DC\u05E7\u05D8\u05E2' },

  /* ---- History table ---- */
  filename:            { en: 'Filename',                   he: '\u05E9\u05DD \u05E7\u05D5\u05D1\u05E5' },
  date:                { en: 'Date',                       he: '\u05EA\u05D0\u05E8\u05D9\u05DA' },
  risk:                { en: 'Risk',                       he: '\u05E1\u05D9\u05DB\u05D5\u05DF' },
  incidents:           { en: 'Incidents',                  he: '\u05D0\u05D9\u05E8\u05D5\u05E2\u05D9\u05DD' },
  actions:             { en: 'Actions',                    he: '\u05E4\u05E2\u05D5\u05DC\u05D5\u05EA' },
  noHistory:           { en: 'No analyses yet',            he: '\u05D0\u05D9\u05DF \u05E0\u05D9\u05EA\u05D5\u05D7\u05D9\u05DD \u05E2\u05D3\u05D9\u05D9\u05DF' },
  loadingHistory:      { en: 'Loading history\u2026',          he: '\u05D8\u05D5\u05E2\u05DF \u05D4\u05D9\u05E1\u05D8\u05D5\u05E8\u05D9\u05D4\u2026' },

  /* ---- Charts ---- */
  severityDistribution:{ en: 'Severity Distribution',      he: '\u05D4\u05EA\u05E4\u05DC\u05D2\u05D5\u05EA \u05D7\u05D5\u05DE\u05E8\u05D4' },
  incidentsByType:     { en: 'Incidents by Type',          he: '\u05D0\u05D9\u05E8\u05D5\u05E2\u05D9\u05DD \u05DC\u05E4\u05D9 \u05E1\u05D5\u05D2' },
  timelineDensity:     { en: 'Timeline Density',           he: '\u05E6\u05E4\u05D9\u05E4\u05D5\u05EA \u05E6\u05D9\u05E8 \u05D6\u05DE\u05DF' },
  emotionProfile:      { en: 'Emotion Profile',            he: '\u05E4\u05E8\u05D5\u05E4\u05D9\u05DC \u05E8\u05D2\u05E9\u05D9' },
  incidentsPerMinute:  { en: 'Incidents / minute',         he: '\u05D0\u05D9\u05E8\u05D5\u05E2\u05D9\u05DD / \u05D3\u05E7\u05D4' },
  timeMinutes:         { en: 'Time (minutes)',             he: '\u05D6\u05DE\u05DF (\u05D3\u05E7\u05D5\u05EA)' },

  /* ---- Misc ---- */
  error:               { en: 'Error',                      he: '\u05E9\u05D2\u05D9\u05D0\u05D4' },
  errorGeneric:        { en: 'Something went wrong',       he: '\u05DE\u05E9\u05D4\u05D5 \u05D4\u05E9\u05EA\u05D1\u05E9' },
  ok:                  { en: 'OK',                         he: '\u05D0\u05D9\u05E9\u05D5\u05E8' },
  loading:             { en: 'Loading\u2026',                  he: '\u05D8\u05D5\u05E2\u05DF\u2026' },
  newAnalysis:         { en: 'New Analysis',               he: '\u05E0\u05D9\u05EA\u05D5\u05D7 \u05D7\u05D3\u05E9' },
  seconds:             { en: 'seconds',                    he: '\u05E9\u05E0\u05D9\u05D5\u05EA' },
  minutes:             { en: 'minutes',                    he: '\u05D3\u05E7\u05D5\u05EA' },
  of:                  { en: 'of',                         he: '\u05DE\u05EA\u05D5\u05DA' },

  /* ---- Auth ---- */
  loginBtn:            { en: 'Sign In',              he: '\u05DB\u05E0\u05D9\u05E1\u05D4' },
  logoutBtn:           { en: 'Sign Out',             he: '\u05D4\u05EA\u05E0\u05EA\u05E7\u05D5\u05EA' },
  welcomeUser:         { en: 'Welcome',              he: '\u05E9\u05DC\u05D5\u05DD' },
  roleAdmin:           { en: 'Admin',                he: '\u05DE\u05E0\u05D4\u05DC' },
  roleAnalyst:         { en: 'Analyst',              he: '\u05DE\u05E0\u05EA\u05D7' },
  roleViewer:          { en: 'Viewer',               he: '\u05E6\u05D5\u05E4\u05D4' },
};

/* ------------------------------------------------------------------ */
/*  Auth helpers                                                       */
/* ------------------------------------------------------------------ */

/**
 * Get auth headers for API calls.
 * Returns object with Authorization header if token exists.
 */
function getAuthHeaders() {
    const token = localStorage.getItem('token');
    const headers = { 'Content-Type': 'application/json' };
    if (token) {
        headers['Authorization'] = 'Bearer ' + token;
    }
    return headers;
}

/* ------------------------------------------------------------------ */
/*  Alpine component: appState                                         */
/* ------------------------------------------------------------------ */
function appState() {
  return {
    /* ---- reactive state ---- */
    language: 'en',
    darkMode: false,
    results: null,
    isAnalyzing: false,
    currentFilename: '',
    history: [],
    historyLoading: false,
    activeView: 'upload',        // 'upload' | 'results' | 'history'
    error: null,

    /* ---- auth state ---- */
    authToken: localStorage.getItem('token') || '',
    currentUser: JSON.parse(localStorage.getItem('user') || 'null'),

    get isLoggedIn() { return !!this.authToken; },
    get userRole() { return this.currentUser?.role || 'viewer'; },
    get isAdmin() { return this.userRole === 'admin'; },

    logout() {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        sessionStorage.removeItem('token');
        sessionStorage.removeItem('user');
        window.location.href = '/login';
    },

    /* ---- audio clips ---- */
    audioClips: [],

    /* ---- translation helper ---- */
    t(key) {
      const entry = translations[key];
      if (!entry) return key;
      return entry[this.language] || entry.en || key;
    },

    /**
     * Find a matching audio clip for an incident by type and timestamp.
     * Returns the clip URL or null.
     */
    findClipUrl(incidentType, timestamp) {
      if (!this.audioClips || !this.audioClips.length) return null;
      // Parse start time from timestamp string like "10.0s - 15.0s"
      let startTime = 0;
      try {
        const parts = timestamp.split(' - ');
        startTime = parseFloat(parts[0].replace('s', ''));
      } catch (e) { return null; }

      for (const clip of this.audioClips) {
        if (clip.incident_type === incidentType) {
          if (Math.abs(clip.start_time - startTime) < 2.0) {
            return '/audio_clip/' + clip.filename;
          }
        }
      }
      return null;
    },

    /* ---- language ---- */
    setLanguage(lang) {
      this.language = lang;
      document.documentElement.lang = lang;
      document.documentElement.dir = lang === 'he' ? 'rtl' : 'ltr';
      localStorage.setItem('ss-lang', lang);
      // Notify other components
      document.dispatchEvent(new CustomEvent('language-changed', { detail: { language: lang } }));
    },

    toggleLanguage() {
      this.setLanguage(this.language === 'en' ? 'he' : 'en');
    },

    /* ---- dark mode ---- */
    setDarkMode(enabled) {
      this.darkMode = enabled;
      document.documentElement.classList.toggle('dark', enabled);
      localStorage.setItem('ss-dark-mode', JSON.stringify(enabled));
      // Notify charts / waveform that may need color updates
      document.dispatchEvent(new CustomEvent('dark-mode-changed', { detail: { darkMode: enabled } }));
    },

    toggleDarkMode() {
      this.setDarkMode(!this.darkMode);
    },

    /* ---- history ---- */
    async loadHistory() {
      this.historyLoading = true;
      try {
        const res = await fetch('/reports');
        if (!res.ok) throw new Error('Failed to load reports');
        const data = await res.json();
        this.history = Array.isArray(data) ? data : (data.reports || []);
      } catch (err) {
        console.error('[SoundShield] loadHistory error:', err);
        this.history = [];
      } finally {
        this.historyLoading = false;
      }
    },

    /* ---- navigation ---- */
    showView(view) {
      this.activeView = view;
      if (view === 'history') {
        this.loadHistory();
      }
    },

    /* ---- helpers ---- */
    formatDuration(seconds) {
      if (!seconds && seconds !== 0) return '--';
      const m = Math.floor(seconds / 60);
      const s = Math.floor(seconds % 60);
      return m > 0 ? `${m}m ${s}s` : `${s}s`;
    },

    formatDate(iso) {
      if (!iso) return '--';
      try {
        const d = new Date(iso);
        return d.toLocaleDateString(this.language === 'he' ? 'he-IL' : 'en-US', {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        });
      } catch (_) {
        return iso;
      }
    },

    severityClass(level) {
      const map = { low: 'low', medium: 'medium', high: 'high', critical: 'critical' };
      return map[(level || '').toLowerCase()] || 'low';
    },

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

    /* ---- init ---- */
    init() {
      // Restore language
      const savedLang = localStorage.getItem('ss-lang');
      if (savedLang === 'he' || savedLang === 'en') {
        this.setLanguage(savedLang);
      }

      // Restore dark mode
      const savedDark = localStorage.getItem('ss-dark-mode');
      if (savedDark !== null) {
        this.setDarkMode(JSON.parse(savedDark));
      } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        this.setDarkMode(true);
      }

      // Listen for system theme changes
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        // Only auto-switch if user hasn't manually set preference
        if (localStorage.getItem('ss-dark-mode') === null) {
          this.setDarkMode(e.matches);
        }
      });

      // Listen for analysis-complete events from upload.js
      document.addEventListener('analysis-complete', (e) => {
        this.results = e.detail;
        this.currentFilename = e.detail.filename || this.currentFilename;
        this.audioClips = e.detail.audio_clips || [];
        this.isAnalyzing = false;
        this.activeView = 'results';
      });

      // Listen for analysis-started
      document.addEventListener('analysis-started', (e) => {
        this.isAnalyzing = true;
        this.currentFilename = e.detail.filename || '';
        this.results = null;
        this.error = null;
      });

      // Listen for analysis-error
      document.addEventListener('analysis-error', (e) => {
        this.isAnalyzing = false;
        this.error = e.detail.message || this.t('errorGeneric');
      });

      // Load history in background
      this.loadHistory();
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Register with Alpine                                               */
/* ------------------------------------------------------------------ */
document.addEventListener('alpine:init', () => {
  Alpine.data('appState', appState);
});

/* Export for external use if needed */
window.SoundShield = window.SoundShield || {};
window.SoundShield.translations = translations;
window.SoundShield.appState = appState;
