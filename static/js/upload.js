/**
 * SoundShield-AI  --  File Upload & Analysis Progress (Alpine.js)
 *
 * Handles drag-drop upload, client-side validation, upload progress via
 * XMLHttpRequest, and SSE-based analysis progress tracking.
 */

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */
const ALLOWED_EXTENSIONS = ['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg'];
const MAX_FILE_SIZE = 500 * 1024 * 1024;   // 500 MB
const WARN_FILE_SIZE = 100 * 1024 * 1024;  // 100 MB
const TOTAL_ANALYSIS_STEPS = 7;

/* ------------------------------------------------------------------ */
/*  Alpine component: uploadManager                                    */
/* ------------------------------------------------------------------ */
function uploadManager() {
  return {
    /* ---- state ---- */
    file: null,
    isDragOver: false,
    isRejected: false,
    validationError: '',
    sizeWarning: '',

    // Upload HTTP progress
    isUploading: false,
    uploadPercent: 0,

    // Analysis SSE progress
    isAnalysisRunning: false,
    analysisStep: 0,
    analysisMessage: '',
    totalSteps: TOTAL_ANALYSIS_STEPS,
    analysisProgress: 0,

    // General
    hasError: false,
    errorMessage: '',
    canRetry: false,

    // Internal handles
    _xhr: null,
    _abortController: null,
    _eventSource: null,

    /* ---- helpers ---- */
    _getAppState() {
      // Walk up the Alpine scope chain to find appState data
      // In the template, uploadManager is usually nested inside appState
      // so we try $store or fall back to document-level state
      return this.$data && this.$data.language
        ? this.$data
        : (window.SoundShield && window.SoundShield._appInstance) || { language: 'en' };
    },

    _t(key) {
      const app = this._getAppState();
      if (app && typeof app.t === 'function') return app.t(key);
      const entry = window.SoundShield && window.SoundShield.translations && window.SoundShield.translations[key];
      if (entry) return entry[app.language || 'en'] || entry.en || key;
      return key;
    },

    _getExtension(filename) {
      const parts = (filename || '').split('.');
      return parts.length > 1 ? parts.pop().toLowerCase() : '';
    },

    _formatSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },

    /* ---- validation ---- */
    validateFile(file) {
      this.validationError = '';
      this.sizeWarning = '';
      this.isRejected = false;

      if (!file) return false;

      const ext = this._getExtension(file.name);
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        this.validationError = this._t('unsupportedFormat') +
          ` (.${ext}). ${this._t('supportedFormats')}`;
        this.isRejected = true;
        return false;
      }

      if (file.size > MAX_FILE_SIZE) {
        this.validationError = this._t('fileTooBig') +
          ` (${this._formatSize(file.size)})`;
        this.isRejected = true;
        return false;
      }

      if (file.size > WARN_FILE_SIZE) {
        this.sizeWarning = this._t('fileBigWarning') +
          ` (${this._formatSize(file.size)})`;
      }

      return true;
    },

    /* ---- drag & drop ---- */
    onDragOver(e) {
      e.preventDefault();
      e.stopPropagation();
      this.isDragOver = true;
      this.isRejected = false;
    },

    onDragLeave(e) {
      e.preventDefault();
      e.stopPropagation();
      this.isDragOver = false;
      this.isRejected = false;
    },

    onDrop(e) {
      e.preventDefault();
      e.stopPropagation();
      this.isDragOver = false;

      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length > 0) {
        this.selectFile(files[0]);
      }
    },

    onFileInput(e) {
      const files = e.target.files;
      if (files && files.length > 0) {
        this.selectFile(files[0]);
      }
    },

    selectFile(file) {
      this.resetState();
      if (this.validateFile(file)) {
        this.file = file;
      } else {
        this.file = null;
      }
    },

    removeFile() {
      this.file = null;
      this.validationError = '';
      this.sizeWarning = '';
      this.isRejected = false;
    },

    /* ---- upload ---- */
    async startUpload() {
      if (!this.file) return;
      if (this.isUploading || this.isAnalysisRunning) return;

      this.resetProgress();
      this.isUploading = true;
      this.hasError = false;
      this.errorMessage = '';
      this.canRetry = false;

      // Notify app that analysis has started
      document.dispatchEvent(new CustomEvent('analysis-started', {
        detail: { filename: this.file.name }
      }));

      const formData = new FormData();
      formData.append('file', this.file);

      // Include language from parent state
      const app = this._getAppState();
      if (app && app.language) {
        formData.append('language', app.language);
      }

      // Use XMLHttpRequest for upload progress tracking
      const xhr = new XMLHttpRequest();
      this._xhr = xhr;

      const uploadPromise = new Promise((resolve, reject) => {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            this.uploadPercent = Math.round((e.loaded / e.total) * 100);
          }
        });

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            let data;
            try {
              data = JSON.parse(xhr.responseText);
            } catch (_) {
              data = { filename: this.file.name };
            }
            resolve(data);
          } else {
            let errMsg = this._t('errorGeneric');
            try {
              const errData = JSON.parse(xhr.responseText);
              errMsg = errData.error || errData.message || errMsg;
            } catch (_) { /* ignore */ }
            reject(new Error(errMsg));
          }
        });

        xhr.addEventListener('error', () => {
          reject(new Error(this._t('errorGeneric')));
        });

        xhr.addEventListener('abort', () => {
          reject(new Error('Upload cancelled'));
        });

        xhr.open('POST', '/upload');
        xhr.send(formData);
      });

      try {
        const uploadResult = await uploadPromise;
        this.isUploading = false;
        this.uploadPercent = 100;

        // If server returns results immediately (no SSE needed)
        if (uploadResult && uploadResult.results) {
          this._handleComplete(uploadResult.results);
          return;
        }

        // Otherwise connect SSE for analysis progress
        const filename = uploadResult.filename || this.file.name;
        this._connectSSE(filename);
      } catch (err) {
        this.isUploading = false;
        if (err.message !== 'Upload cancelled') {
          this._handleError(err.message);
        }
      }
    },

    /* ---- SSE progress stream ---- */
    _connectSSE(filename) {
      this.isAnalysisRunning = true;
      this.analysisStep = 0;
      this.analysisMessage = this._t('step1');

      // Clean up any existing connection
      if (this._eventSource) {
        this._eventSource.close();
      }

      const encodedFilename = encodeURIComponent(filename);
      const es = new EventSource(`/progress-stream/${encodedFilename}`);
      this._eventSource = es;

      es.addEventListener('progress', (e) => {
        try {
          const data = JSON.parse(e.data);
          this.analysisStep = data.step || this.analysisStep;
          this.analysisMessage = data.message || '';
          this.analysisProgress = data.progress || Math.round((this.analysisStep / this.totalSteps) * 100);
        } catch (err) {
          console.warn('[SoundShield] SSE progress parse error:', err);
        }
      });

      es.addEventListener('complete', (e) => {
        try {
          const data = JSON.parse(e.data);
          es.close();
          this._eventSource = null;
          this.isAnalysisRunning = false;
          this.analysisStep = this.totalSteps;
          this.analysisProgress = 100;
          this._handleComplete(data);
        } catch (err) {
          console.error('[SoundShield] SSE complete parse error:', err);
          this._handleError(this._t('errorGeneric'));
        }
      });

      es.addEventListener('error', (e) => {
        // Check if it's an SSE-sent error event with data
        if (e.data) {
          try {
            const data = JSON.parse(e.data);
            es.close();
            this._eventSource = null;
            this.isAnalysisRunning = false;
            this._handleError(data.message || data.error || this._t('errorGeneric'));
            return;
          } catch (_) { /* fall through */ }
        }

        // Connection error
        if (es.readyState === EventSource.CLOSED) {
          // Server closed connection without sending complete - may be an error
          this._eventSource = null;
          this.isAnalysisRunning = false;
          if (!this.hasError && this.analysisProgress < 100) {
            this._handleError(this._t('errorGeneric'));
          }
        }
        // If CONNECTING, EventSource will auto-retry
      });

      // Generic onerror for connection failures
      es.onerror = (e) => {
        if (es.readyState === EventSource.CLOSED) {
          this._eventSource = null;
          this.isAnalysisRunning = false;
          if (!this.hasError && this.analysisProgress < 100) {
            this._handleError(this._t('errorGeneric'));
          }
        }
      };
    },

    /* ---- result handling ---- */
    _handleComplete(data) {
      this.isAnalysisRunning = false;
      this.isUploading = false;
      this.analysisProgress = 100;
      this.analysisStep = this.totalSteps;

      // Dispatch custom event for app.js and other components
      document.dispatchEvent(new CustomEvent('analysis-complete', {
        detail: data
      }));
    },

    _handleError(message) {
      this.hasError = true;
      this.errorMessage = message || this._t('errorGeneric');
      this.canRetry = true;
      this.isUploading = false;
      this.isAnalysisRunning = false;

      document.dispatchEvent(new CustomEvent('analysis-error', {
        detail: { message: this.errorMessage }
      }));
    },

    /* ---- cancel ---- */
    cancelUpload() {
      // Abort XHR upload
      if (this._xhr) {
        this._xhr.abort();
        this._xhr = null;
      }

      // Close SSE
      if (this._eventSource) {
        this._eventSource.close();
        this._eventSource = null;
      }

      this.isUploading = false;
      this.isAnalysisRunning = false;
      this.uploadPercent = 0;
      this.analysisStep = 0;
      this.analysisProgress = 0;
      this.analysisMessage = '';
    },

    /* ---- retry ---- */
    retry() {
      this.hasError = false;
      this.errorMessage = '';
      this.canRetry = false;
      if (this.file) {
        this.startUpload();
      }
    },

    /* ---- state reset ---- */
    resetProgress() {
      this.uploadPercent = 0;
      this.analysisStep = 0;
      this.analysisMessage = '';
      this.analysisProgress = 0;
      this.hasError = false;
      this.errorMessage = '';
      this.canRetry = false;
    },

    resetState() {
      this.cancelUpload();
      this.resetProgress();
      this.validationError = '';
      this.sizeWarning = '';
      this.isRejected = false;
    },

    /* ---- computed-like getters ---- */
    get isActive() {
      return this.isUploading || this.isAnalysisRunning;
    },

    get overallProgress() {
      if (this.isUploading && !this.isAnalysisRunning) {
        // Upload is ~15% of overall
        return Math.round(this.uploadPercent * 0.15);
      }
      if (this.isAnalysisRunning) {
        // Upload done (15%) + analysis progress (85%)
        return Math.round(15 + this.analysisProgress * 0.85);
      }
      if (this.analysisProgress >= 100) return 100;
      return 0;
    },

    get currentStepLabel() {
      if (!this.isActive) return '';
      if (this.isUploading && !this.isAnalysisRunning) {
        return this._t('uploadingFile');
      }
      return this.analysisMessage || this._t('analyzingAudio');
    },

    /* ---- cleanup ---- */
    destroy() {
      this.cancelUpload();
    },

    /* ---- init ---- */
    init() {
      // Nothing special needed on init - state is ready
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Register with Alpine                                               */
/* ------------------------------------------------------------------ */
document.addEventListener('alpine:init', () => {
  Alpine.data('uploadManager', uploadManager);
});

/* Export */
window.SoundShield = window.SoundShield || {};
window.SoundShield.uploadManager = uploadManager;
