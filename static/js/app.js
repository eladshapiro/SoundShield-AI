/* SoundShield dashboard — single front-end module.
 *
 * Registers Alpine stores before Alpine boots (this file is loaded without
 * `defer`; Alpine is deferred).  Stores:
 *   ui        language, theme, formatting helpers
 *   health    /health poll (models loaded?)
 *   toasts    transient messages
 *   upload    file validation → /upload-async → SSE progress → /job-status
 *   analysis  the normalised view-model rendered by the template
 *   history   /reports list (file-based)
 *   player    wavesurfer + incident markers
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /*  i18n                                                               */
  /* ------------------------------------------------------------------ */
  const STRINGS = {
    en: {
      skipToContent: 'Skip to content',
      tagline: 'Kindergarten recording analysis',
      uiLanguage: 'Interface language',
      appearance: 'Light or dark appearance',
      admin: 'Admin',
      serverReady: 'Ready',
      serverOffline: 'Server unreachable',
      saveClip: 'Save clip',
      newAnalysis: 'New analysis',
      dropTitle: 'Drop a recording here',
      dropSub: 'or click to choose a file · WAV, MP3, M4A, FLAC, AAC, OGG · up to 500 MB',
      dropSubShort: 'or click to choose · up to 500 MB',
      recordingLanguage: 'Language spoken in the recording',
      hebrew: 'Hebrew',
      english: 'English',
      analyze: 'Analyze recording',
      cancel: 'Cancel',
      remove: 'Remove file',
      dismiss: 'Dismiss',
      tryAgain: 'Try again',
      recent: 'Recent analyses',
      noRecent: 'Analyses you run will appear here.',
      refresh: 'Refresh list',
      emptyTitle: 'Check a recording for signs of harm',
      emptyLead: 'Upload a recording from the kindergarten. SoundShield listens for crying that goes unanswered, shouting, angry speech and inappropriate language, then shows you the exact moments so you can hear them yourself.',
      hint1t: 'Stays on this computer',
      hint1: 'The recording is analyzed locally and never sent elsewhere.',
      hint2t: 'Hebrew and English',
      hint2: 'Speech is transcribed in the language you pick.',
      hint3t: 'You make the call',
      hint3: 'Every flagged moment comes with its audio.',
      uploading: 'Uploading',
      analyzing: 'Analyzing',
      queued: 'Waiting for the models',
      step1: 'Reading the audio',
      step2: 'Listening for tone of voice',
      step3: 'Listening for crying',
      step4: 'Listening for shouting and impacts',
      step5: 'Checking responses to distress',
      step6: 'Transcribing speech',
      step7: 'Checking for inappropriate language',
      analysisFailed: 'The analysis could not be completed',
      cancelledNote: 'The analysis keeps running in the background and will appear under Recent analyses.',
      done: 'Analysis complete',
      loadingReport: 'Loading report',
      reportMissing: 'That report could not be loaded.',
      risk_none: 'Low concern',
      risk_low: 'Low concern',
      risk_medium: 'Moderate concern',
      risk_high: 'High concern',
      risk_critical: 'Critical',
      verdictZero: 'Nothing to review in this {d} recording.',
      verdictOne: 'One moment to review in this {d} recording.',
      verdictMany: '{n} moments to review in this {d} recording.',
      verdictCritical: '{c} of them critical.',
      moments: 'Moments',
      criticalCount: 'Critical',
      duration: 'Duration',
      scoreLabel: 'Weighted severity score',
      zoneNormal: 'Normal',
      zoneModerate: 'Moderate',
      zoneConcerning: 'Concerning',
      zoneCritical: 'Critical',
      recording: 'Recording',
      play: 'Play',
      pause: 'Pause',
      prevMoment: 'Previous moment',
      nextMoment: 'Next moment',
      audioMissing: 'The audio file is no longer available; the report is shown without playback.',
      sev_low: 'Low',
      sev_medium: 'Medium',
      sev_high: 'High',
      sev_critical: 'Critical',
      findings: 'Moments to review',
      noFindings: 'No concerning moments were detected in this recording.',
      listen: 'Listen',
      confidence: 'confidence',
      mlTag: 'Neural model',
      type_emotion: 'Tone of voice',
      type_violence: 'Violence',
      type_cry: 'Crying',
      type_neglect: 'Unanswered crying',
      type_language: 'Language',
      label_anger: 'Angry speech',
      label_aggression: 'Aggressive tone',
      label_shouting: 'Shouting',
      label_aggressive_tone: 'Aggressive tone',
      label_potential_physical_violence: 'Possible impact or slap',
      label_response: 'Crying, an adult responded',
      label_no_response: 'Crying with no response',
      label_unanswered_cry: 'Crying left unanswered',
      label_cry: 'Crying',
      transcript: 'Transcript',
      transcriptEmpty: 'No speech was transcribed.',
      transcriptFailed: 'Transcription was not available for this recording.',
      flagged: 'flagged',
      signals: 'Voice signals',
      noSignals: 'The neural emotion models were not available for this recording.',
      emo_arousal: 'Arousal',
      emo_dominance: 'Dominance',
      emo_valence: 'Positivity',
      emo_anger: 'Anger',
      emo_calm: 'Calm',
      emo_stress: 'Stress',
      speakers: 'Speakers',
      adultSpeech: 'Adult speech',
      childSpeech: 'Child speech',
      keyFindings: 'Key findings',
      recommendations: 'Recommendations',
      analyzedOn: 'Analyzed',
      errUnsupported: 'This file type is not supported. Use WAV, MP3, M4A, FLAC, AAC or OGG.',
      errTooBig: 'This file is larger than 500 MB.',
      warnBig: 'Large file: the analysis may take several minutes.',
      errNetwork: 'The server could not be reached. Check that SoundShield is running.',
      errGeneric: 'Something went wrong.',
    },
    he: {
      skipToContent: 'דילוג לתוכן',
      tagline: 'ניתוח הקלטות מגן הילדים',
      uiLanguage: 'שפת הממשק',
      appearance: 'מראה בהיר או כהה',
      admin: 'ניהול',
      serverReady: 'מוכן',
      serverOffline: 'אין חיבור לשרת',
      saveClip: 'שמירת הקטע',
      newAnalysis: 'ניתוח חדש',
      dropTitle: 'גררו לכאן הקלטה',
      dropSub: 'או לחצו לבחירת קובץ · WAV, MP3, M4A, FLAC, AAC, OGG · עד 500MB',
      dropSubShort: 'או לחצו לבחירה · עד 500MB',
      recordingLanguage: 'השפה המדוברת בהקלטה',
      hebrew: 'עברית',
      english: 'אנגלית',
      analyze: 'נתחו את ההקלטה',
      cancel: 'ביטול',
      remove: 'הסרת הקובץ',
      dismiss: 'סגירה',
      tryAgain: 'ניסיון נוסף',
      recent: 'ניתוחים אחרונים',
      noRecent: 'ניתוחים שתריצו יופיעו כאן.',
      refresh: 'רענון הרשימה',
      emptyTitle: 'בדקו הקלטה לאיתור סימני פגיעה',
      emptyLead: 'העלו הקלטה מהגן. SoundShield מאתר בכי שלא נענה, צעקות, דיבור כועס ושפה לא הולמת, ומציג את הרגעים המדויקים כדי שתשמעו אותם בעצמכם.',
      hint1t: 'נשאר במחשב הזה',
      hint1: 'ההקלטה מנותחת מקומית ואינה נשלחת לשום מקום.',
      hint2t: 'עברית ואנגלית',
      hint2: 'הדיבור מתומלל בשפה שתבחרו.',
      hint3t: 'ההחלטה שלכם',
      hint3: 'לכל רגע שסומן מצורף הקטע המוקלט.',
      uploading: 'מעלה',
      analyzing: 'מנתח',
      queued: 'ממתין למודלים',
      step1: 'קריאת האודיו',
      step2: 'האזנה לטון הדיבור',
      step3: 'האזנה לבכי',
      step4: 'האזנה לצעקות ולחבטות',
      step5: 'בדיקת התגובה למצוקה',
      step6: 'תמלול הדיבור',
      step7: 'בדיקת שפה לא הולמת',
      analysisFailed: 'לא ניתן היה להשלים את הניתוח',
      cancelledNote: 'הניתוח ממשיך לרוץ ברקע ויופיע תחת ניתוחים אחרונים.',
      done: 'הניתוח הושלם',
      loadingReport: 'טוען דוח',
      reportMissing: 'לא ניתן לטעון את הדוח.',
      risk_none: 'חשש נמוך',
      risk_low: 'חשש נמוך',
      risk_medium: 'חשש בינוני',
      risk_high: 'חשש גבוה',
      risk_critical: 'קריטי',
      verdictZero: 'לא נמצא מה לבדוק בהקלטה של {d}.',
      verdictOne: 'רגע אחד לבדיקה בהקלטה של {d}.',
      verdictMany: '{n} רגעים לבדיקה בהקלטה של {d}.',
      verdictCritical: '{c} מהם קריטיים.',
      moments: 'רגעים',
      criticalCount: 'קריטיים',
      duration: 'משך',
      scoreLabel: 'ציון חומרה משוקלל',
      zoneNormal: 'תקין',
      zoneModerate: 'בינוני',
      zoneConcerning: 'מדאיג',
      zoneCritical: 'קריטי',
      recording: 'ההקלטה',
      play: 'נגן',
      pause: 'השהה',
      prevMoment: 'הרגע הקודם',
      nextMoment: 'הרגע הבא',
      audioMissing: 'קובץ האודיו כבר אינו זמין; הדוח מוצג ללא נגינה.',
      sev_low: 'נמוך',
      sev_medium: 'בינוני',
      sev_high: 'גבוה',
      sev_critical: 'קריטי',
      findings: 'רגעים לבדיקה',
      noFindings: 'לא זוהו רגעים מדאיגים בהקלטה הזו.',
      listen: 'האזנה',
      confidence: 'ביטחון',
      mlTag: 'מודל נוירוני',
      type_emotion: 'טון דיבור',
      type_violence: 'אלימות',
      type_cry: 'בכי',
      type_neglect: 'בכי ללא מענה',
      type_language: 'שפה',
      label_anger: 'דיבור כועס',
      label_aggression: 'טון תוקפני',
      label_shouting: 'צעקות',
      label_aggressive_tone: 'טון תוקפני',
      label_potential_physical_violence: 'חבטה או מכה אפשרית',
      label_response: 'בכי, מבוגר הגיב',
      label_no_response: 'בכי ללא תגובה',
      label_unanswered_cry: 'בכי שלא נענה',
      label_cry: 'בכי',
      transcript: 'תמלול',
      transcriptEmpty: 'לא תומלל דיבור.',
      transcriptFailed: 'התמלול לא היה זמין להקלטה הזו.',
      flagged: 'סומן',
      signals: 'אותות קוליים',
      noSignals: 'מודלי הרגש הנוירוניים לא היו זמינים להקלטה הזו.',
      emo_arousal: 'עוררות',
      emo_dominance: 'דומיננטיות',
      emo_valence: 'חיוביות',
      emo_anger: 'כעס',
      emo_calm: 'רוגע',
      emo_stress: 'לחץ',
      speakers: 'דוברים',
      adultSpeech: 'דיבור מבוגרים',
      childSpeech: 'דיבור ילדים',
      keyFindings: 'ממצאים עיקריים',
      recommendations: 'המלצות',
      analyzedOn: 'נותח',
      errUnsupported: 'סוג הקובץ אינו נתמך. השתמשו ב-WAV, MP3, M4A, FLAC, AAC או OGG.',
      errTooBig: 'הקובץ גדול מ-500MB.',
      warnBig: 'קובץ גדול: הניתוח עשוי להימשך כמה דקות.',
      errNetwork: 'אין חיבור לשרת. ודאו ש-SoundShield פועל.',
      errGeneric: 'משהו השתבש.',
    },
  };

  /* Hebrew labels the report generator writes into detailed_findings → raw keys */
  const HE_LABELS = {
    'כעס': 'anger', 'תוקפנות': 'aggression', 'צעקות': 'shouting', 'טון תוקפני': 'aggressive_tone',
    'אלימות פיזית אפשרית': 'potential_physical_violence', 'בכי': 'cry',
  };
  const HE_SEVERITY = { 'נמוכה': 'low', 'בינונית': 'medium', 'גבוהה': 'high', 'קריטית': 'critical', 'נמוך': 'low', 'בינוני': 'medium', 'גבוה': 'high', 'קריטי': 'critical' };

  const ALLOWED = ['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg'];
  const MAX_BYTES = 500 * 1024 * 1024;
  const WARN_BYTES = 100 * 1024 * 1024;
  const TOTAL_STEPS = 7;

  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const num = (x, d = 0) => { const n = parseFloat(x); return Number.isFinite(n) ? n : d; };
  const sevClass = (s) => {
    const k = String(s || '').toLowerCase().trim();
    if (HE_SEVERITY[k]) return HE_SEVERITY[k];
    if (k === 'minimal' || k === 'none') return 'low';
    if (k === 'moderate') return 'medium';
    if (k === 'severe') return 'critical';
    return ['low', 'medium', 'high', 'critical'].includes(k) ? k : 'low';
  };
  const stripStamp = (name) => String(name || '').replace(/^\d{9,11}_/, '');
  const parseStamp = (ts) => {
    // "12.5s - 17.5s" → [12.5, 17.5]
    const m = String(ts || '').match(/([\d.]+)\s*s?\s*[-–]\s*([\d.]+)/);
    if (m) return [num(m[1]), num(m[2])];
    const one = String(ts || '').match(/([\d.]+)/);
    return one ? [num(one[1]), num(one[1]) + 2] : [0, 0];
  };

  /* ------------------------------------------------------------------ */
  /*  Normalisation → view-model                                         */
  /* ------------------------------------------------------------------ */
  function normalize(src, kind, extra) {
    extra = extra || {};
    const results = kind === 'payload' ? (src.results || {}) : src;
    const meta = results.metadata || {};
    const summary = results.summary || {};
    const findings = results.detailed_findings || {};
    const adv = findings.advanced_analysis || {};
    const serverFile = kind === 'payload' ? src.filename : (meta.file_name || (meta.file_path || '').split(/[\\/]/).pop());
    const duration = num(kind === 'payload' ? (src.duration || meta.audio_duration) : meta.audio_duration, 0)
      || num((results.statistics || {}).audio_duration_minutes, 0) * 60;
    const clips = (kind === 'payload' ? src.audio_clips : src.audio_clips) || [];

    /* incidents */
    let incidents = [];
    if (kind === 'payload' && Array.isArray(src.incidents)) {
      incidents = src.incidents.map((inc) => ({
        type: inc.type || 'emotion',
        label: inc.label || '',
        start: num(inc.start_time), end: num(inc.end_time),
        sev: sevClass(inc.severity),
        confidence: inc.confidence != null ? num(inc.confidence) : null,
        mlBacked: inc.ml_backed === true ? true : (inc.ml_backed === false ? false : null),
        description: '',
      }));
      // descriptions from the report's detailed findings (matched by time)
      const described = [];
      [['emotion', findings.emotional_analysis], ['violence', findings.violence_analysis], ['cry', findings.cry_analysis],
       ['neglect', findings.neglect_analysis], ['language', findings.inappropriate_language]].forEach(([type, list]) => {
        (list || []).forEach((f) => { const [s] = parseStamp(f.timestamp); described.push({ type, start: s, description: f.description || '' }); });
      });
      incidents.forEach((inc) => {
        const d = described.find((x) => x.type === inc.type && Math.abs(x.start - inc.start) < 1.0);
        if (d) inc.description = d.description;
      });
    } else {
      [['emotion', findings.emotional_analysis], ['violence', findings.violence_analysis], ['cry', findings.cry_analysis],
       ['neglect', findings.neglect_analysis], ['language', findings.inappropriate_language]].forEach(([type, list]) => {
        (list || []).forEach((f) => {
          const [s, e] = parseStamp(f.timestamp);
          const rawLabel = f.detected_emotion || f.violence_types || f.word || '';
          incidents.push({
            type, start: s, end: e,
            label: HE_LABELS[String(rawLabel).trim()] || rawLabel,
            sev: sevClass(f.severity),
            confidence: f.confidence != null && f.confidence !== '' ? num(f.confidence) : null,
            mlBacked: null,
            description: f.description || '',
          });
        });
      });
    }
    incidents.sort((a, b) => a.start - b.start);
    incidents.forEach((inc) => {
      inc.leftPct = duration > 0 ? clamp(inc.start / duration * 100, 0, 100) : 0;
      inc.widthPct = duration > 0 ? clamp((inc.end - inc.start) / duration * 100, 0.6, 100 - inc.leftPct) : 1;
      const clip = clips.find((c) => c.incident_type === inc.type && Math.abs(num(c.start_time) - inc.start) < 2.0);
      inc.clipUrl = clip ? '/audio_clip/' + encodeURIComponent(clip.filename) : null;
    });

    /* transcript */
    const w = adv.whisper_transcription || {};
    const flaggedStarts = (w.concerning_segments || []).map((c) => num(c.start_time));
    const segments = (w.segments || []).map((s) => ({
      start: num(s.start), end: num(s.end), text: String(s.text || '').trim(),
      flagged: flaggedStarts.some((f) => Math.abs(f - num(s.start)) < 0.05),
    })).filter((s) => s.text);

    /* emotion summary */
    const top = (adv.emotion_detection || {}).top_emotions || null;
    const emotion = top && Object.keys(top).length ? top : null;

    /* speakers */
    const dia = (kind === 'payload' ? src.diarization : null) || null;
    const diaSum = meta.diarization_summary || null;
    let speakers = null;
    if (dia && dia.speaker_count != null) {
      speakers = { count: dia.speaker_count, adultTime: num(dia.total_adult_time), childTime: num(dia.total_child_time) };
    } else if (diaSum && diaSum.speaker_count != null) {
      speakers = { count: diaSum.speaker_count, adultTime: null, childTime: null };
    }

    const score = num(summary.weighted_severity_score, 0);
    const riskClass = sevClass(summary.risk_level || 'low');
    return {
      serverFile,
      name: stripStamp(serverFile) || serverFile || '',
      audioUrl: serverFile ? '/uploaded-audio/' + encodeURIComponent(serverFile) : null,
      duration,
      language: extra.language || null,
      analyzedAt: meta.analysis_timestamp || '',
      models: (kind === 'payload' ? src.models_used : meta.models_used) || [],
      summary,
      score,
      scorePct: clamp(score / 4 * 100, 0, 100),
      riskClass,
      incidents,
      transcript: { segments, model: w.model || '', error: w.error ? String(w.error) : null },
      emotion,
      speakers,
      keyFindings: summary.key_findings || [],
      recommendations: results.recommendations || [],
    };
  }

  /* ------------------------------------------------------------------ */
  /*  Stores                                                             */
  /* ------------------------------------------------------------------ */
  document.addEventListener('alpine:init', () => {

    Alpine.magic('t', () => (key, vars) => Alpine.store('ui').t(key, vars));

    /* ---- ui ---- */
    Alpine.store('ui', {
      lang: document.documentElement.lang === 'he' ? 'he' : 'en',
      dark: document.documentElement.classList.contains('dark'),
      t(key, vars) {
        const table = STRINGS[this.lang] || STRINGS.en;
        let s = table[key] != null ? table[key] : (STRINGS.en[key] != null ? STRINGS.en[key] : key);
        if (vars) Object.keys(vars).forEach((k) => { s = s.replace('{' + k + '}', vars[k]); });
        return s;
      },
      setLang(lang) {
        this.lang = lang;
        document.documentElement.lang = lang;
        document.documentElement.dir = lang === 'he' ? 'rtl' : 'ltr';
        document.title = 'SoundShield';
        try { localStorage.setItem('ss-lang', lang); } catch (e) { /* ignore */ }
      },
      setDark(on) {
        this.dark = !!on;
        document.documentElement.classList.toggle('dark', this.dark);
        try { localStorage.setItem('ss-dark-mode', String(this.dark)); } catch (e) { /* ignore */ }
        Alpine.store('player').applyTheme();
      },
      toggleDark() { this.setDark(!this.dark); },
      locale() { return this.lang === 'he' ? 'he-IL' : 'en-US'; },
      fmtDate(iso) {
        if (!iso) return '';
        const d = new Date(iso);
        if (isNaN(d)) return String(iso);
        return d.toLocaleString(this.locale(), { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
      },
      fmtClock(sec) {
        if (sec == null || !Number.isFinite(sec)) return '–';
        const s = Math.max(0, Math.round(sec));
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
        const mm = h ? String(m).padStart(2, '0') : String(m);
        return (h ? h + ':' : '') + mm + ':' + String(r).padStart(2, '0');
      },
      fmtRange(a, b) { return this.fmtClock(a) + ' – ' + this.fmtClock(b); },
      fmtDuration(sec) {
        const s = Math.round(num(sec));
        if (s < 60) {
          if (this.lang === 'he') return s === 1 ? 'שנייה אחת' : s + ' שניות';
          return s + ' s';
        }
        const m = Math.round(s / 60);
        if (this.lang === 'he') return m === 1 ? 'דקה אחת' : (m === 2 ? 'שתי דקות' : m + ' דקות');
        return m + ' min';
      },
    });

    /* ---- health ---- */
    Alpine.store('health', {
      checked: false, online: false, _timer: null,
      async check() {
        try {
          const r = await fetch('/health', { cache: 'no-store' });
          const d = await r.json();
          this.online = r.ok && d.status !== 'unhealthy';
        } catch (e) { this.online = false; }
        this.checked = true;
      },
      start() { this.check(); this._timer = setInterval(() => this.check(), 60000); },
    });

    /* ---- toasts ---- */
    Alpine.store('toasts', {
      items: [], _id: 0,
      show(message, type) {
        const id = ++this._id;
        this.items.push({ id, message, type: type || 'info' });
        setTimeout(() => { this.items = this.items.filter((t) => t.id !== id); }, type === 'error' ? 7000 : 4000);
      },
    });

    /* ---- upload ---- */
    Alpine.store('upload', {
      file: null, error: '', warning: '', isOver: false,
      phase: 'idle',            // idle | uploading | analyzing | error
      uploadPct: 0, step: 0, stepMsg: '', jobId: null, failure: '',
      recordingLang: (function () { try { return localStorage.getItem('ss-rec-lang') || 'he'; } catch (e) { return 'he'; } })(),
      _xhr: null, _es: null, _poll: null, _cancelled: false,

      get busy() { return this.phase === 'uploading' || this.phase === 'analyzing'; },
      get failed() { return this.phase === 'error'; },

      t(k, v) { return Alpine.store('ui').t(k, v); },
      fileMeta() {
        if (!this.file) return '';
        const b = this.file.size;
        const size = b < 1048576 ? (b / 1024).toFixed(0) + ' KB' : (b / 1048576).toFixed(1) + ' MB';
        return size;
      },
      pick(file) {
        if (!file) return;
        this.error = ''; this.warning = '';
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (!ALLOWED.includes(ext)) { this.file = file; this.error = this.t('errUnsupported'); return; }
        if (file.size > MAX_BYTES) { this.file = file; this.error = this.t('errTooBig'); return; }
        if (file.size > WARN_BYTES) this.warning = this.t('warnBig');
        this.file = file;
        if (this.phase === 'error') this.phase = 'idle';
      },
      drop(ev) {
        this.isOver = false;
        const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
        if (f) this.pick(f);
      },
      clear() { if (this.busy) return; this.file = null; this.error = ''; this.warning = ''; },
      reset() { this.phase = 'idle'; this.failure = ''; this.step = 0; this.uploadPct = 0; },
      phaseLabel() {
        if (this.phase === 'uploading') return this.t('uploading') + '…';
        if (this.phase === 'analyzing') return this.step > 0 ? this.t('step' + Math.min(this.step, TOTAL_STEPS)) : this.t('queued') + '…';
        return '';
      },
      percent() {
        if (this.phase === 'uploading') return Math.round(this.uploadPct * 0.15);
        if (this.phase === 'analyzing') return 15 + Math.round(clamp(this.step, 0, TOTAL_STEPS) / TOTAL_STEPS * 85);
        return 0;
      },
      percentLabel() { return this.percent() + '%'; },
      indeterminate() { return this.phase === 'analyzing' && this.step === 0; },

      start() {
        if (!this.file || this.error || this.busy) return;
        try { localStorage.setItem('ss-rec-lang', this.recordingLang); } catch (e) { /* ignore */ }
        this._cancelled = false;
        this.failure = ''; this.step = 0; this.uploadPct = 0; this.jobId = null;
        this.phase = 'uploading';

        const fd = new FormData();
        fd.append('file', this.file);
        fd.append('language', this.recordingLang);
        const xhr = new XMLHttpRequest();
        this._xhr = xhr;
        xhr.upload.addEventListener('progress', (e) => { if (e.lengthComputable) this.uploadPct = e.loaded / e.total * 100; });
        xhr.addEventListener('load', () => {
          this._xhr = null;
          let data = null;
          try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
          if (xhr.status >= 200 && xhr.status < 300 && data && data.job_id) {
            this.phase = 'analyzing';
            this.jobId = data.job_id;
            this._watch(data.job_id);
          } else {
            this._fail((data && (data.error || data.message)) || this.t('errGeneric'));
          }
        });
        xhr.addEventListener('error', () => { this._xhr = null; this._fail(this.t('errNetwork')); });
        xhr.addEventListener('abort', () => { this._xhr = null; });
        const token = (function () { try { return localStorage.getItem('token'); } catch (e) { return null; } })();
        xhr.open('POST', '/upload-async');
        if (token) xhr.setRequestHeader('Authorization', 'Bearer ' + token);
        xhr.send(fd);
      },

      _watch(jobId) {
        this._closeStreams();
        const onEvent = (raw) => {
          let d; try { d = JSON.parse(raw); } catch (e) { return; }
          if (d.step != null) this.step = num(d.step, this.step);
          this.stepMsg = d.message || '';
          if (d.status === 'completed') this._finish(jobId);
          if (d.status === 'error') this._fail(d.message || this.t('errGeneric'));
        };
        try {
          const es = new EventSource('/progress-stream/' + encodeURIComponent(jobId));
          this._es = es;
          es.addEventListener('progress', (e) => onEvent(e.data));
          es.addEventListener('complete', (e) => onEvent(e.data));
          es.addEventListener('error', (e) => { if (e.data) onEvent(e.data); });
        } catch (e) { /* polling below covers it */ }
        // Backstop: poll job status (also catches a job that finished before SSE attached)
        this._poll = setInterval(() => this._pollStatus(jobId), 3000);
      },
      async _pollStatus(jobId) {
        if (this.phase !== 'analyzing') return;
        try {
          const r = await fetch('/job-status/' + encodeURIComponent(jobId), { cache: 'no-store' });
          if (r.status === 404) return;
          const d = await r.json();
          if (d.status === 'completed') this._accept(d);
          else if (d.status === 'error') this._fail(d.error || this.t('errGeneric'));
          else if (d.progress && d.progress.step != null) this.step = num(d.progress.step, this.step);
        } catch (e) { /* transient */ }
      },
      async _finish(jobId) {
        if (this.phase !== 'analyzing') return;
        try {
          const r = await fetch('/job-status/' + encodeURIComponent(jobId), { cache: 'no-store' });
          const d = await r.json();
          if (d.status === 'completed') this._accept(d);
          else if (d.status === 'error') this._fail(d.error || this.t('errGeneric'));
        } catch (e) { this._fail(this.t('errNetwork')); }
      },
      _accept(payload) {
        if (this.phase !== 'analyzing') return;
        this._closeStreams();
        this.phase = 'idle';
        this.step = TOTAL_STEPS;
        const lang = this.recordingLang;
        this.file = null;
        Alpine.store('analysis').showPayload(payload, { language: lang });
        Alpine.store('toasts').show(this.t('done'), 'success');
        Alpine.store('health').check();
        Alpine.store('history').load().then(() => Alpine.store('analysis').resolveReportFilename());
      },
      _fail(message) {
        this._closeStreams();
        this.phase = 'error';
        this.failure = message || this.t('errGeneric');
        Alpine.store('toasts').show(this.failure, 'error');
      },
      cancel() {
        this._cancelled = true;
        if (this._xhr) { try { this._xhr.abort(); } catch (e) { /* ignore */ } this._xhr = null; this.phase = 'idle'; return; }
        const wasAnalyzing = this.phase === 'analyzing';
        this._closeStreams();
        this.phase = 'idle';
        this.step = 0;
        if (wasAnalyzing) Alpine.store('toasts').show(this.t('cancelledNote'), 'info');
      },
      _closeStreams() {
        if (this._es) { try { this._es.close(); } catch (e) { /* ignore */ } this._es = null; }
        if (this._poll) { clearInterval(this._poll); this._poll = null; }
      },
    });

    /* ---- analysis (view-model) ---- */
    Alpine.store('analysis', {
      view: null, loading: false, reportFilename: null,
      t(k, v) { return Alpine.store('ui').t(k, v); },
      riskClass(level) { return sevClass(level || 'low'); },
      displayName(name) { return stripStamp(name); },
      typeIcon(type) {
        return { emotion: 'i-voice', violence: 'i-alert', cry: 'i-drop', neglect: 'i-clock', language: 'i-bubble' }[type] || 'i-info';
      },
      showDesc(inc) {
        if (!inc || !inc.description) return false;
        if (inc.type === 'cry' || inc.type === 'neglect' || inc.type === 'language') return true;
        const key = 'label_' + String(inc.label || '').toLowerCase().replace(/\s+/g, '_');
        return !(STRINGS.en[key]);   // no structured title → the description is all we have
      },
      incidentTitle(inc) {
        if (!inc) return '';
        const key = 'label_' + String(inc.label || '').toLowerCase().replace(/\s+/g, '_');
        const table = STRINGS[Alpine.store('ui').lang] || STRINGS.en;
        if (table[key] || STRINGS.en[key]) return this.t(key);
        if (inc.label) return String(inc.label);
        return this.t('type_' + inc.type);
      },
      verdictLine(v) {
        const d = Alpine.store('ui').fmtDuration(v.duration);
        const n = v.incidents.length;
        let s = n === 0 ? this.t('verdictZero', { d }) : (n === 1 ? this.t('verdictOne', { d }) : this.t('verdictMany', { n, d }));
        const c = num(v.summary.critical_incidents);
        if (c > 0) s += ' ' + this.t('verdictCritical', { c });
        return s;
      },
      _show(vm) {
        this.view = vm;
        this.loading = false;
        Alpine.store('player').reset();
        Alpine.nextTick(() => {
          const el = document.getElementById('wave');
          if (el) Alpine.store('player').mount(el, vm);
        });
      },
      showPayload(payload, extra) {
        this.reportFilename = null;
        this._show(normalize(payload, 'payload', extra));
      },
      async openReport(filename) {
        this.loading = true;
        try {
          const r = await fetch('/report/' + encodeURIComponent(filename), { cache: 'no-store' });
          const d = await r.json();
          if (!r.ok || !d.report) throw new Error('bad report');
          this.reportFilename = filename;
          this._show(normalize(d.report, 'report', {}));
          if (window.innerWidth < 900) document.getElementById('main').scrollIntoView({ behavior: 'smooth' });
        } catch (e) {
          this.loading = false;
          Alpine.store('toasts').show(this.t('reportMissing'), 'error');
        }
      },
      resolveReportFilename() {
        if (!this.view || this.reportFilename) return;
        const hit = Alpine.store('history').items.find((it) => it.original_file === this.view.serverFile);
        if (hit) this.reportFilename = hit.filename;
      },
    });

    /* ---- history ---- */
    Alpine.store('history', {
      items: [], loading: false,
      async load() {
        this.loading = true;
        try {
          const r = await fetch('/reports', { cache: 'no-store' });
          const d = await r.json();
          this.items = Array.isArray(d) ? d : (d.reports || []);
        } catch (e) { this.items = []; }
        this.loading = false;
      },
    });

    /* ---- player ---- */
    Alpine.store('player', {
      ws: null, ready: false, playing: false, time: 0, duration: 0, error: false, activeIdx: null,
      _incidents: [], _stopAt: null, _url: null,
      colors() {
        const cs = getComputedStyle(document.documentElement);
        return { waveColor: cs.getPropertyValue('--wave-color').trim() || '#C7C7CC', progressColor: cs.getPropertyValue('--wave-progress').trim() || '#007AFF' };
      },
      reset() {
        if (this.ws) { try { this.ws.destroy(); } catch (e) { /* ignore */ } }
        this.ws = null; this.ready = false; this.playing = false; this.time = 0; this.duration = 0; this.error = false; this.activeIdx = null; this._stopAt = null; this._url = null;
      },
      mount(el, vm) {
        if (!el || !vm || typeof WaveSurfer === 'undefined') { this.error = true; return; }
        if (this.ws && this._url === vm.audioUrl) return;
        this.reset();
        if (!vm.audioUrl) { this.error = true; return; }
        this._incidents = vm.incidents;
        this._url = vm.audioUrl;
        el.innerHTML = '';
        const c = this.colors();
        try {
          this.ws = WaveSurfer.create({
            container: el, url: vm.audioUrl, height: 120, normalize: true,
            waveColor: c.waveColor, progressColor: c.progressColor, cursorColor: c.progressColor, cursorWidth: 1,
            barWidth: 2, barGap: 1, barRadius: 2, dragToSeek: true,
          });
        } catch (e) { this.error = true; return; }
        this.ws.on('ready', (d) => { this.ready = true; this.duration = d || this.ws.getDuration(); });
        this.ws.on('timeupdate', (t) => {
          this.time = t;
          if (this._stopAt != null && t >= this._stopAt) { this._stopAt = null; this.ws.pause(); }
        });
        this.ws.on('play', () => { this.playing = true; });
        this.ws.on('pause', () => { this.playing = false; });
        this.ws.on('finish', () => { this.playing = false; this._stopAt = null; });
        this.ws.on('interaction', () => { this.activeIdx = null; this._stopAt = null; });
        this.ws.on('error', () => { this.error = true; this.ready = false; });
      },
      applyTheme() { if (this.ws) { const c = this.colors(); try { this.ws.setOptions({ waveColor: c.waveColor, progressColor: c.progressColor, cursorColor: c.progressColor }); } catch (e) { /* ignore */ } } },
      toggle() { if (!this.ws || !this.ready) return; this._stopAt = null; this.ws.playPause(); },
      seekPlay(t) { if (!this.ws || !this.ready) return; this._stopAt = null; this.activeIdx = null; this.ws.setTime(clamp(t, 0, this.duration)); this.ws.play(); },
      playIncident(i) {
        const inc = this._incidents[i];
        if (!inc) return;
        this.activeIdx = i;
        const row = document.getElementById('finding-' + i);
        if (row) row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        if (!this.ws || !this.ready) return;
        const lead = 0.5;
        this.ws.setTime(clamp(inc.start - lead, 0, this.duration));
        this._stopAt = Math.min(inc.end + lead, this.duration || inc.end + lead);
        this.ws.play();
      },
      next() {
        if (!this._incidents.length) return;
        const t = this.time + 0.25;
        let idx = this._incidents.findIndex((inc) => inc.start > t);
        if (idx < 0) idx = 0;
        this.playIncident(idx);
      },
      prev() {
        if (!this._incidents.length) return;
        const t = this.time - 1.0;
        let idx = -1;
        this._incidents.forEach((inc, i) => { if (inc.start < t) idx = i; });
        if (idx < 0) idx = this._incidents.length - 1;
        this.playIncident(idx);
      },
    });

    /* ---- boot ---- */
    Alpine.store('history').load();
    Alpine.store('health').start();
    document.addEventListener('keydown', (e) => {
      if (e.code !== 'Space' || e.target.closest('input, textarea, button, select, [contenteditable], audio')) return;
      const p = Alpine.store('player');
      if (p.ready) { e.preventDefault(); p.toggle(); }
    });
  });
})();
