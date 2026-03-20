/**
 * SoundShield-AI  --  Chart.js Visualizations (Alpine.js)
 *
 * Provides severity doughnut, incident-type bar, timeline density,
 * and emotion radar charts.  All charts are responsive and dark-mode aware.
 */

/* ------------------------------------------------------------------ */
/*  Color constants                                                    */
/* ------------------------------------------------------------------ */
const SEVERITY_COLORS = {
  low:      '#22C55E',
  medium:   '#EAB308',
  high:     '#F97316',
  critical: '#EF4444',
};

const INCIDENT_TYPE_COLORS = {
  emotions: '#F97316',
  violence: '#EF4444',
  cries:    '#3B82F6',
  neglect:  '#6B7280',
  language: '#8B5CF6',
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
function isDarkMode() {
  return document.documentElement.classList.contains('dark');
}

function getGridColor() {
  return isDarkMode() ? 'rgba(148, 163, 184, 0.15)' : 'rgba(15, 23, 42, 0.08)';
}

function getTextColor() {
  return isDarkMode() ? '#94A3B8' : '#475569';
}

function getFontConfig() {
  return {
    family: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    size: 12,
    weight: '500',
  };
}

function getDefaultChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: true,
    animation: {
      duration: 600,
      easing: 'easeOutQuart',
    },
    plugins: {
      legend: {
        labels: {
          color: getTextColor(),
          font: getFontConfig(),
          padding: 16,
          usePointStyle: true,
          pointStyleWidth: 8,
        },
      },
      tooltip: {
        backgroundColor: isDarkMode() ? '#1E293B' : '#FFFFFF',
        titleColor: isDarkMode() ? '#F1F5F9' : '#0F172A',
        bodyColor: isDarkMode() ? '#CBD5E1' : '#475569',
        borderColor: isDarkMode() ? '#334155' : '#E2E8F0',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        titleFont: { ...getFontConfig(), weight: '600' },
        bodyFont: getFontConfig(),
        displayColors: true,
        boxPadding: 4,
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Alpine component: chartsManager                                    */
/* ------------------------------------------------------------------ */
function chartsManager() {
  return {
    /* ---- internal chart instances ---- */
    _charts: {},

    /* ---- render all charts ---- */
    renderCharts(results, stats) {
      // Destroy existing charts first
      this.destroyAll();

      if (!results) return;

      // Build severity distribution from stats or results
      const distribution = this._buildSeverityDistribution(results, stats);
      this.severityDoughnut('severity-chart', distribution);

      // Incident type counts
      this.incidentTypeBar('type-chart', results);

      // Timeline density
      const allIncidents = this._collectAllIncidents(results);
      const duration = (results.duration || results.audio_duration || stats?.duration || 0);
      if (allIncidents.length > 0 && duration > 0) {
        this.timelineDensity('timeline-chart', allIncidents, duration);
      }

      // Emotion radar if HuBERT data is available
      const emotionData = this._extractEmotionData(results);
      if (emotionData) {
        this.emotionRadar('emotion-chart', emotionData);
      }
    },

    /* ---- Severity Doughnut ---- */
    severityDoughnut(canvasId, distribution) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      const labels = ['Low', 'Medium', 'High', 'Critical'];
      const data = [
        distribution.low || 0,
        distribution.medium || 0,
        distribution.high || 0,
        distribution.critical || 0,
      ];
      const colors = [
        SEVERITY_COLORS.low,
        SEVERITY_COLORS.medium,
        SEVERITY_COLORS.high,
        SEVERITY_COLORS.critical,
      ];

      const options = getDefaultChartOptions();
      options.cutout = '65%';
      options.plugins.legend.position = 'bottom';

      this._charts[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: labels,
          datasets: [{
            data: data,
            backgroundColor: colors,
            borderColor: isDarkMode() ? '#1E293B' : '#FFFFFF',
            borderWidth: 3,
            hoverOffset: 6,
          }],
        },
        options: options,
        plugins: [{
          // Center text plugin
          id: 'doughnutCenter',
          beforeDraw(chart) {
            const total = chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
            if (total === 0) return;

            const { ctx: c, chartArea: { width, height, top } } = chart;
            c.save();
            c.textAlign = 'center';
            c.textBaseline = 'middle';

            const centerX = width / 2 + chart.chartArea.left;
            const centerY = top + height / 2;

            c.font = `700 ${Math.round(height * 0.15)}px Inter, sans-serif`;
            c.fillStyle = getTextColor();
            c.fillText(total.toString(), centerX, centerY - 8);

            c.font = `500 ${Math.round(height * 0.07)}px Inter, sans-serif`;
            c.fillStyle = getTextColor();
            c.fillText('incidents', centerX, centerY + 14);

            c.restore();
          },
        }],
      });
    },

    /* ---- Incident Type Bar ---- */
    incidentTypeBar(canvasId, results) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

      const ctx = canvas.getContext('2d');

      const types = {
        emotions: this._countIncidents(results, ['emotions', 'emotion', 'emotion_incidents']),
        violence: this._countIncidents(results, ['violence', 'violence_incidents']),
        cries: this._countIncidents(results, ['cries', 'cry', 'cry_incidents']),
        neglect: this._countIncidents(results, ['neglect', 'neglect_incidents']),
        language: this._countIncidents(results, ['language', 'language_incidents']),
      };

      const labels = Object.keys(types).map(k => k.charAt(0).toUpperCase() + k.slice(1));
      const data = Object.values(types);
      const colors = Object.keys(types).map(k => INCIDENT_TYPE_COLORS[k] || '#6B7280');

      const options = getDefaultChartOptions();
      options.indexAxis = 'y';
      options.plugins.legend.display = false;
      options.scales = {
        x: {
          beginAtZero: true,
          ticks: {
            color: getTextColor(),
            font: getFontConfig(),
            stepSize: 1,
            precision: 0,
          },
          grid: {
            color: getGridColor(),
          },
        },
        y: {
          ticks: {
            color: getTextColor(),
            font: { ...getFontConfig(), size: 13 },
          },
          grid: {
            display: false,
          },
        },
      };

      this._charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            data: data,
            backgroundColor: colors.map(c => c + '33'),  // 20% opacity fill
            borderColor: colors,
            borderWidth: 2,
            borderRadius: 6,
            barPercentage: 0.7,
          }],
        },
        options: options,
      });
    },

    /* ---- Timeline Density ---- */
    timelineDensity(canvasId, incidents, durationSeconds) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      const durationMinutes = durationSeconds / 60;

      // Create per-minute buckets
      const bucketCount = Math.max(1, Math.ceil(durationMinutes));
      const buckets = new Array(bucketCount).fill(0);

      incidents.forEach((inc) => {
        const time = inc.start_time || inc.start || inc.timestamp || 0;
        const bucket = Math.min(Math.floor(time / 60), bucketCount - 1);
        buckets[bucket]++;
      });

      const labels = buckets.map((_, i) => i + 1);

      const options = getDefaultChartOptions();
      options.plugins.legend.display = false;
      options.scales = {
        x: {
          title: {
            display: true,
            text: 'Time (minutes)',
            color: getTextColor(),
            font: getFontConfig(),
          },
          ticks: {
            color: getTextColor(),
            font: getFontConfig(),
          },
          grid: {
            color: getGridColor(),
          },
        },
        y: {
          title: {
            display: true,
            text: 'Incidents / minute',
            color: getTextColor(),
            font: getFontConfig(),
          },
          beginAtZero: true,
          ticks: {
            color: getTextColor(),
            font: getFontConfig(),
            stepSize: 1,
            precision: 0,
          },
          grid: {
            color: getGridColor(),
          },
        },
      };

      // Gradient fill
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, 'rgba(37, 99, 235, 0.3)');
      gradient.addColorStop(1, 'rgba(37, 99, 235, 0.02)');

      this._charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            data: buckets,
            borderColor: '#2563EB',
            backgroundColor: gradient,
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointBackgroundColor: '#2563EB',
            pointBorderColor: isDarkMode() ? '#1E293B' : '#FFFFFF',
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
          }],
        },
        options: options,
      });
    },

    /* ---- Emotion Radar ---- */
    emotionRadar(canvasId, emotionData) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

      const ctx = canvas.getContext('2d');

      // emotionData expected: { angry: 0.1, happy: 0.5, sad: 0.2, ... }
      const labels = Object.keys(emotionData).map(k => k.charAt(0).toUpperCase() + k.slice(1));
      const data = Object.values(emotionData);

      const options = getDefaultChartOptions();
      options.plugins.legend.display = false;
      options.scales = {
        r: {
          angleLines: {
            color: getGridColor(),
          },
          grid: {
            color: getGridColor(),
          },
          pointLabels: {
            color: getTextColor(),
            font: { ...getFontConfig(), size: 11 },
          },
          ticks: {
            display: false,
            stepSize: 0.2,
          },
          suggestedMin: 0,
          suggestedMax: 1,
        },
      };

      this._charts[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: {
          labels: labels,
          datasets: [{
            data: data,
            backgroundColor: 'rgba(37, 99, 235, 0.15)',
            borderColor: '#2563EB',
            borderWidth: 2,
            pointBackgroundColor: '#2563EB',
            pointBorderColor: isDarkMode() ? '#1E293B' : '#FFFFFF',
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
          }],
        },
        options: options,
      });
    },

    /* ---- internal helpers ---- */
    _buildSeverityDistribution(results, stats) {
      // If stats provides it directly
      if (stats && stats.severity_distribution) {
        return stats.severity_distribution;
      }

      // Build from results
      const dist = { low: 0, medium: 0, high: 0, critical: 0 };
      const allIncidents = this._collectAllIncidents(results);

      allIncidents.forEach((inc) => {
        const sev = (inc.severity || inc.risk_level || 'low').toLowerCase();
        if (dist.hasOwnProperty(sev)) {
          dist[sev]++;
        } else {
          dist.low++;
        }
      });

      return dist;
    },

    _countIncidents(results, keys) {
      let count = 0;
      for (const key of keys) {
        const items = results[key];
        if (Array.isArray(items)) {
          count += items.length;
        }
      }

      // Also check nested results.incidents object
      if (results.incidents && typeof results.incidents === 'object' && !Array.isArray(results.incidents)) {
        for (const key of keys) {
          const items = results.incidents[key];
          if (Array.isArray(items)) {
            count += items.length;
          }
        }
      }

      return count;
    },

    _collectAllIncidents(results) {
      const incidents = [];
      const keys = [
        'violence', 'violence_incidents',
        'emotions', 'emotion', 'emotion_incidents',
        'cries', 'cry', 'cry_incidents',
        'neglect', 'neglect_incidents',
        'language', 'language_incidents',
      ];

      for (const key of keys) {
        const items = results[key];
        if (Array.isArray(items)) {
          incidents.push(...items);
        }
      }

      // Flat incidents array
      if (Array.isArray(results.incidents)) {
        incidents.push(...results.incidents);
      } else if (results.incidents && typeof results.incidents === 'object') {
        for (const items of Object.values(results.incidents)) {
          if (Array.isArray(items)) {
            incidents.push(...items);
          }
        }
      }

      return incidents;
    },

    _extractEmotionData(results) {
      // Look for HuBERT emotion profile data
      if (results.emotion_profile) return results.emotion_profile;
      if (results.hubert_emotions) return results.hubert_emotions;

      // Try to aggregate from emotion incidents
      const emotionIncidents = [
        ...(results.emotions || []),
        ...(results.emotion || []),
        ...(results.emotion_incidents || []),
      ];

      if (emotionIncidents.length === 0) return null;

      // Aggregate emotion scores
      const scores = {};
      let counted = 0;

      emotionIncidents.forEach((inc) => {
        if (inc.scores && typeof inc.scores === 'object') {
          for (const [emotion, score] of Object.entries(inc.scores)) {
            scores[emotion] = (scores[emotion] || 0) + score;
          }
          counted++;
        } else if (inc.emotion && inc.confidence) {
          scores[inc.emotion] = (scores[inc.emotion] || 0) + inc.confidence;
          counted++;
        }
      });

      if (counted === 0 || Object.keys(scores).length < 3) return null;

      // Average
      for (const key of Object.keys(scores)) {
        scores[key] = scores[key] / counted;
      }

      return scores;
    },

    /* ---- cleanup ---- */
    destroyAll() {
      for (const [id, chart] of Object.entries(this._charts)) {
        if (chart) {
          try {
            chart.destroy();
          } catch (_) { /* ignore */ }
        }
      }
      this._charts = {};
    },

    /* ---- init ---- */
    init() {
      // Re-render charts when analysis completes
      document.addEventListener('analysis-complete', (e) => {
        this.$nextTick(() => {
          this.renderCharts(e.detail, e.detail.stats || null);
        });
      });

      // Update chart colors on dark mode change
      document.addEventListener('dark-mode-changed', () => {
        // Re-render if we have active charts
        if (Object.keys(this._charts).length > 0) {
          // We need the original data to re-render, so we trigger a full re-render
          // via the app state results. Dispatch a request for it.
          document.dispatchEvent(new CustomEvent('charts-refresh-requested'));
        }
      });
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Register with Alpine                                               */
/* ------------------------------------------------------------------ */
document.addEventListener('alpine:init', () => {
  Alpine.data('chartsManager', chartsManager);
});

/* Export */
window.SoundShield = window.SoundShield || {};
window.SoundShield.chartsManager = chartsManager;
