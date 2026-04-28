import { useState, useRef } from 'react'
import { startAnalysis, getReportUrl, getAnnotatedVideoUrl } from '../api'

const METHODS = ['REBA', 'RULA', 'OWAS']
const RISK_ORDER = ['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']

function maxRisk(frameScores) {
  if (!frameScores?.length) return 'NEGLIGIBLE'
  return frameScores.reduce((best, f) => {
    const r = f.overall_risk || 'NEGLIGIBLE'
    return RISK_ORDER.indexOf(r) > RISK_ORDER.indexOf(best) ? r : best
  }, 'NEGLIGIBLE')
}

export default function ControlSidebar({ file, onJobStarted, job, result, analyzing }) {
  const [height, setHeight]   = useState(170)
  const [methods, setMethods] = useState(['REBA'])
  const [error, setError]     = useState(null)
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [showRisk, setShowRisk]         = useState(true)
  const [showTrail, setShowTrail]       = useState(true)

  const toggleMethod = (m) =>
    setMethods(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m])

  const handleAnalyze = async () => {
    if (!file) return
    setError(null)
    try {
      const fd = new FormData()
      fd.append('video', file)
      fd.append('person_height_cm', height)
      fd.append('ergo_methods', methods.join(','))
      const { job_id } = await startAnalysis(fd)
      onJobStarted(job_id)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    }
  }

  const pctHighRisk = result ? ((result.summary?.pct_frames_high_risk || 0) * 100).toFixed(1) : null
  const maxRiskLevel = result ? maxRisk(result.frame_scores || []) : null

  return (
    <aside className="control-sidebar">

      {/* Visualization toggles */}
      <div className="sidebar-section">
        <div className="sidebar-section-header">
          <span className="material-symbols-outlined">tune</span>
          <span className="sidebar-section-title">Visualización</span>
        </div>
        <div className="sidebar-section-body">
          <div className="toggle-row">
            <span className="toggle-label">Esqueleto</span>
            <label className="toggle-switch">
              <input type="checkbox" checked={showSkeleton} onChange={e => setShowSkeleton(e.target.checked)} />
              <span className="toggle-slider" />
            </label>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">HUD de riesgo</span>
            <label className="toggle-switch">
              <input type="checkbox" checked={showRisk} onChange={e => setShowRisk(e.target.checked)} />
              <span className="toggle-slider" />
            </label>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">Trail de trayectoria</span>
            <label className="toggle-switch">
              <input type="checkbox" checked={showTrail} onChange={e => setShowTrail(e.target.checked)} />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>
      </div>

      {/* Analysis pipeline */}
      <div className="sidebar-section">
        <div className="sidebar-section-header">
          <span className="material-symbols-outlined">rocket_launch</span>
          <span className="sidebar-section-title">Analysis Pipeline</span>
        </div>
        <div className="sidebar-section-body">
          <div className="ctrl-group">
            <div className="ctrl-label">Altura (cm)</div>
            <input
              type="number"
              className="ctrl-input"
              value={height}
              min={100}
              max={220}
              onChange={e => setHeight(Number(e.target.value))}
            />
          </div>

          <div className="ctrl-group">
            <div className="ctrl-label">Métodos ergonómicos</div>
            <div className="method-chips">
              {METHODS.map(m => (
                <button
                  key={m}
                  className={`method-chip ${methods.includes(m) ? 'active' : ''}`}
                  onClick={() => toggleMethod(m)}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}

          <button
            className="btn-analyze"
            onClick={handleAnalyze}
            disabled={!file || methods.length === 0 || analyzing}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
              {analyzing ? 'hourglass_top' : 'biotech'}
            </span>
            {analyzing ? 'Procesando...' : 'Iniciar Análisis'}
          </button>
        </div>
      </div>

      {/* Job status */}
      {job && (
        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="material-symbols-outlined">monitoring</span>
            <span className="sidebar-section-title">Estado</span>
          </div>
          <div className="sidebar-section-body">
            <div className="job-progress">
              <div className="job-status-row">
                <span className={`job-status-badge ${job.status}`}>{job.status}</span>
                {job.elapsed != null && (
                  <span style={{ fontSize: 11, color: 'var(--on-surface-variant)' }}>{job.elapsed}s</span>
                )}
              </div>
              <div className="progress-track">
                <div
                  className={`progress-fill-cyber ${['pending','running'].includes(job.status) ? 'indeterminate' : ''}`}
                  style={job.status === 'completed' ? { width: '100%' } : job.status === 'failed' ? { width: '0%' } : {}}
                />
              </div>
              {job.error && <div className="error-banner">{job.error}</div>}
            </div>
          </div>
        </div>
      )}

      {/* Results summary */}
      {result && (
        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="material-symbols-outlined">analytics</span>
            <span className="sidebar-section-title">Resultados</span>
          </div>
          <div className="sidebar-section-body">
            <div className="results-metrics">
              <div className="metric-card">
                <div className="metric-value">{result.duration_s?.toFixed(1) ?? '—'}</div>
                <div className="metric-label">Segundos</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{result.analyzed_frames ?? '—'}</div>
                <div className="metric-label">Frames</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{pctHighRisk}%</div>
                <div className="metric-label">Riesgo alto</div>
              </div>
              {result.summary?.max_reba_score != null && (
                <div className="metric-card">
                  <div className="metric-value">{result.summary.max_reba_score}</div>
                  <div className="metric-label">REBA máx</div>
                </div>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
              <span style={{ fontSize: 11, color: 'var(--on-surface-variant)' }}>Riesgo:</span>
              <span className={`risk-pill ${maxRiskLevel}`}>{maxRiskLevel}</span>
            </div>
            <div className="download-row">
              <a href={getAnnotatedVideoUrl(result.id)} download className="btn-sm btn-cyber">
                <span className="material-symbols-outlined">videocam</span>
                Video
              </a>
              <a href={getReportUrl(result.id)} target="_blank" rel="noreferrer" className="btn-sm btn-secondary">
                <span className="material-symbols-outlined">picture_as_pdf</span>
                PDF
              </a>
            </div>
          </div>
        </div>
      )}

    </aside>
  )
}
