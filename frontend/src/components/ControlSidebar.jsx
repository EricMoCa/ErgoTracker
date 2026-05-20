import { useState } from 'react'
import { startAnalysis, getReportUrl, getAnnotatedVideoUrl } from '../api'

const METHODS = ['REBA', 'RULA', 'OWAS']
const RISK_ORDER = ['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']

const ENGINES = [
  { id: 'auto',    label: 'Auto',    icon: 'auto_awesome',  desc: 'Selección automática' },
  { id: 'gvhmr',  label: 'GVHMR',   icon: 'model_training', desc: 'World-grounded · cámara móvil' },
  { id: 'wham',   label: 'WHAM',    icon: 'directions_walk', desc: 'Contacto pie-suelo · marcha' },
  { id: 'tram',   label: 'TRAM',    icon: 'route',          desc: 'SLAM + ViT-H · escala métrica' },
  { id: 'humanmm',label: 'HumanMM', icon: 'movie',          desc: 'Multi-plano · cortes de cámara' },
]

function maxRisk(frameScores) {
  if (!frameScores?.length) return 'NEGLIGIBLE'
  return frameScores.reduce((best, f) => {
    const r = f.overall_risk || 'NEGLIGIBLE'
    return RISK_ORDER.indexOf(r) > RISK_ORDER.indexOf(best) ? r : best
  }, 'NEGLIGIBLE')
}

export default function ControlSidebar({
  file,
  onJobStarted,
  job,
  result,
  analyzing,
  progress = 0,
  stage = '',
  showSkeleton: showSkeletonProp,
  setShowSkeleton: setShowSkeletonProp,
  showRisk: showRiskProp,
  setShowRisk: setShowRiskProp,
  showTrail: showTrailProp,
  setShowTrail: setShowTrailProp,
  showMesh: showMeshProp,
  setShowMesh: setShowMeshProp,
}) {
  const [height, setHeight]   = useState(170)
  const [methods, setMethods] = useState(['REBA'])
  const [error, setError]     = useState(null)
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [showRisk, setShowRisk]         = useState(true)
  const [showTrail, setShowTrail]       = useState(true)
  const [showMesh, setShowMesh]         = useState(true)

  // Allow parent to control visualization toggles (optional)
  const ctrlShowSkeleton = showSkeletonProp ?? showSkeleton
  const ctrlSetShowSkeleton = setShowSkeletonProp ?? setShowSkeleton
  const ctrlShowRisk = showRiskProp ?? showRisk
  const ctrlSetShowRisk = setShowRiskProp ?? setShowRisk
  const ctrlShowTrail = showTrailProp ?? showTrail
  const ctrlSetShowTrail = setShowTrailProp ?? setShowTrail
  const ctrlShowMesh = showMeshProp ?? showMesh
  const ctrlSetShowMesh = setShowMeshProp ?? setShowMesh

  // Advanced pipeline state
  const [gpuMode, setGpuMode]               = useState(false)
  const [engine, setEngine]                 = useState('auto')
  const [gaitAnalysis, setGaitAnalysis]     = useState(false)
  const [multiShot, setMultiShot]           = useState(false)
  const [cameraMotion, setCameraMotion]     = useState(false)

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
      fd.append('processing_mode', gpuMode ? 'gpu_enhanced' : 'cpu_only')
      fd.append('preferred_engine', engine)
      fd.append('requires_gait_analysis', gaitAnalysis)
      fd.append('has_multiple_shots', multiShot)
      fd.append('camera_motion_high', cameraMotion)
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
              <input type="checkbox" checked={ctrlShowSkeleton} onChange={e => ctrlSetShowSkeleton(e.target.checked)} />
              <span className="toggle-slider" />
            </label>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">HUD de riesgo</span>
            <label className="toggle-switch">
              <input type="checkbox" checked={ctrlShowRisk} onChange={e => ctrlSetShowRisk(e.target.checked)} />
              <span className="toggle-slider" />
            </label>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">Trail de trayectoria</span>
            <label className="toggle-switch">
              <input type="checkbox" checked={ctrlShowTrail} onChange={e => ctrlSetShowTrail(e.target.checked)} />
              <span className="toggle-slider" />
            </label>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">Malla (SMPL-like)</span>
            <label className="toggle-switch">
              <input type="checkbox" checked={ctrlShowMesh} onChange={e => ctrlSetShowMesh(e.target.checked)} />
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

          {/* ── Motor 3D ── */}
          <div className="ctrl-group">
            <div className="ctrl-label" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span>Motor 3D</span>
              <label className="toggle-switch" style={{ marginLeft: 8 }}>
                <input type="checkbox" checked={gpuMode} onChange={e => setGpuMode(e.target.checked)} />
                <span className="toggle-slider" />
              </label>
            </div>
            <div style={{ fontSize: 10, color: 'var(--on-surface-variant)', marginBottom: 6 }}>
              {gpuMode
                ? <span style={{ color: 'var(--primary)' }}>GPU Enhanced — GVHMR / WHAM / TRAM / HumanMM</span>
                : 'CPU Only — MotionBERT Lite'}
            </div>

            {gpuMode && (
              <>
                <div style={{ display: 'grid', gap: 4, marginBottom: 8 }}>
                  {ENGINES.map(eng => (
                    <button
                      key={eng.id}
                      className={`engine-chip ${engine === eng.id ? 'active' : ''}`}
                      onClick={() => setEngine(eng.id)}
                      title={eng.desc}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 13 }}>{eng.icon}</span>
                      <span style={{ fontWeight: 600 }}>{eng.label}</span>
                      <span className="engine-desc">{eng.desc}</span>
                    </button>
                  ))}
                </div>

                <div className="ctrl-label" style={{ marginBottom: 4 }}>Perfil de video</div>
                <div className="toggle-row">
                  <span className="toggle-label">Análisis de marcha</span>
                  <label className="toggle-switch">
                    <input type="checkbox" checked={gaitAnalysis} onChange={e => setGaitAnalysis(e.target.checked)} />
                    <span className="toggle-slider" />
                  </label>
                </div>
                <div className="toggle-row">
                  <span className="toggle-label">Múltiples planos</span>
                  <label className="toggle-switch">
                    <input type="checkbox" checked={multiShot} onChange={e => setMultiShot(e.target.checked)} />
                    <span className="toggle-slider" />
                  </label>
                </div>
                <div className="toggle-row">
                  <span className="toggle-label">Cámara en movimiento</span>
                  <label className="toggle-switch">
                    <input type="checkbox" checked={cameraMotion} onChange={e => setCameraMotion(e.target.checked)} />
                    <span className="toggle-slider" />
                  </label>
                </div>
              </>
            )}
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

      {/* Job status — show as soon as analysis starts */}
      {(analyzing || job) && (
        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="material-symbols-outlined">monitoring</span>
            <span className="sidebar-section-title">Estado</span>
          </div>
          <div className="sidebar-section-body">
            <div className="job-progress">
              <div className="job-status-row" style={{ marginBottom: 4 }}>
                <span className={`job-status-badge ${job?.status ?? 'running'}`}>
                  {job?.status ?? 'running'}
                </span>
                {(stage || analyzing) && (
                  <span style={{ fontSize: 10, color: 'var(--primary)', flex: 1, textAlign: 'right' }}>
                    {stage || 'Iniciando...'}
                  </span>
                )}
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill-cyber"
                  style={{
                    width: `${job?.status === 'completed' ? 100 : progress}%`,
                    transition: 'width 0.4s ease',
                  }}
                />
              </div>
              <div style={{ fontSize: 10, color: 'var(--on-surface-variant)', marginTop: 3 }}>
                {job?.status === 'completed' ? '100%' : `${Math.round(progress)}%`}
              </div>
              {job?.error && <div className="error-banner">{job.error}</div>}
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
