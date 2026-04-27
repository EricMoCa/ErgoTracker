import { useState } from 'react'
import { getReportUrl, getAnnotatedVideoUrl } from '../api'
import RiskTimeline from './charts/RiskTimeline'
import RiskPieChart from './charts/RiskPieChart'

const RISK_ORDER = ['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']

function maxRisk(frameScores) {
  if (!frameScores?.length) return 'NEGLIGIBLE'
  return frameScores.reduce((best, f) => {
    const r = f.overall_risk || 'NEGLIGIBLE'
    return RISK_ORDER.indexOf(r) > RISK_ORDER.indexOf(best) ? r : best
  }, 'NEGLIGIBLE')
}

function AnnotatedVideoPlayer({ reportId }) {
  const [error, setError] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const videoUrl = getAnnotatedVideoUrl(reportId)

  if (error) {
    return (
      <div className="video-unavailable">
        <span>Video anotado no disponible</span>
        <small>El video se genera durante el análisis. Verifica que opencv-python esté instalado.</small>
      </div>
    )
  }

  return (
    <div className="video-player-wrapper">
      {!loaded && (
        <div className="video-loading">
          <div className="spinner" />
          <span>Cargando video anotado...</span>
        </div>
      )}
      <video
        src={videoUrl}
        controls
        autoPlay
        loop
        muted
        playsInline
        onLoadedData={() => setLoaded(true)}
        onError={() => setError(true)}
        style={{ display: loaded ? 'block' : 'none' }}
        className="annotated-video"
      />
      {loaded && (
        <div className="video-labels">
          <span>◀ Video + Esqueleto 2D</span>
          <span>Vista 3D Mundo ▶</span>
        </div>
      )}
    </div>
  )
}

export default function ResultsPanel({ result }) {
  const { id, video_path, duration_s, analyzed_frames,
          methods_used, frame_scores = [], summary = {} } = result

  const filename = video_path?.split(/[\\/]/).pop() || 'video'
  const pctHighRisk = ((summary.pct_frames_high_risk || 0) * 100).toFixed(1)
  const maxRiskLevel = maxRisk(frame_scores)

  return (
    <div className="results-panel">

      {/* Video side-by-side */}
      <div className="card video-card">
        <h2>Análisis Visual — {filename}</h2>
        <AnnotatedVideoPlayer reportId={id} />
      </div>

      {/* Resumen numérico */}
      <div className="card">
        <h2>Resumen</h2>
        <div className="summary-grid">
          <div className="summary-stat">
            <div className="stat-value">{duration_s?.toFixed(1) ?? '—'}s</div>
            <div className="stat-label">Duración</div>
          </div>
          <div className="summary-stat">
            <div className="stat-value">{analyzed_frames ?? '—'}</div>
            <div className="stat-label">Frames</div>
          </div>
          <div className="summary-stat">
            <div className="stat-value">{pctHighRisk}%</div>
            <div className="stat-label">Riesgo alto</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10 }}>
          <span style={{ fontSize: 13, color: '#64748b' }}>Riesgo máximo:</span>
          <span className={`risk-badge risk-${maxRiskLevel}`}>{maxRiskLevel}</span>
          <span style={{ fontSize: 13, color: '#64748b', marginLeft: 'auto' }}>
            {methods_used?.join(' · ')}
          </span>
        </div>

        {summary.max_reba_score != null && (
          <div style={{ marginTop: 6, fontSize: 13, color: '#64748b' }}>
            REBA máximo: <strong>{summary.max_reba_score}</strong> / 15
          </div>
        )}
      </div>

      {/* Gráficas */}
      {frame_scores.length > 0 && (
        <div className="charts-row">
          <div className="chart-container">
            <h3>Evolución temporal del riesgo</h3>
            <RiskTimeline frameScores={frame_scores} />
          </div>
          <div className="chart-container">
            <h3>Distribución de riesgo</h3>
            <RiskPieChart frameScores={frame_scores} />
          </div>
        </div>
      )}

      {/* Descargas */}
      <div className="download-section">
        <span style={{ fontSize: 12, color: '#64748b' }}>ID: {id}</span>
        <a href={getAnnotatedVideoUrl(id)} download className="btn btn-outline">
          Descargar video
        </a>
        <a href={getReportUrl(id)} target="_blank" rel="noreferrer" className="btn btn-primary">
          Descargar PDF
        </a>
      </div>
    </div>
  )
}
