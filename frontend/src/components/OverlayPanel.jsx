import { useState } from 'react'
import { getAnnotatedVideoUrl } from '../api'

export default function OverlayPanel({ reportId, analyzing }) {
  const [error, setError] = useState(false)
  const [loaded, setLoaded] = useState(false)

  if (analyzing) {
    return (
      <div className="viewport-panel">
        <div className="viewport-header">
          <span className="viewport-label">SMPL_OVERLAY</span>
          <span className="viewport-badge">PROCESSING</span>
        </div>
        <div className="viewport-body">
          <div className="vp-loading">
            <div className="spinner-ring" />
            <p>Analizando video...</p>
          </div>
        </div>
      </div>
    )
  }

  if (!reportId) {
    return (
      <div className="viewport-panel">
        <div className="viewport-header">
          <span className="viewport-label">SMPL_OVERLAY</span>
          <span className="viewport-badge">STANDBY</span>
        </div>
        <div className="viewport-body">
          <div className="vp-overlay-empty">
            <span className="material-symbols-outlined">accessibility_new</span>
            <p>Inicia el análisis para ver el overlay</p>
          </div>
        </div>
      </div>
    )
  }

  const videoUrl = getAnnotatedVideoUrl(reportId)

  return (
    <div className="viewport-panel">
      <div className="viewport-header">
        <span className="viewport-label">SMPL_OVERLAY</span>
        <span className="viewport-badge">{loaded ? 'LIVE' : 'LOADING'}</span>
      </div>
      <div className="viewport-body">
        {!loaded && !error && (
          <div className="vp-loading" style={{ position: 'absolute', zIndex: 2 }}>
            <div className="spinner-ring" />
            <p>Cargando video anotado...</p>
          </div>
        )}
        {error ? (
          <div className="vp-overlay-empty">
            <span className="material-symbols-outlined">error_outline</span>
            <p>Video no disponible</p>
          </div>
        ) : (
          <video
            className="viewport-video"
            src={videoUrl}
            controls
            autoPlay
            loop
            muted
            playsInline
            onLoadedData={() => setLoaded(true)}
            onError={() => setError(true)}
            style={{ display: loaded ? 'block' : 'none' }}
          />
        )}
      </div>
    </div>
  )
}
