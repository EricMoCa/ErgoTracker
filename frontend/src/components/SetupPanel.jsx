import { useState, useEffect, useCallback } from 'react'
import { getSetupStatus, startModelDownload, pollDownloadJob } from '../api'

const MODEL_META = {
  yolov8n:        { label: 'YOLOv8n', desc: 'Detector de personas (~6 MB)' },
  rtmpose_m:      { label: 'RTMPose-m', desc: 'Estimación de pose 2D (~90 MB ZIP)' },
  motionbert_lite: { label: 'MotionBERT Lite', desc: 'Lifting 2D→3D (~200 MB)' },
}

const PKG_LABELS = {
  cv2:         'OpenCV (cv2)',
  onnxruntime: 'ONNX Runtime',
  numpy:       'NumPy',
  torch:       'PyTorch (opcional)',
}

function StatusDot({ ok, loading }) {
  if (loading) return <span className="health-dot loading" style={{ display: 'inline-block' }} />
  return <span className="health-dot" style={{ display: 'inline-block', background: ok ? '#4ade80' : '#f87171' }} />
}

function PkgRow({ name, info }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
      <StatusDot ok={info.available} />
      <span style={{ flex: 1, fontSize: 13 }}>{PKG_LABELS[name] || name}</span>
      <span style={{ fontSize: 11, color: info.available ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>
        {info.available ? 'OK' : 'No instalado'}
      </span>
    </div>
  )
}

function ModelRow({ modelKey, info, download, onDownload }) {
  const meta = MODEL_META[modelKey] || { label: modelKey, desc: '' }
  const isDownloading = download?.status === 'running' || download?.status === 'pending'
  const pct = download?.progress?.pct || 0

  return (
    <div style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <StatusDot ok={info.available} />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 500 }}>{meta.label}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {meta.desc}
            {info.available && ` · ${info.size_mb} MB en disco`}
          </div>
        </div>
        {!info.available && !isDownloading && (
          <button className="btn btn-primary" style={{ fontSize: 12, padding: '5px 12px' }} onClick={() => onDownload(modelKey)}>
            Descargar
          </button>
        )}
        {isDownloading && (
          <span style={{ fontSize: 12, color: 'var(--primary-light)', fontWeight: 500 }}>
            {pct > 0 ? `${pct}%` : 'Iniciando...'}
          </span>
        )}
        {download?.status === 'completed' && !info.available && (
          <span style={{ fontSize: 12, color: 'var(--success)', fontWeight: 600 }}>Listo</span>
        )}
        {download?.status === 'failed' && (
          <span style={{ fontSize: 12, color: 'var(--danger)' }} title={download.error}>Error</span>
        )}
      </div>
      {isDownloading && pct > 0 && (
        <div className="progress-bar" style={{ marginTop: 6 }}>
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
      )}
      {isDownloading && pct === 0 && (
        <div className="progress-bar" style={{ marginTop: 6 }}>
          <div className="progress-fill indeterminate" />
        </div>
      )}
      {download?.status === 'failed' && download.error && (
        <div className="error-msg" style={{ marginTop: 4, fontSize: 11 }}>{download.error}</div>
      )}
    </div>
  )
}

export default function SetupPanel() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [downloads, setDownloads] = useState({})

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const s = await getSetupStatus()
      setStatus(s)
    } catch (e) {
      setError('No se puede conectar al servidor API en localhost:8000')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function handleDownload(modelKey) {
    try {
      const { job_id } = await startModelDownload(modelKey)
      setDownloads(d => ({ ...d, [modelKey]: { jobId: job_id, status: 'pending', progress: { pct: 0 } } }))
      pollProgress(modelKey, job_id)
    } catch (e) {
      setDownloads(d => ({ ...d, [modelKey]: { status: 'failed', error: e.message } }))
    }
  }

  function pollProgress(modelKey, jobId) {
    const iv = setInterval(async () => {
      try {
        const data = await pollDownloadJob(jobId)
        setDownloads(d => ({
          ...d,
          [modelKey]: { jobId, status: data.status, progress: data.progress, error: data.error },
        }))
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(iv)
          if (data.status === 'completed') setTimeout(refresh, 800)
        }
      } catch {
        clearInterval(iv)
      }
    }, 1500)
  }

  async function handleDownloadAll() {
    if (!status) return
    for (const key of Object.keys(MODEL_META)) {
      if (!status.models[key]?.available && !downloads[key]?.status) {
        await handleDownload(key)
        await new Promise(r => setTimeout(r, 300))
      }
    }
  }

  const allModelsOk = status && Object.keys(MODEL_META).every(k => status.models[k]?.available)
  const anyMissing  = status && Object.keys(MODEL_META).some(k => !status.models[k]?.available)
  const anyDownloading = Object.values(downloads).some(d => d.status === 'running' || d.status === 'pending')

  return (
    <div style={{ maxWidth: 640 }}>
      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ margin: 0, padding: 0, border: 'none' }}>Estado del Entorno</h2>
          <button className="btn btn-outline" onClick={refresh} disabled={loading}>
            {loading ? 'Actualizando...' : 'Actualizar'}
          </button>
        </div>

        {error && (
          <div className="video-unavailable" style={{ marginBottom: 12 }}>
            <span>{error}</span>
            <small>Asegúrate de que el servidor API esté corriendo: <code>uvicorn api.main:app --reload</code></small>
          </div>
        )}

        {status && (
          <>
            {/* Paquetes Python */}
            <h3 style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
              Paquetes Python
            </h3>
            <div style={{ marginBottom: 20 }}>
              {Object.entries(status.packages).map(([name, info]) => (
                <PkgRow key={name} name={name} info={info} />
              ))}
            </div>

            {/* Modelos ONNX */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <h3 style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
                Modelos ONNX — {status.model_dir}
              </h3>
              {anyMissing && !anyDownloading && (
                <button className="btn btn-success" style={{ fontSize: 12, padding: '4px 12px' }} onClick={handleDownloadAll}>
                  Descargar todos
                </button>
              )}
            </div>

            {allModelsOk && (
              <div className="rules-result" style={{ marginBottom: 12 }}>
                Todos los modelos ONNX están disponibles. La detección de personas estará activa.
              </div>
            )}

            <div style={{ marginBottom: 20 }}>
              {Object.entries(status.models).map(([key, info]) => (
                <ModelRow
                  key={key}
                  modelKey={key}
                  info={info}
                  download={downloads[key]}
                  onDownload={handleDownload}
                />
              ))}
            </div>

            {/* Ollama */}
            <h3 style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
              Ollama (LLM — opcional)
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
                <StatusDot ok={status.ollama.available} />
                <span style={{ flex: 1, fontSize: 13 }}>Servidor Ollama</span>
                <span style={{ fontSize: 11, color: status.ollama.available ? 'var(--success)' : 'var(--text-muted)', fontWeight: 600 }}>
                  {status.ollama.available ? 'Corriendo' : 'No disponible'}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 0' }}>
                <StatusDot ok={status.ollama.gemma3_4b} />
                <span style={{ flex: 1, fontSize: 13 }}>Modelo gemma3:4b</span>
                <span style={{ fontSize: 11, color: status.ollama.gemma3_4b ? 'var(--success)' : 'var(--text-muted)', fontWeight: 600 }}>
                  {status.ollama.gemma3_4b ? 'Disponible' : 'No instalado'}
                </span>
              </div>
              {!status.ollama.available && (
                <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '6px 0', lineHeight: 1.6 }}>
                  Para instalar Ollama: <a href="https://ollama.com/download" target="_blank" rel="noreferrer">ollama.com/download</a>
                  {' '}y luego ejecutar <code style={{ background: '#f1f5f9', padding: '1px 5px', borderRadius: 3 }}>ollama pull gemma3:4b</code>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Instrucciones de instalación */}
      {status && !allModelsOk && (
        <div className="card">
          <h2>Instalación Manual</h2>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>
            Si la descarga automática falla, coloca los modelos manualmente en <code>{status.model_dir}</code>:
          </p>
          <div style={{ fontSize: 12, lineHeight: 2, fontFamily: 'monospace', background: '#f8fafc', padding: '12px 14px', borderRadius: 6, border: '1px solid var(--border)' }}>
            <div># YOLOv8n (detector)</div>
            <div style={{ color: '#64748b' }}>https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx</div>
            <div style={{ marginTop: 8 }}># RTMPose-m (pose 2D) — descomprimir el ZIP</div>
            <div style={{ color: '#64748b' }}>https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip</div>
            <div style={{ marginTop: 8 }}># MotionBERT Lite (lifting 3D)</div>
            <div style={{ color: '#64748b' }}>https://huggingface.co/walterzhu/MotionBERT/resolve/main/MB_lite.onnx → guardar como motionbert_lite.onnx</div>
          </div>
        </div>
      )}
    </div>
  )
}
