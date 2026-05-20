import { useState } from 'react'
import { extractRules, pollRulesJob } from '../api'

export default function RulesPanel() {
  const [file, setFile]               = useState(null)
  const [profileName, setProfileName] = useState('custom')
  const [loading, setLoading]         = useState(false)
  const [status, setStatus]           = useState(null)
  const [result, setResult]           = useState(null)
  const [error, setError]             = useState(null)

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('pdf', file)
      fd.append('profile_name', profileName)
      const { job_id } = await extractRules(fd)
      setStatus('running')

      let data
      while (true) {
        await new Promise(r => setTimeout(r, 2000))
        data = await pollRulesJob(job_id)
        setStatus(data.status)
        if (data.status === 'completed' || data.status === 'failed') break
      }

      if (data.status === 'completed') setResult(data.result)
      else setError(data.error || 'Error desconocido')
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="tab-content">
      <div className="cyber-card" style={{ maxWidth: 600 }}>
        <div className="cyber-card-header">
          <span className="material-symbols-outlined">psychology</span>
          Extracción de Reglas con LLM
        </div>
        <div className="cyber-card-body">
          <p style={{ fontSize: 12, color: 'var(--on-surface-variant)', lineHeight: 1.6 }}>
            Sube un PDF con normativa ergonómica y Gemma 3 4B extraerá las reglas automáticamente.
            Este proceso utiliza la GPU — no iniciar mientras hay un análisis de video en curso.
          </p>

          <div>
            <div className="cyber-label">Documento PDF</div>
            <label
              className="cyber-drop"
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) setFile(f) }}
            >
              <span className="material-symbols-outlined">upload_file</span>
              {file
                ? <span className="file-name">{file.name}</span>
                : <p>Arrastra un PDF o haz clic para seleccionar</p>
              }
              <input
                type="file"
                accept=".pdf"
                style={{ display: 'none' }}
                onChange={e => setFile(e.target.files[0])}
              />
            </label>
          </div>

          <div>
            <div className="cyber-label">Nombre del perfil</div>
            <input
              type="text"
              className="ctrl-input"
              value={profileName}
              onChange={e => setProfileName(e.target.value)}
              placeholder="ej: iso_11226"
            />
          </div>

          {error && <div className="error-banner">{error}</div>}

          {status && !result && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`job-status-badge ${status}`}>{status}</span>
              {(status === 'pending' || status === 'running') && (
                <span style={{ fontSize: 12, color: 'var(--on-surface-variant)' }}>
                  Procesando con Gemma 3...
                </span>
              )}
            </div>
          )}

          {result && (
            <div className="rules-success">
              Extracción completada: <strong>{result.rules_count}</strong> reglas guardadas en <code>{result.profile}</code>
            </div>
          )}

          <button
            className="btn-analyze"
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
              {loading ? 'hourglass_top' : 'auto_awesome'}
            </span>
            {loading ? 'Procesando con Gemma 3...' : 'Extraer Reglas'}
          </button>
        </div>
      </div>
    </div>
  )
}
