import { useState } from 'react'
import { extractRules, pollRulesJob } from '../api'

export default function RulesPanel() {
  const [file, setFile] = useState(null)
  const [profileName, setProfileName] = useState('custom')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

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
        await new Promise((r) => setTimeout(r, 2000))
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
    <div className="rules-layout">
      <div className="card">
        <h2>Extracción de Reglas Ergonómicas con LLM</h2>
        <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
          Sube un PDF con normativa ergonómica y Gemma 3 4B extraerá las reglas automáticamente.
          Este proceso utiliza la GPU — no iniciar mientras hay un análisis de video en curso.
        </p>

        <div className="form-group">
          <label>Documento PDF</label>
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />
          {file && <p style={{ fontSize: 12, color: '#16a34a', marginTop: 4 }}>{file.name} seleccionado</p>}
        </div>

        <div className="form-group">
          <label>Nombre del perfil</label>
          <input
            type="text"
            value={profileName}
            onChange={(e) => setProfileName(e.target.value)}
            placeholder="ej: iso_11226"
          />
        </div>

        {error && <p className="error-msg">{error}</p>}

        {status && !result && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <span className={`status-badge status-${status}`}>{status}</span>
            {(status === 'pending' || status === 'running') && (
              <span style={{ fontSize: 12, color: '#64748b' }}>Procesando con Gemma 3...</span>
            )}
          </div>
        )}

        {result && (
          <div className="rules-result">
            Extracción completada: <strong>{result.rules_count}</strong> reglas guardadas en{' '}
            <code>{result.profile}</code>
          </div>
        )}

        <button
          className="btn btn-primary btn-full"
          style={{ marginTop: 12 }}
          onClick={handleSubmit}
          disabled={!file || loading}
        >
          {loading ? 'Procesando...' : 'Extraer Reglas'}
        </button>
      </div>
    </div>
  )
}
