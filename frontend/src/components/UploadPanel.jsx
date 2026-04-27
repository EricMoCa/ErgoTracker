import { useState, useRef } from 'react'
import { startAnalysis } from '../api'

const METHODS = ['REBA', 'RULA', 'OWAS']

export default function UploadPanel({ onJobStarted }) {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [height, setHeight] = useState(170)
  const [methods, setMethods] = useState(['REBA'])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef()

  const toggleMethod = (m) =>
    setMethods((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    )

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('video/')) setFile(f)
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
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
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Cargar Video</h2>

      <div
        className={`drop-zone ${dragging ? 'dragging' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <div className="drop-icon">🎬</div>
        {file ? (
          <p className="file-selected">{file.name}</p>
        ) : (
          <p>Arrastra un video o haz clic para seleccionar<br /><small>MP4, AVI, MOV</small></p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => setFile(e.target.files[0])}
        />
      </div>

      <div className="form-group">
        <label>Altura de la persona (cm)</label>
        <input
          type="number"
          value={height}
          min={100}
          max={220}
          onChange={(e) => setHeight(Number(e.target.value))}
        />
      </div>

      <div className="form-group">
        <label>Métodos de análisis</label>
        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          {METHODS.map((m) => (
            <label key={m} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: 13 }}>
              <input
                type="checkbox"
                checked={methods.includes(m)}
                onChange={() => toggleMethod(m)}
              />
              {m}
            </label>
          ))}
        </div>
      </div>

      {error && <p className="error-msg">{error}</p>}

      <button
        className="btn btn-primary btn-full"
        onClick={handleSubmit}
        disabled={!file || methods.length === 0 || loading}
      >
        {loading ? 'Enviando...' : 'Iniciar Análisis'}
      </button>
    </div>
  )
}
