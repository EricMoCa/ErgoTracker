import { useState, useEffect, useRef } from 'react'
import { pollJob } from '../api'

const POLL_INTERVAL = 2000

export default function JobMonitor({ jobId, onCompleted }) {
  const [job, setJob] = useState(null)
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef(null)

  useEffect(() => {
    let active = true
    const start = Date.now()

    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000))
    }, 1000)

    const poll = async () => {
      while (active) {
        try {
          const data = await pollJob(jobId)
          setJob(data)
          if (data.status === 'completed') {
            if (data.result) onCompleted(data.result)
            break
          }
          if (data.status === 'failed') break
        } catch {
          // retry silently
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL))
      }
      clearInterval(timerRef.current)
    }

    poll()
    return () => {
      active = false
      clearInterval(timerRef.current)
    }
  }, [jobId, onCompleted])

  if (!job) return null

  const isRunning = job.status === 'pending' || job.status === 'running'

  return (
    <div className="job-monitor">
      <h2>Estado del Trabajo</h2>
      <div className="status-row">
        <span className={`status-badge status-${job.status}`}>{job.status}</span>
        {isRunning && <span style={{ fontSize: 12, color: '#64748b' }}>{elapsed}s</span>}
      </div>
      <div className="progress-bar">
        <div
          className={`progress-fill ${isRunning ? 'indeterminate' : ''}`}
          style={!isRunning ? { width: job.status === 'completed' ? '100%' : '0%' } : {}}
        />
      </div>
      <p className="job-id">ID: {job.job_id}</p>
      {job.error && <p className="error-msg">Error: {job.error}</p>}
    </div>
  )
}
