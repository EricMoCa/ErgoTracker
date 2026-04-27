import { useState, useEffect } from 'react'
import { checkHealth } from '../api'

export default function HealthBadge() {
  const [state, setState] = useState('loading') // 'loading' | 'ok' | 'error'

  useEffect(() => {
    let interval
    const check = async () => {
      try {
        await checkHealth()
        setState('ok')
      } catch {
        setState('error')
      }
    }
    check()
    interval = setInterval(check, 15000)
    return () => clearInterval(interval)
  }, [])

  const labels = { loading: 'Conectando...', ok: 'API Online', error: 'API Offline' }

  return (
    <div className="health-badge">
      <div className={`health-dot ${state}`} />
      {labels[state]}
    </div>
  )
}
