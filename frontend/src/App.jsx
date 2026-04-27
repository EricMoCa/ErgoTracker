import { useState } from 'react'
import HealthBadge from './components/HealthBadge'
import UploadPanel from './components/UploadPanel'
import JobMonitor from './components/JobMonitor'
import ResultsPanel from './components/ResultsPanel'
import RulesPanel from './components/RulesPanel'
import SetupPanel from './components/SetupPanel'
import './App.css'

export default function App() {
  const [activeJob, setActiveJob] = useState(null)
  const [result, setResult] = useState(null)
  const [activeTab, setActiveTab] = useState('analysis')

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>ErgoTracker</h1>
          <span className="subtitle">Análisis Ergonómico de Video</span>
        </div>
        <HealthBadge />
      </header>

      <nav className="tab-nav">
        <button
          className={activeTab === 'analysis' ? 'active' : ''}
          onClick={() => setActiveTab('analysis')}
        >
          Análisis de Video
        </button>
        <button
          className={activeTab === 'rules' ? 'active' : ''}
          onClick={() => setActiveTab('rules')}
        >
          Extracción de Reglas LLM
        </button>
        <button
          className={activeTab === 'setup' ? 'active' : ''}
          onClick={() => setActiveTab('setup')}
        >
          Configuración
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'analysis' && (
          <div className="analysis-layout">
            <div className="left-col">
              <UploadPanel
                onJobStarted={(jobId) => {
                  setActiveJob(jobId)
                  setResult(null)
                }}
              />
              {activeJob && (
                <JobMonitor
                  jobId={activeJob}
                  onCompleted={(r) => setResult(r)}
                />
              )}
            </div>
            <div className="right-col">
              {result ? (
                <ResultsPanel result={result} />
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">📊</div>
                  <p>Sube un video para ver los resultados aquí</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'rules' && <RulesPanel />}
        {activeTab === 'setup' && <SetupPanel />}
      </main>
    </div>
  )
}
