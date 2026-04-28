import HealthBadge from './HealthBadge'

const TABS = [
  { id: 'workspace', label: 'Workspace' },
  { id: 'rules',     label: 'Reglas LLM' },
  { id: 'setup',     label: 'Setup' },
]

export default function TopNav({ activeTab, onTabChange }) {
  return (
    <header className="top-nav">
      <div className="logo">
        <div className="logo-icon">
          <span className="material-symbols-outlined">accessibility_new</span>
        </div>
        BIO-MECH ANALYTICS
      </div>

      <nav className="top-nav-tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={activeTab === t.id ? 'active' : ''}
            onClick={() => onTabChange(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="top-nav-right">
        <HealthBadge />
      </div>
    </header>
  )
}
