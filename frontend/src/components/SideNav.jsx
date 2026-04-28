const NAV_ITEMS = [
  { icon: 'play_circle',       label: 'Análisis',  id: 'workspace' },
  { icon: 'description',       label: 'Reglas',    id: 'rules' },
  { icon: 'settings',          label: 'Setup',     id: 'setup' },
]

export default function SideNav({ activeTab, onTabChange }) {
  return (
    <aside className="side-nav">
      {NAV_ITEMS.map((item, i) => (
        <button
          key={item.id}
          className={`side-nav-btn ${activeTab === item.id ? 'active' : ''}`}
          onClick={() => onTabChange(item.id)}
          title={item.label}
        >
          <span className="material-symbols-outlined">{item.icon}</span>
          <span className="nav-label">{item.label}</span>
        </button>
      ))}
    </aside>
  )
}
