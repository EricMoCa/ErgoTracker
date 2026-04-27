import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const RISK_COLORS = {
  NEGLIGIBLE: '#22c55e',
  LOW: '#86efac',
  MEDIUM: '#fcd34d',
  HIGH: '#f97316',
  VERY_HIGH: '#ef4444',
}

const RISK_LABELS = {
  NEGLIGIBLE: 'Negligible',
  LOW: 'Bajo',
  MEDIUM: 'Medio',
  HIGH: 'Alto',
  VERY_HIGH: 'Muy Alto',
}

function buildData(frameScores) {
  const counts = {}
  for (const f of frameScores) {
    const r = f.overall_risk || 'NEGLIGIBLE'
    counts[r] = (counts[r] || 0) + 1
  }
  return Object.entries(counts)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({ name: RISK_LABELS[k] || k, value: v, key: k }))
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const { name, value } = payload[0]
  return (
    <div style={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 6, padding: '6px 10px', fontSize: 12 }}>
      {name}: {value} frames
    </div>
  )
}

export default function RiskPieChart({ frameScores }) {
  const data = buildData(frameScores)

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" outerRadius={70} dataKey="value" label={false}>
          {data.map((entry) => (
            <Cell key={entry.key} fill={RISK_COLORS[entry.key] || '#94a3b8'} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
      </PieChart>
    </ResponsiveContainer>
  )
}
