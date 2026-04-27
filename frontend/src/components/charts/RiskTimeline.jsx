import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts'

const RISK_TO_NUM = { NEGLIGIBLE: 0, LOW: 1, MEDIUM: 2, HIGH: 3, VERY_HIGH: 4 }
const NUM_TO_LABEL = ['Negligible', 'Bajo', 'Medio', 'Alto', 'Muy Alto']

function buildData(frameScores) {
  return frameScores.map((f) => ({
    t: f.timestamp_s?.toFixed(1) ?? f.frame_idx,
    reba: f.reba?.total ?? null,
    rula: f.rula?.total ?? null,
    risk: RISK_TO_NUM[f.overall_risk] ?? 0,
  }))
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 6, padding: '8px 12px', fontSize: 12 }}>
      <p style={{ marginBottom: 4, color: '#64748b' }}>t = {label}s</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value}
        </p>
      ))}
    </div>
  )
}

export default function RiskTimeline({ frameScores }) {
  const data = buildData(frameScores)
  const hasReba = data.some((d) => d.reba !== null)
  const hasRula = data.some((d) => d.rula !== null)

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="t" tick={{ fontSize: 10 }} label={{ value: 'seg', position: 'insideRight', fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} domain={[0, 15]} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {hasReba && (
          <Line type="monotone" dataKey="reba" name="REBA" stroke="#2E75B6" dot={false} strokeWidth={2} connectNulls />
        )}
        {hasRula && (
          <Line type="monotone" dataKey="rula" name="RULA" stroke="#d97706" dot={false} strokeWidth={2} connectNulls />
        )}
        <ReferenceLine y={7} stroke="#ef4444" strokeDasharray="4 2" label={{ value: 'Alto', fontSize: 9, fill: '#ef4444' }} />
      </LineChart>
    </ResponsiveContainer>
  )
}
