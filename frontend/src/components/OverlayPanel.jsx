import { useEffect, useMemo, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { Line, OrthographicCamera } from '@react-three/drei'
import { useThree } from '@react-three/fiber'

const BONE_PAIRS = [
  ['nose',           'neck'],
  ['neck',           'mid_shoulder'],
  ['mid_shoulder',   'right_shoulder'],
  ['mid_shoulder',   'left_shoulder'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow',    'right_wrist'],
  ['left_shoulder',  'left_elbow'],
  ['left_elbow',     'left_wrist'],
  ['neck',           'mid_hip'],
  ['mid_hip',        'right_hip'],
  ['mid_hip',        'left_hip'],
  ['right_hip',      'right_knee'],
  ['right_knee',     'right_ankle'],
  ['left_hip',       'left_knee'],
  ['left_knee',      'left_ankle'],
]

function computeProjection(keypoints, W, H) {
  const pts = Object.values(keypoints || {}).filter(k => (k?.confidence ?? 1) > 0.15)
  if (!pts.length) return { scale: 1, ox: W / 2, oy: H / 2 }

  const ys = pts.map(p => p.y)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const height = Math.max(0.001, maxY - minY)

  const scale = (H * 0.82) / height

  const hip = keypoints?.mid_hip || keypoints?.left_hip || keypoints?.right_hip
  const hipX = hip && (hip.confidence ?? 1) > 0.1 ? hip.x : 0
  const ox = W / 2 - hipX * scale

  const ankles = []
  const la = keypoints?.left_ankle
  const ra = keypoints?.right_ankle
  if (la && (la.confidence ?? 1) > 0.1) ankles.push(la.y)
  if (ra && (ra.confidence ?? 1) > 0.1) ankles.push(ra.y)
  const avgAnkleY = ankles.length ? ankles.reduce((a, b) => a + b, 0) / ankles.length : minY
  // Match backend heuristic: oy = H*0.88 - avg_ankle_y*scale
  const oy = H * 0.88 - avgAnkleY * scale

  return { scale, ox, oy }
}

function SkeletonOverlay({ keypoints, W, H, showMesh = true, showSkeleton = true }) {
  const { scale, ox, oy } = useMemo(() => computeProjection(keypoints, W, H), [keypoints, W, H])

  const proj = (name) => {
    const kp = keypoints?.[name]
    if (!kp || (kp.confidence ?? 1) < 0.15) return null
    const x = ox + kp.x * scale
    const y = oy - kp.y * scale
    return [x, y, 0]
  }

  const lines = []
  for (const [a, b] of BONE_PAIRS) {
    const pa = proj(a)
    const pb = proj(b)
    if (!pa || !pb) continue
    lines.push({ key: `${a}-${b}`, points: [pa, pb] })
  }

  return (
    <>
      {showMesh && lines.map(l => (
        <Line
          key={`mesh-${l.key}`}
          points={l.points}
          color="#1ee3ff"
          lineWidth={10}
          transparent
          opacity={0.35}
        />
      ))}
      {showSkeleton && lines.map(l => (
        <Line
          key={`skel-${l.key}`}
          points={l.points}
          color="#00f2ff"
          lineWidth={2}
          transparent
          opacity={0.95}
        />
      ))}
    </>
  )
}

export default function OverlayPanel({
  file,
  analyzing,
  progress = 0,
  stage = '',
  currentTime = 0,
  playing = false,
  frameData,
  showSkeleton = true,
  showMesh = true,
}) {
  const videoRef = useRef()
  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl)
    }
  }, [videoUrl])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (!Number.isFinite(currentTime)) return
    if (Math.abs((v.currentTime ?? 0) - currentTime) > 0.12) v.currentTime = currentTime
  }, [currentTime])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (playing) v.play().catch(() => {})
    else v.pause()
  }, [playing])

  if (analyzing) {
    return (
      <div className="viewport-panel">
        <div className="viewport-header">
          <span className="viewport-label">SMPL_OVERLAY</span>
          <span className="viewport-badge">PROCESANDO</span>
        </div>
        <div className="viewport-body" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24 }}>
          <div className="spinner-ring" />
          <div style={{ textAlign: 'center', width: '100%', maxWidth: 260 }}>
            <div style={{ fontSize: 12, color: 'var(--primary)', marginBottom: 8, letterSpacing: '0.05em' }}>
              {stage || 'Procesando...'}
            </div>
            <div className="progress-track" style={{ width: '100%' }}>
              <div
                className="progress-fill-cyber"
                style={{ width: `${progress}%`, transition: 'width 0.4s ease' }}
              />
            </div>
            <div style={{ fontSize: 10, color: 'var(--on-surface-variant)', marginTop: 6 }}>
              {progress.toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!file) {
    return (
      <div className="viewport-panel">
        <div className="viewport-header">
          <span className="viewport-label">SMPL_OVERLAY</span>
          <span className="viewport-badge">STANDBY</span>
        </div>
        <div className="viewport-body">
          <div className="vp-overlay-empty">
            <span className="material-symbols-outlined">accessibility_new</span>
            <p>Carga un video y ejecuta el análisis</p>
          </div>
        </div>
      </div>
    )
  }

  const keypoints = frameData?.keypoints || null

  return (
    <div className="viewport-panel">
      <div className="viewport-header">
        <span className="viewport-label">SMPL_OVERLAY</span>
        <span className="viewport-badge">REALTIME</span>
      </div>
      <div className="viewport-body" style={{ position: 'relative', background: '#000' }}>
        <video
          ref={videoRef}
          className="viewport-video"
          src={videoUrl || undefined}
          muted
          loop
          playsInline
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
        <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
          <Canvas orthographic gl={{ alpha: true, antialias: true }} style={{ background: 'transparent' }}>
            {/* Screen space camera configured by ScreenSpace */}
            <ScreenSpace keypoints={keypoints} showMesh={showMesh} showSkeleton={showSkeleton} />
          </Canvas>
        </div>
      </div>
    </div>
  )
}

function ScreenSpace({ keypoints, showMesh, showSkeleton }) {
  const { size } = useThree()
  const W = Math.max(1, size.width)
  const H = Math.max(1, size.height)

  // Recreate an orthographic camera with pixel-space bounds.
  // This avoids mutating the camera object returned by hooks (lint rule).
  const camKey = `${W}x${H}`

  return (
    <group>
      <OrthographicCamera
        key={camKey}
        makeDefault
        args={[0, W, 0, H, -100, 100]}
        position={[0, 0, 10]}
      />
      {keypoints ? (
        <SkeletonOverlay
          keypoints={keypoints}
          W={W}
          H={H}
          showMesh={showMesh}
          showSkeleton={showSkeleton}
        />
      ) : null}
    </group>
  )
}
