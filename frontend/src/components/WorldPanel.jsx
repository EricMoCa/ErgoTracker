import { useEffect, useRef, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import * as THREE from 'three'

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

const TRUNK_JOINTS = new Set([
  'neck','mid_shoulder','mid_hip',
  'left_hip','right_hip','left_shoulder','right_shoulder',
])

// ---------------------------------------------------------------------------
// Bone segment rendered as a cylinder between two 3D points
// ---------------------------------------------------------------------------
function BoneSegment({ start, end, isTrunk }) {
  const ref = useRef()

  const sx = start[0], sy = start[1], sz = start[2]
  const ex = end[0],   ey = end[1],   ez = end[2]
  const dx = ex - sx, dy = ey - sy, dz = ez - sz
  const len = Math.sqrt(dx*dx + dy*dy + dz*dz)
  if (len < 0.001) return null

  const midX = (sx + ex) / 2
  const midY = (sy + ey) / 2
  const midZ = (sz + ez) / 2

  // Align cylinder from (0,1,0) direction to the bone direction
  const dir = new THREE.Vector3(dx, dy, dz).normalize()
  const up  = new THREE.Vector3(0, 1, 0)
  const quaternion = new THREE.Quaternion().setFromUnitVectors(up, dir)

  const r = isTrunk ? 0.038 : 0.022

  return (
    <mesh position={[midX, midY, midZ]} quaternion={quaternion} ref={ref}>
      <cylinderGeometry args={[r, r, len, 8]} />
      <meshStandardMaterial color="#00f2ff" emissive="#003344" emissiveIntensity={0.4} />
    </mesh>
  )
}

// ---------------------------------------------------------------------------
// Full skeleton as bones + joint spheres
// ---------------------------------------------------------------------------
function Skeleton3D({ frameData }) {
  if (!frameData?.keypoints) return null
  const kps = frameData.keypoints

  function getKp(name) {
    const k = kps[name]
    return k && k.confidence > 0.1 ? k : null
  }

  const bones = []
  for (const [a, b] of BONE_PAIRS) {
    const ka = getKp(a), kb = getKp(b)
    if (!ka || !kb) continue
    const isTrunk = TRUNK_JOINTS.has(a) && TRUNK_JOINTS.has(b)
    bones.push({ key: `${a}-${b}`, a: [ka.x, ka.y, ka.z], b: [kb.x, kb.y, kb.z], isTrunk })
  }

  const joints = Object.entries(kps)
    .filter(([, kp]) => kp.confidence > 0.1)
    .map(([name, kp]) => ({ name, pos: [kp.x, kp.y, kp.z] }))

  return (
    <>
      {bones.map(bone => (
        <BoneSegment key={bone.key} start={bone.a} end={bone.b} isTrunk={bone.isTrunk} />
      ))}
      {joints.map(({ name, pos }) => {
        const isHead  = name === 'nose'
        const isMajor = ['mid_hip', 'neck', 'mid_shoulder'].includes(name)
        const r = isHead ? 0.07 : isMajor ? 0.05 : 0.035
        return (
          <mesh key={name} position={pos}>
            <sphereGeometry args={[r, 10, 7]} />
            <meshStandardMaterial
              color={isHead ? '#d0bcff' : '#00f2ff'}
              emissive={isHead ? '#280050' : '#003344'}
              emissiveIntensity={0.5}
            />
          </mesh>
        )
      })}
    </>
  )
}

// ---------------------------------------------------------------------------
// WorldPanel — always uses Three.js (installed)
// ---------------------------------------------------------------------------
export default function WorldPanel({ frameData }) {
  return (
    <div className="viewport-panel">
      <div className="viewport-header">
        <span className="viewport-label">3D_ORBIT_VIEW</span>
        <span className="viewport-badge">THREE.JS · DRAG TO ORBIT</span>
      </div>
      <div className="viewport-body" style={{ background: '#0e0e12' }}>
        <Canvas
          className="world-panel-canvas"
          camera={{ position: [2.2, 1.8, 3.0], fov: 45 }}
        >
          <color attach="background" args={['#0e0e12']} />
          <ambientLight intensity={0.35} />
          <pointLight position={[0, 3, 0]}   color="#00f2ff" intensity={1.2} />
          <pointLight position={[2, 1, 2]}   color="#d0bcff" intensity={0.5} />
          <pointLight position={[-2, 0, -1]} color="#004488" intensity={0.4} />

          <Suspense fallback={null}>
            <Grid
              args={[6, 6]}
              cellColor="#1a3a3a"
              sectionColor="rgba(0,242,255,0.18)"
              fadeDistance={20}
              position={[0, 0, 0]}
            />
          </Suspense>

          <OrbitControls
            makeDefault
            enablePan={false}
            minDistance={0.8}
            maxDistance={10}
            target={[0, 0.9, 0]}
          />

          {frameData && <Skeleton3D frameData={frameData} />}
        </Canvas>
      </div>
    </div>
  )
}
