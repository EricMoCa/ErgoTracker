import { useState, useEffect, useRef } from 'react'
import TopNav       from './components/TopNav'
import SideNav      from './components/SideNav'
import PlaybackBar  from './components/PlaybackBar'
import VideoPanel   from './components/VideoPanel'
import OverlayPanel from './components/OverlayPanel'
import WorldPanel   from './components/WorldPanel'
import ControlSidebar from './components/ControlSidebar'
import RulesPanel   from './components/RulesPanel'
import SetupPanel   from './components/SetupPanel'
import { pollJob, getSkeletonFrames } from './api'
import './App.css'

const POLL_MS = 1500

export default function App() {
  const [activeTab, setActiveTab] = useState('workspace')

  const [file, setFile]   = useState(null)
  const videoRef           = useRef()

  const [jobId, setJobId]       = useState(null)
  const [job, setJob]           = useState(null)
  const [result, setResult]     = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stage, setStage]       = useState('')

  // Skeleton frames for 3D viewer
  const [skeletonData, setSkeletonData] = useState(null) // {fps,total_frames,coordinate_system,frames}

  // Playback
  const [currentFrame, setCurrentFrame]   = useState(0)
  const [currentTime, setCurrentTime]     = useState(0)
  const [duration, setDuration]           = useState(0)
  const [playing, setPlaying]             = useState(false)

  // Visualization toggles
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [showTrail, setShowTrail] = useState(true)
  const [showRisk, setShowRisk] = useState(true)
  const [showMesh, setShowMesh] = useState(true)

  const skeletonFrames = skeletonData?.frames || null
  const skeletonFps = skeletonData?.fps || 0

  // Job polling
  useEffect(() => {
    if (!jobId) return
    let active = true

    const poll = async () => {
      while (active) {
        try {
          const data = await pollJob(jobId)
          setJob(data)
          setProgress(data.progress ?? 0)
          setStage(data.stage ?? '')

          if (data.status === 'completed') {
            if (data.result) {
              setResult(data.result)
              setCurrentFrame(0)
              // Fetch skeleton JSON for 3D viewer
              try {
                const skel = await getSkeletonFrames(data.result.id)
                setSkeletonData(skel || null)
              } catch {
                setSkeletonData(null)
              }
            }
            setAnalyzing(false)
            setProgress(100)
            break
          }
          if (data.status === 'failed') {
            setAnalyzing(false)
            break
          }
        } catch {
          // retry silently
        }
        await new Promise(r => setTimeout(r, POLL_MS))
      }
    }

    poll()
    return () => { active = false }
  }, [jobId])

  const handleJobStarted = (id) => {
    setJobId(id)
    setResult(null)
    setJob(null)
    setSkeletonData(null)
    setCurrentFrame(0)
    setCurrentTime(0)
    setDuration(0)
    setPlaying(false)
    setAnalyzing(true)
    setProgress(0)
    setStage('Iniciando...')
  }

  const totalFrames = skeletonFrames?.length || result?.frame_scores?.length || 0

  const mapTimeToFrameIndex = (t) => {
    if (!skeletonFrames?.length || !skeletonFps) return 0
    const targetFrameIdx = Math.max(0, Math.round(t * skeletonFps))
    let lo = 0
    let hi = skeletonFrames.length - 1
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2)
      const midIdx = skeletonFrames[mid]?.frame_idx ?? mid
      if (midIdx < targetFrameIdx) lo = mid + 1
      else hi = mid
    }
    const a = Math.max(0, lo - 1)
    const b = lo
    const aIdx = skeletonFrames[a]?.frame_idx ?? a
    const bIdx = skeletonFrames[b]?.frame_idx ?? b
    return Math.abs(aIdx - targetFrameIdx) <= Math.abs(bIdx - targetFrameIdx) ? a : b
  }

  // Current 3D frame data for WorldPanel
  const currentFrameData = skeletonFrames?.[currentFrame]
    ? { keypoints: skeletonFrames[currentFrame].keypoints }
    : null

  const trailPoints = (() => {
    if (!skeletonFrames?.length) return []
    const start = Math.max(0, currentFrame - 120)
    const pts = []
    for (let i = start; i <= currentFrame; i++) {
      const kps = skeletonFrames[i]?.keypoints
      const hip = kps?.mid_hip || kps?.left_hip || kps?.right_hip
      if (hip && (hip.confidence ?? 1) > 0.1) pts.push([hip.x, hip.y, hip.z])
    }
    return pts
  })()

  return (
    <div className="app-shell">
      <TopNav activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="app-body">
        <SideNav activeTab={activeTab} onTabChange={setActiveTab} />

        <main className="app-main">
          {activeTab === 'workspace' && (
            <div className="workspace">
              <div className="viewport-grid">
                <VideoPanel
                  file={file}
                  onFileChange={setFile}
                  videoRef={videoRef}
                  onLoadedMetadata={(e) => {
                    const d = e.currentTarget?.duration
                    if (Number.isFinite(d)) setDuration(d)
                  }}
                  onTimeUpdate={(e) => {
                    const t = e.currentTarget?.currentTime
                    if (Number.isFinite(t)) {
                      setCurrentTime(t)
                      if (skeletonFrames?.length && skeletonFps) {
                        setCurrentFrame(mapTimeToFrameIndex(t))
                      }
                    }
                  }}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                />
                <OverlayPanel
                  file={file}
                  analyzing={analyzing}
                  progress={progress}
                  stage={stage}
                  currentTime={currentTime}
                  playing={playing}
                  frameData={currentFrameData}
                  showSkeleton={showSkeleton}
                  showMesh={showMesh}
                />
                <WorldPanel
                  frameData={currentFrameData}
                  showSkeleton={showSkeleton}
                  showMesh={showMesh}
                  showTrail={showTrail}
                  trailPoints={trailPoints}
                />
              </div>

              <ControlSidebar
                file={file}
                onJobStarted={handleJobStarted}
                job={job}
                result={result}
                analyzing={analyzing}
                progress={progress}
                stage={stage}
                showSkeleton={showSkeleton}
                setShowSkeleton={setShowSkeleton}
                showTrail={showTrail}
                setShowTrail={setShowTrail}
                showRisk={showRisk}
                setShowRisk={setShowRisk}
                showMesh={showMesh}
                setShowMesh={setShowMesh}
              />
            </div>
          )}

          {activeTab === 'rules' && <RulesPanel />}

          {activeTab === 'setup' && (
            <div className="tab-content">
              <SetupPanel />
            </div>
          )}
        </main>
      </div>

      <PlaybackBar
        duration={duration}
        currentTime={currentTime}
        totalFrames={totalFrames}
        currentFrame={currentFrame}
        playing={playing}
        onTogglePlaying={async (next) => {
          setPlaying(next)
          const v = videoRef.current
          if (!v) return
          try {
            if (next) await v.play()
            else v.pause()
          } catch {
            // ignore autoplay restrictions
          }
        }}
        onFrameChange={(idx) => {
          setCurrentFrame(idx)
          if (skeletonFrames?.[idx] && skeletonFps && videoRef.current) {
            const frameIdx = skeletonFrames[idx].frame_idx ?? idx
            videoRef.current.currentTime = frameIdx / skeletonFps
          }
        }}
        onSeek={(t) => {
          setCurrentTime(t)
          if (videoRef.current) videoRef.current.currentTime = t
        }}
      />
    </div>
  )
}
