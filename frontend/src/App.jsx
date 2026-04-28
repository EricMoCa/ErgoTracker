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
import { pollJob }  from './api'
import './App.css'

const POLL_MS = 2000

export default function App() {
  const [activeTab, setActiveTab] = useState('workspace')

  // Video file
  const [file, setFile]   = useState(null)
  const videoRef           = useRef()

  // Job tracking
  const [jobId, setJobId]       = useState(null)
  const [job, setJob]           = useState(null)
  const [result, setResult]     = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  // Playback
  const [currentFrame, setCurrentFrame]   = useState(0)
  const [currentTime, setCurrentTime]     = useState(0)
  const [duration, setDuration]           = useState(0)

  // Frame data for 3D viewer (derived from result)
  const frameData = result?.frame_scores?.[currentFrame]
    ? {
        keypoints: result.frame_scores[currentFrame]._skeleton_kps || null,
        person_height_cm: result.person_height_cm || 170,
      }
    : null

  // Job polling
  useEffect(() => {
    if (!jobId) return
    setAnalyzing(true)
    let active = true

    const poll = async () => {
      while (active) {
        try {
          const data = await pollJob(jobId)
          setJob({ ...data, elapsed: undefined })
          if (data.status === 'completed') {
            if (data.result) {
              setResult(data.result)
              setCurrentFrame(0)
            }
            setAnalyzing(false)
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
    setCurrentFrame(0)
  }

  const totalFrames = result?.frame_scores?.length || 0

  return (
    <div className="app-shell">
      <TopNav activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="app-body">
        <SideNav activeTab={activeTab} onTabChange={setActiveTab} />

        <main className="app-main">
          {activeTab === 'workspace' && (
            <div className="workspace">
              {/* Three viewport panels */}
              <div className="viewport-grid">
                <VideoPanel
                  file={file}
                  onFileChange={setFile}
                  videoRef={videoRef}
                />
                <OverlayPanel
                  reportId={result?.id}
                  analyzing={analyzing}
                />
                <WorldPanel frameData={frameData} />
              </div>

              {/* Right control sidebar */}
              <ControlSidebar
                file={file}
                onJobStarted={handleJobStarted}
                job={job}
                result={result}
                analyzing={analyzing}
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
        onFrameChange={setCurrentFrame}
        onSeek={(t) => {
          setCurrentTime(t)
          if (videoRef.current) videoRef.current.currentTime = t
        }}
      />
    </div>
  )
}
