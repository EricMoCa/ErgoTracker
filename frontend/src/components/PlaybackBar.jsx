function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export default function PlaybackBar({
  duration = 0,
  currentTime = 0,
  onSeek,
  totalFrames = 0,
  currentFrame = 0,
  onFrameChange,
  playing = false,
  onTogglePlaying,
}) {

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0
  const frameProgress = totalFrames > 0 ? (currentFrame / Math.max(totalFrames - 1, 1)) * 100 : 0

  const handleSlider = (e) => {
    const val = Number(e.target.value)
    if (onFrameChange) onFrameChange(Math.round(val * (totalFrames - 1) / 100))
    if (onSeek && duration > 0) onSeek((val / 100) * duration)
  }

  const effectiveProgress = totalFrames > 0 ? frameProgress : progress

  return (
    <footer className="playback-bar">
      <div className="playback-controls">
        <button className="pb-btn" title="Inicio" onClick={() => onFrameChange?.(0)}>
          <span className="material-symbols-outlined">skip_previous</span>
        </button>
        <button
          className="pb-btn play"
          onClick={() => onTogglePlaying?.(!playing)}
          title={playing ? 'Pausar' : 'Reproducir'}
        >
          <span className="material-symbols-outlined">{playing ? 'pause' : 'play_arrow'}</span>
        </button>
        <button className="pb-btn" title="Final" onClick={() => onFrameChange?.(totalFrames - 1)}>
          <span className="material-symbols-outlined">skip_next</span>
        </button>
      </div>

      <div className="playback-timeline">
        <span className="pb-time">{formatTime(currentTime)} / {formatTime(duration)}</span>
        <input
          type="range"
          className="timeline-slider"
          min={0}
          max={100}
          step={0.1}
          value={effectiveProgress}
          onChange={handleSlider}
        />
        {totalFrames > 0 && (
          <span className="pb-fps">
            Frame {currentFrame + 1}/{totalFrames}
          </span>
        )}
      </div>
    </footer>
  )
}
