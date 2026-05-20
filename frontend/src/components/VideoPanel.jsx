import { useRef } from 'react'

export default function VideoPanel({
  file,
  onFileChange,
  videoRef,
  onLoadedMetadata,
  onTimeUpdate,
  onPlay,
  onPause,
}) {
  const inputRef = useRef()

  const handleDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('video/')) onFileChange(f)
  }

  return (
    <div className="viewport-panel">
      <div className="viewport-header">
        <span className="viewport-label">SRC_RGB_FEED</span>
        <span className="viewport-badge">{file ? file.name.split('.').pop().toUpperCase() : 'NO FEED'}</span>
      </div>
      <div className="viewport-body">
        {file ? (
          <video
            ref={videoRef}
            className="viewport-video"
            src={URL.createObjectURL(file)}
            muted
            loop
            playsInline
            onLoadedMetadata={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            onPlay={onPlay}
            onPause={onPause}
          />
        ) : (
          <div
            className="vp-drop-zone"
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <span className="material-symbols-outlined">video_file</span>
            <p>Arrastra un video o haz clic</p>
            <p style={{ fontSize: 11, opacity: 0.6 }}>MP4 · AVI · MOV</p>
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              style={{ display: 'none' }}
              onChange={(e) => onFileChange(e.target.files[0])}
            />
          </div>
        )}
      </div>
    </div>
  )
}
