// Horloge de lecture synchronisée sur le lecteur vidéo (animation.txt §D :
// le curseur de la timeline suit l'horloge du lecteur, pas une animation indépendante).
import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

export function useVideoClock(videoRef: RefObject<HTMLVideoElement | null>) {
  const [temps, setTemps] = useState(0)
  const [lecture, setLecture] = useState(false)
  const finSegment = useRef<number | null>(null)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    let frame = 0
    const tick = () => {
      setTemps(video.currentTime)
      // Lecture d'un seul segment : on s'arrête à sa fin
      if (finSegment.current != null && video.currentTime >= finSegment.current) {
        video.pause()
        finSegment.current = null
      }
      if (!video.paused) frame = requestAnimationFrame(tick)
    }
    const onPlay = () => {
      setLecture(true)
      frame = requestAnimationFrame(tick)
    }
    const onPause = () => {
      setLecture(false)
      cancelAnimationFrame(frame)
      setTemps(video.currentTime)
    }
    const onSeeked = () => setTemps(video.currentTime)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('seeked', onSeeked)
    return () => {
      cancelAnimationFrame(frame)
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('seeked', onSeeked)
    }
  }, [videoRef])

  const seek = useCallback(
    (t: number) => {
      const video = videoRef.current
      if (!video) return
      finSegment.current = null
      video.currentTime = Math.max(0, t)
      setTemps(video.currentTime)
    },
    [videoRef],
  )

  const toggle = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    finSegment.current = null
    if (video.paused) void video.play()
    else video.pause()
  }, [videoRef])

  /** Lit uniquement l'extrait [debut, fin]. */
  const playRange = useCallback(
    (debut: number, fin: number) => {
      const video = videoRef.current
      if (!video) return
      video.currentTime = debut
      finSegment.current = fin
      void video.play()
    },
    [videoRef],
  )

  return { temps, lecture, seek, toggle, playRange }
}

/** 12.4 → "00:00:12" */
export function formatHorodatage(s: number) {
  const t = Math.floor(s)
  const h = String(Math.floor(t / 3600)).padStart(2, '0')
  const m = String(Math.floor((t % 3600) / 60)).padStart(2, '0')
  return `${h}:${m}:${String(t % 60).padStart(2, '0')}`
}

/** 12.4 → "00:00:12:10" (images à 25 i/s), comme le compteur de la maquette. */
export function formatTimecode(s: number) {
  return `${formatHorodatage(s)}:${String(Math.floor((s % 1) * 25)).padStart(2, '0')}`
}
