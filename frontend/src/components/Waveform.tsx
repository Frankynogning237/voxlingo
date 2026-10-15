// Forme d'onde réelle de l'audio (amplitudes calculées par le backend),
// colorée en dégradé violet → cyan. La partie déjà lue suit l'horloge du lecteur vidéo.
import { useEffect, useId, useState, type RefObject } from 'react'

import { cn } from '@/lib/utils'

export function Waveform({
  peaks,
  videoRef,
  className,
  onSeek,
}: {
  peaks: number[]
  videoRef?: RefObject<HTMLVideoElement | null>
  className?: string
  onSeek?: (fraction: number) => void
}) {
  const id = useId()
  const [lu, setLu] = useState(0)

  // Position de lecture pilotée par le lecteur vidéo (pas d'animation indépendante)
  useEffect(() => {
    const video = videoRef?.current
    if (!video) return
    const update = () => setLu(video.duration ? video.currentTime / video.duration : 0)
    video.addEventListener('timeupdate', update)
    video.addEventListener('seeked', update)
    return () => {
      video.removeEventListener('timeupdate', update)
      video.removeEventListener('seeked', update)
    }
  }, [videoRef])

  const n = peaks.length
  const largeur = n * 6

  return (
    <svg
      viewBox={`0 0 ${largeur} 100`}
      preserveAspectRatio="none"
      className={cn('h-20 w-full', onSeek && 'cursor-pointer', className)}
      onClick={(e) => {
        if (!onSeek) return
        const rect = e.currentTarget.getBoundingClientRect()
        onSeek((e.clientX - rect.left) / rect.width)
      }}
      role={onSeek ? 'slider' : 'img'}
      aria-label="Forme d'onde de l'audio"
      aria-valuenow={Math.round(lu * 100)}
    >
      <defs>
        <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="0" y1="0" x2={largeur} y2="0">
          <stop offset="0%" stopColor="#c026d3" />
          <stop offset="45%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      {peaks.map((p, i) => {
        const h = Math.max(3, p * 96)
        const joue = i / n <= lu
        return (
          <rect
            key={i}
            x={i * 6 + 1.5}
            y={50 - h / 2}
            width="3"
            height={h}
            rx="1.5"
            fill={`url(#${id})`}
            opacity={lu > 0 && !joue ? 0.45 : 1}
          />
        )
      })}
    </svg>
  )
}
