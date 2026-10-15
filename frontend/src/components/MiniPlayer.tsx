// Mini lecteur audio des maquettes (Locuteurs, Bibliothèque) : bouton lecture, forme d'onde
// réelle (calculée dans le navigateur à partir du fichier), temps écoulé / durée.
import { Pause, Play, Volume2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { cn } from '@/lib/utils'

const BARRES = 44
const cache = new Map<string, number[]>()

/** Amplitudes réelles du fichier audio (maximum par tranche), mises en cache par adresse. */
function useAmplitudes(src: string | null) {
  const [amplitudes, setAmplitudes] = useState<number[] | null>(src ? (cache.get(src) ?? null) : null)
  useEffect(() => {
    if (!src || cache.has(src)) {
      setAmplitudes(src ? (cache.get(src) ?? null) : null)
      return
    }
    let annule = false
    void (async () => {
      try {
        const reponse = await fetch(src)
        if (!reponse.ok) throw new Error()
        const contexte = new AudioContext()
        const tampon = await contexte.decodeAudioData(await reponse.arrayBuffer())
        void contexte.close()
        const canal = tampon.getChannelData(0)
        const taille = Math.floor(canal.length / BARRES)
        const valeurs = Array.from({ length: BARRES }, (_, i) => {
          let max = 0
          for (let j = i * taille; j < (i + 1) * taille; j += 16) max = Math.max(max, Math.abs(canal[j]))
          return max
        })
        const plafond = Math.max(...valeurs) || 1
        const normalisees = valeurs.map((v) => v / plafond)
        cache.set(src, normalisees)
        if (!annule) setAmplitudes(normalisees)
      } catch {
        if (!annule) setAmplitudes(null)
      }
    })()
    return () => {
      annule = true
    }
  }, [src])
  return amplitudes
}

function formatTemps(s: number) {
  if (!Number.isFinite(s)) return '00:00'
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

export function MiniPlayer({
  src,
  couleur,
  libelle,
  vide = 'Pas encore disponible',
  className,
}: {
  src: string | null
  couleur: string
  libelle: React.ReactNode
  vide?: string
  className?: string
}) {
  const audio = useRef<HTMLAudioElement>(null)
  const [lecture, setLecture] = useState(false)
  const [temps, setTemps] = useState(0)
  const [duree, setDuree] = useState(0)
  const [erreur, setErreur] = useState(false)
  const amplitudes = useAmplitudes(erreur ? null : src)
  const disponible = !!src && !erreur

  useEffect(() => {
    setErreur(false)
    setLecture(false)
    setTemps(0)
  }, [src])

  return (
    <div className={cn('rounded-2xl border bg-[#0b0d16]/80 px-4 py-3', className)} style={{ borderColor: `${couleur}55` }}>
      <p className="mb-2 flex items-center gap-2 text-[15px] text-white">
        <span className="size-3 rounded-full" style={{ background: couleur }} /> {libelle}
      </p>
      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={!disponible}
          onClick={() => {
            const a = audio.current
            if (!a) return
            if (a.paused) void a.play()
            else a.pause()
          }}
          aria-label={lecture ? 'Pause' : 'Écouter'}
          className="flex size-10 shrink-0 items-center justify-center rounded-full border border-white/70 text-white transition-transform duration-200 hover:scale-105 disabled:opacity-30"
        >
          {lecture ? <Pause className="size-4 fill-white" /> : <Play className="ml-0.5 size-4 fill-white" />}
        </button>
        <div className="flex h-10 flex-1 items-center gap-[2px]">
          {disponible && amplitudes ? (
            amplitudes.map((a, i) => (
              <span
                key={i}
                className="flex-1 rounded-full transition-opacity duration-150"
                style={{
                  height: `${Math.max(8, a * 100)}%`,
                  background: couleur,
                  opacity: duree && i / BARRES > temps / duree ? 0.45 : 1,
                }}
              />
            ))
          ) : (
            <span className="text-[13px] text-white/50">{disponible ? 'Chargement…' : vide}</span>
          )}
        </div>
        <Volume2 className="size-4 shrink-0 text-white/70" />
      </div>
      <p className="mt-1 pl-[52px] text-[13px] text-white/75 tabular-nums">
        {formatTemps(temps)} / {formatTemps(duree)}
      </p>
      {src && (
        <audio
          ref={audio}
          src={src}
          preload="metadata"
          onLoadedMetadata={(e) => setDuree(e.currentTarget.duration)}
          onTimeUpdate={(e) => setTemps(e.currentTarget.currentTime)}
          onPlay={() => setLecture(true)}
          onPause={() => setLecture(false)}
          onEnded={() => setLecture(false)}
          onError={() => setErreur(true)}
        />
      )}
    </div>
  )
}
