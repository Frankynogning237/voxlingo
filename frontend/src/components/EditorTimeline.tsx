// Timeline de l'éditeur (maquette « Translation Editor ») : règle temporelle,
// bande d'images, piste du locuteur avec les segments traduits, forme d'onde réelle.
import { ChevronRight, Crosshair, Pause, Play, SkipBack, SkipForward, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useId, useRef, useState, type PointerEvent } from 'react'

import { Slider } from '@/components/Controls'
import type { Segment } from '@/lib/api'
import { cn } from '@/lib/utils'
import { formatTimecode } from '@/lib/useVideoClock'

export type PisteLocuteur = { code: string; nom: string; couleur: string }
const PISTE_UNIQUE: PisteLocuteur[] = [{ code: 'SPEAKER_00', nom: 'Toutes les voix', couleur: '#a855f7' }]
const PAS_REGLE = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
const LARGEUR_LIBELLES = 200
const IMAGES_BANDE = 24 // images dans filmstrip.jpg (backend/pipeline/extract.py)
const HAUTEUR_BANDE = 58

/** Images de la vidéo à leur format naturel : chaque case montre l'image du moment correspondant. */
function Filmstrip({ src, largeur }: { src: string; largeur: number }) {
  const caseL = (HAUTEUR_BANDE * 16) / 9
  const cases = Math.max(1, Math.ceil(largeur / caseL))
  return (
    <div className="flex h-[58px] overflow-hidden border-b border-white/5 bg-black/40">
      {Array.from({ length: cases }, (_, k) => {
        const image = Math.min(IMAGES_BANDE - 1, Math.floor((k * IMAGES_BANDE) / cases))
        return (
          <span
            key={k}
            className="h-full shrink-0 border-r border-black/40"
            style={{
              width: caseL,
              backgroundImage: `url(${src})`,
              backgroundSize: `${caseL * IMAGES_BANDE}px ${HAUTEUR_BANDE}px`,
              backgroundPosition: `-${image * caseL}px 0`,
            }}
          />
        )
      })}
    </div>
  )
}

function formatRegle(s: number) {
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${String(Math.round(s % 60)).padStart(2, '0')}`
}

/** Forme d'onde réelle, dessinée en vectoriel (des centaines de points pour les longues vidéos). */
function WaveTrack({ peaks }: { peaks?: number[] }) {
  const id = useId()
  if (!peaks?.length) return <div className="h-14" />
  return (
    <svg viewBox={`0 0 ${peaks.length} 100`} preserveAspectRatio="none" className="block h-14 w-full" aria-hidden="true">
      <defs>
        <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="0" y1="0" x2={peaks.length} y2="0">
          <stop offset="0%" stopColor="#a855f7" />
          <stop offset="55%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      {peaks.map((p, i) => {
        const h = Math.max(4, p * 90)
        return <rect key={i} x={i + 0.15} y={50 - h / 2} width="0.7" height={h} fill={`url(#${id})`} />
      })}
    </svg>
  )
}

export function EditorTimeline({
  duree,
  temps,
  lecture,
  segments,
  locuteurs,
  traductions,
  selection,
  peaks,
  filmstrip,
  onSeek,
  onToggle,
  onSelect,
}: {
  duree: number
  temps: number
  lecture: boolean
  segments: Segment[]
  locuteurs: PisteLocuteur[]
  traductions: Record<number, string>
  selection: number | null
  peaks?: number[]
  filmstrip: string
  onSeek: (t: number) => void
  onToggle: () => void
  onSelect: (id: number) => void
}) {
  const pistes = locuteurs.length ? locuteurs : PISTE_UNIQUE
  // Au-delà de 4 locuteurs, pistes plus fines pour garder la frise compacte
  const compact = pistes.length > 4
  const [zoom, setZoom] = useState(0) // 0 = toute la vidéo visible, 100 = zoom maximal
  const piste = useRef<HTMLDivElement>(null)
  const defilement = useRef<HTMLDivElement>(null)
  // Zoom maximal : environ 20 s visibles (×4 au minimum), pour les vidéos longues comme courtes
  const zoomMax = Math.max(4, duree / 20)
  const echelle = Math.pow(zoomMax, zoom / 100)
  const [largeur, setLargeur] = useState(1000)

  useEffect(() => {
    const el = defilement.current
    if (!el) return
    const obs = new ResizeObserver(() => setLargeur(el.clientWidth))
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const largeurPiste = largeur * echelle
  const x = (t: number) => (duree > 0 ? (t / duree) * largeurPiste : 0)
  // Pas de la règle : au moins 90 px entre deux graduations
  const pas = PAS_REGLE.find((p) => (largeurPiste * p) / Math.max(duree, 0.1) >= 90) ?? 600
  const graduations = Array.from({ length: Math.floor(duree / pas) + 1 }, (_, i) => i * pas)

  // Le curseur reste visible quand la vidéo avance (zoom)
  useEffect(() => {
    const el = defilement.current
    if (!el || !lecture) return
    const pos = x(temps)
    if (pos < el.scrollLeft || pos > el.scrollLeft + el.clientWidth - 40) el.scrollLeft = pos - 40
  })

  const tempsDepuisPointeur = (e: PointerEvent) => {
    const rect = piste.current!.getBoundingClientRect()
    return Math.min(duree, Math.max(0, ((e.clientX - rect.left) / rect.width) * duree))
  }

  const segmentCourant = segments.findIndex((s) => temps >= s.debut && temps < s.fin)
  const precedent = () => {
    const avant = [...segments].reverse().find((s) => s.debut < temps - 0.3)
    onSeek(avant?.debut ?? 0)
  }
  const suivant = () => {
    const apres = segments.find((s) => s.debut > temps + 0.05)
    if (apres) onSeek(apres.debut)
  }

  return (
    <div className="mx-6 mt-4 mb-3 overflow-hidden rounded-2xl border border-border bg-card/85">
      <div className="flex items-center gap-4 px-5 py-3">
        <p className="w-56 text-[16px] text-white tabular-nums">
          {formatTimecode(temps)} <span className="text-white/60">/ {formatTimecode(duree)}</span>
        </p>
        <div className="flex flex-1 items-center justify-center gap-7 text-white">
          <button type="button" onClick={precedent} aria-label="Segment précédent" className="hover:text-violet-300">
            <SkipBack className="size-5 fill-current" />
          </button>
          <button type="button" onClick={onToggle} aria-label={lecture ? 'Pause' : 'Lecture'} className="hover:text-violet-300">
            {lecture ? <Pause className="size-6 fill-current" /> : <Play className="size-6 fill-current" />}
          </button>
          <button type="button" onClick={suivant} aria-label="Segment suivant" className="hover:text-violet-300">
            <SkipForward className="size-5 fill-current" />
          </button>
          <button
            type="button"
            onClick={() => segmentCourant >= 0 && onSelect(segments[segmentCourant].id)}
            aria-label="Sélectionner le segment sous le curseur"
            title="Sélectionner le segment sous le curseur"
            className="hover:text-violet-300"
          >
            <Crosshair className="size-5" />
          </button>
        </div>
        <div className="flex w-56 items-center gap-3 text-white/80">
          <ZoomOut className="size-5" />
          <Slider value={zoom} onChange={setZoom} label="Zoom de la timeline" />
          <ZoomIn className="size-5" />
        </div>
      </div>

      {/* Beaucoup de locuteurs : pistes plus fines, et la zone défile verticalement au lieu de pousser
          les boutons du bas (« Valider et générer les voix ») hors de l'écran */}
      <div className="flex max-h-[38svh] overflow-y-auto">
        {/* Libellés des pistes */}
        <div className="shrink-0 border-r border-white/5" style={{ width: LARGEUR_LIBELLES }}>
          <div className="h-[86px]" />
          {pistes.map((loc) => (
            <div key={loc.code} className={cn('flex items-center gap-3 px-4 text-white', compact ? 'h-7 text-[13px]' : 'h-11 text-[15px]')}>
              <span className={cn('shrink-0 rounded-full', compact ? 'size-3.5' : 'size-5')} style={{ background: loc.couleur }} />
              <span className="truncate">{loc.nom}</span>
              <ChevronRight className="ml-auto size-4 shrink-0 text-white/60" />
            </div>
          ))}
          <div className="h-14" />
        </div>

        {/* Pistes, défilables horizontalement quand on zoome */}
        <div ref={defilement} className={cn('relative min-w-0 flex-1 overflow-y-hidden', zoom > 0 ? 'overflow-x-auto' : 'overflow-x-hidden')}>
          <div
            ref={piste}
            className="relative cursor-pointer select-none"
            style={{ width: largeurPiste }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              onSeek(tempsDepuisPointeur(e))
            }}
            onPointerMove={(e) => {
              if (e.buttons === 1) onSeek(tempsDepuisPointeur(e))
            }}
          >
            {/* Règle */}
            <div className="relative h-7 border-b border-white/10">
              {graduations.map((g) => (
                <span key={g} className="absolute top-1 -translate-x-1/2 text-[13px] text-white/80 tabular-nums" style={{ left: x(g) }}>
                  {formatRegle(g)}
                </span>
              ))}
            </div>
            {/* Bande d'images */}
            <Filmstrip src={filmstrip} largeur={largeurPiste} />
            {/* Une piste par locuteur */}
            {pistes.map((loc) => (
            <div key={loc.code} className={cn('relative border-b border-white/5', compact ? 'h-7' : 'h-11')}>
              {segments.filter((seg) => pistes.length === 1 || (seg.locuteur ?? 'SPEAKER_00') === loc.code).map((seg) => (
                <button
                  key={seg.id}
                  type="button"
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={() => {
                    onSelect(seg.id)
                    onSeek(seg.debut)
                  }}
                  title={traductions[seg.id] ?? seg.traduction}
                  className={cn(
                    'absolute overflow-hidden rounded-md px-2 text-left whitespace-nowrap text-white transition-[filter,box-shadow] duration-200 hover:brightness-110',
                    compact ? 'top-1 h-5 text-[11px] leading-5' : 'top-1.5 h-8 text-[13px]',
                    selection === seg.id && 'ring-2 ring-white/80',
                    seg.trop_long && 'outline-2 outline-offset-1 outline-warning',
                  )}
                  style={{
                    left: x(seg.debut) + 1,
                    width: Math.max(6, x(seg.fin) - x(seg.debut) - 3),
                    background: loc.couleur,
                  }}
                >
                  {traductions[seg.id] ?? seg.traduction}
                </button>
              ))}
            </div>
            ))}
            {/* Forme d'onde réelle */}
            <WaveTrack peaks={peaks} />
            {/* Curseur de lecture */}
            <div className="pointer-events-none absolute top-0 bottom-0 w-0.5 bg-violet-500" style={{ left: x(temps) }}>
              <span className="absolute -top-0.5 -left-[7px] h-4 w-4 rounded-sm bg-violet-500 [clip-path:polygon(0_0,100%_0,100%_60%,50%_100%,0_60%)]" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
