// Nouveau doublage — maquette reference/VoxLingo Neon Dubbing Interface.png
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowUp,
  Captions,
  ChevronDown,
  CircleCheck,
  Clock3,
  Database,
  Info,
  Library,
  Music,
  PenLine,
  Play,
  Scissors,
  Search,
  Settings,
  Trash2,
  Volume2,
} from 'lucide-react'
import { useEffect, useRef, useState, type DragEvent, type ReactNode } from 'react'

import { LogoMark, NeonWaves } from '@/components/Brand'
import { LipsIcon, Panel, Slider, Switch } from '@/components/Controls'
import { Flag } from '@/components/Flag'
import { GlossaireConsignes } from '@/components/GlossaireConsignes'
import { Gradient, PageHeader } from '@/components/Layout'
import { PrimaryButton } from '@/components/PrimaryButton'
import { createJob, duplicateJob, getHealth, getSettings, type JobOptions } from '@/lib/api'
import { formatDuree, formatTaille, nomLangue } from '@/lib/format'
import { navigate } from '@/lib/router'
import { cn } from '@/lib/utils'
import { badgeVariants, cardVariants, staggerVariants, voxTransition } from '@/motion/variants'

// Aucune limite de taille ni de durée : seul l'espace disque libre compte (vérifié par le serveur)
const EXTENSIONS = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.wmv', '.flv', '.ts', '.mpg', '.mpeg']
const LANGUES_VISIBLES = ['fr', 'de', 'es', 'it']

function verifierFichier(file: File): string | null {
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
  if (!EXTENSIONS.includes(ext)) return 'Format non pris en charge : utilisez MP4, MOV, MKV, WEBM, AVI, M4V, WMV, FLV, TS ou MPG.'
  return null
}

/** Bordure en pointillés dégradée violet → cyan, dessinée en SVG. */
function DashedBorder({ active }: { active: boolean }) {
  return (
    <svg className="pointer-events-none absolute inset-0 size-full" aria-hidden="true">
      <defs>
        <linearGradient id="dropzone-border" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#c026d3" />
          <stop offset="50%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <rect
        x="1.5"
        y="1.5"
        style={{ width: 'calc(100% - 3px)', height: 'calc(100% - 3px)' }}
        rx="22"
        fill="none"
        stroke="url(#dropzone-border)"
        strokeWidth={active ? 3 : 2.5}
        strokeDasharray="9 8"
        className={cn('transition-opacity duration-300', active ? 'opacity-100' : 'opacity-85')}
      />
    </svg>
  )
}

function VideoUploadIcon() {
  return (
    <div className="relative h-[100px] w-[140px]" aria-hidden="true">
      <div className="absolute top-0 left-0 flex h-[86px] w-[116px] items-center rounded-xl border-[3px] border-violet-400/90 bg-violet-500/10 shadow-[0_0_24px_rgba(139,92,246,0.45)]">
        <div className="ml-3 flex flex-col gap-2">
          {[0, 1, 2].map((i) => (
            <span key={i} className="size-2.5 rounded-[2px] bg-violet-400/90" />
          ))}
        </div>
        <Play className="ml-6 size-10 fill-violet-300 text-violet-300" />
      </div>
      <div className="absolute right-0 bottom-0 flex size-14 items-center justify-center rounded-full border-[3px] border-cyan-400 bg-[#0b1220] shadow-[0_0_20px_rgba(34,211,238,0.55)]">
        <ArrowUp className="size-7 text-cyan-300" strokeWidth={2.4} />
      </div>
    </div>
  )
}

function DropZone({ onFile, refus }: { onFile: (file: File) => void; refus: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  const [survol, setSurvol] = useState(false)

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setSurvol(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }

  return (
    <motion.button
      type="button"
      onClick={() => input.current?.click()}
      onDragEnter={(e) => {
        e.preventDefault()
        setSurvol(true)
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => setSurvol(false)}
      onDrop={onDrop}
      // Pulsation uniquement quand un fichier est glissé au-dessus (animation.txt §C)
      animate={
        refus
          ? { x: [0, -3, 3, -3, 0], scale: 1 }
          : survol
            ? { scale: [1, 1.01, 1], boxShadow: '0 0 32px rgba(124,92,255,0.28)' }
            : { scale: 1, boxShadow: '0 0 0 rgba(0,0,0,0)' }
      }
      transition={survol ? { duration: 0.4, repeat: Infinity, ease: [0.22, 1, 0.36, 1] } : { duration: 0.3 }}
      className="relative flex min-h-[290px] w-full flex-col items-center justify-center overflow-hidden rounded-3xl bg-[#0b0d16] px-8 py-10 text-center"
    >
      <NeonWaves className="absolute inset-x-0 top-1/2 h-48 w-full -translate-y-1/2 opacity-30" />
      <NeonWaves flip className="absolute inset-x-0 top-1/3 h-40 w-full opacity-20" />
      <DashedBorder active={survol} />
      <motion.div animate={{ y: survol ? -3 : 0 }} transition={voxTransition(0.25)} className="relative">
        <VideoUploadIcon />
      </motion.div>
      <p className="relative mt-7 text-[24px] font-semibold text-white">Glissez-déposez votre vidéo ici</p>
      <p className="relative mt-1 text-[17px] text-white/75">ou cliquez pour parcourir vos fichiers</p>
      <p className="relative mt-7 text-[15px] text-white/75">
        Formats supportés : MP4, MOV, AVI, MKV, WEBM… <span className="mx-1.5">•</span> Courtes ou longues vidéos
      </p>
      <input
        ref={input}
        type="file"
        accept={EXTENSIONS.join(',')}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
    </motion.button>
  )
}

type Decoupe = { debut: number; fin: number }

/** Curseur à deux poignées : début et fin du passage gardé. */
function DoubleCurseur({ duree, valeur, onChange }: { duree: number; valeur: Decoupe; onChange: (d: Decoupe) => void }) {
  const pas = duree > 600 ? 1 : 0.1
  const poignee =
    'pointer-events-none absolute inset-0 h-6 w-full appearance-none bg-transparent [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:bg-violet-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:size-4 [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-white [&::-moz-range-thumb]:bg-violet-500'
  return (
    <div className="relative h-6">
      <div className="absolute top-1/2 h-1.5 w-full -translate-y-1/2 rounded-full bg-white/12" />
      <div
        className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-linear-to-r from-violet-500 to-cyan-400"
        style={{ left: `${(valeur.debut / duree) * 100}%`, right: `${100 - (valeur.fin / duree) * 100}%` }}
      />
      <input
        type="range"
        aria-label="Début du passage gardé"
        min={0}
        max={duree}
        step={pas}
        value={valeur.debut}
        onChange={(e) => onChange({ ...valeur, debut: Math.min(Number(e.target.value), valeur.fin - 1) })}
        className={poignee}
      />
      <input
        type="range"
        aria-label="Fin du passage gardé"
        min={0}
        max={duree}
        step={pas}
        value={valeur.fin}
        onChange={(e) => onChange({ ...valeur, fin: Math.max(Number(e.target.value), valeur.debut + 1) })}
        className={poignee}
      />
    </div>
  )
}

/**
 * Option « Raccourcir la vidéo » : seul le passage choisi sera doublé (la coupe est faite par
 * le moteur au début du traitement, à l'image près).
 */
function Raccourcir({
  duree,
  decoupe,
  onChange,
  position,
  aller,
}: {
  duree: number
  decoupe: Decoupe | null
  onChange: (d: Decoupe | null) => void
  /** Position de lecture de l'aperçu, et déplacement de cette position */
  position: () => number
  aller: (t: number) => void
}) {
  if (!decoupe) {
    return (
      <button
        type="button"
        onClick={() => onChange({ debut: 0, fin: duree })}
        className="flex h-10 items-center justify-center gap-2 rounded-xl border border-white/12 text-[15px] text-white/90 transition-colors hover:border-white/30"
      >
        <Scissors className="size-4" /> Raccourcir la vidéo
      </button>
    )
  }
  const bouton = 'rounded-lg border border-white/12 px-2.5 py-1 text-[13px] text-white/85 transition-colors hover:border-white/30'
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-violet-500/40 bg-violet-500/[0.06] p-3">
      <div className="flex items-center justify-between text-[14px]">
        <span className="flex items-center gap-2 text-white">
          <Scissors className="size-4 text-violet-300" /> Passage gardé : {formatDuree(decoupe.fin - decoupe.debut)}
        </span>
        <button type="button" onClick={() => onChange(null)} className="text-white/60 hover:text-white">
          Garder toute la vidéo
        </button>
      </div>
      <DoubleCurseur
        duree={duree}
        valeur={decoupe}
        onChange={(d) => {
          // L'aperçu se place sur la poignée déplacée
          aller(d.debut !== decoupe.debut ? d.debut : d.fin)
          onChange(d)
        }}
      />
      <div className="flex items-center justify-between text-[13px] text-white/75">
        <span className="flex items-center gap-2">
          {formatDuree(decoupe.debut)}
          <button type="button" className={bouton} onClick={() => onChange({ ...decoupe, debut: Math.min(position(), decoupe.fin - 1) })}>
            Début ici
          </button>
        </span>
        <span className="flex items-center gap-2">
          <button type="button" className={bouton} onClick={() => onChange({ ...decoupe, fin: Math.max(position(), decoupe.debut + 1) })}>
            Fin ici
          </button>
          {formatDuree(decoupe.fin)}
        </span>
      </div>
    </div>
  )
}

function VideoPreview({
  file,
  onRemove,
  decoupe,
  onDecoupe,
}: {
  file: File
  onRemove: () => void
  decoupe: Decoupe | null
  onDecoupe: (d: Decoupe | null) => void
}) {
  const [url, setUrl] = useState<string>()
  const [duree, setDuree] = useState<number | null>(null)
  const [lecture, setLecture] = useState(false)
  const video = useRef<HTMLVideoElement>(null)
  // Lien temporaire vers le fichier local, libéré quand l'aperçu disparaît
  useEffect(() => {
    const objectUrl = URL.createObjectURL(file)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])

  return (
    <motion.div variants={cardVariants} initial="hidden" animate="visible" className="h-full">
      <Panel className="flex h-full flex-col gap-3 p-5">
        <div className="flex items-center justify-between">
          <p className="flex items-center gap-2.5 text-[16px] text-success">
            <CircleCheck className="size-6 fill-success text-[#0e1119]" /> Vidéo chargée avec succès !
          </p>
          <button type="button" onClick={onRemove} aria-label="Retirer la vidéo" className="text-white/80 hover:text-white">
            <Trash2 className="size-5" />
          </button>
        </div>
        <div className="relative overflow-hidden rounded-xl bg-black">
          <video
            ref={video}
            src={url}
            preload="metadata"
            controls={lecture}
            className="aspect-video w-full object-cover"
            onLoadedMetadata={(e) => {
              setDuree(e.currentTarget.duration)
              e.currentTarget.currentTime = Math.min(1, e.currentTarget.duration / 3)
            }}
            onTimeUpdate={(e) => {
              // Avec « Raccourcir la vidéo », la lecture reste dans le passage gardé
              const v = e.currentTarget
              if (decoupe && !v.paused && (v.currentTime >= decoupe.fin || v.currentTime < decoupe.debut - 0.5)) {
                v.currentTime = decoupe.debut
                if (v.currentTime >= decoupe.fin) v.pause()
              }
            }}
          />
          {!lecture && (
            <button
              type="button"
              aria-label="Lire la vidéo"
              onClick={() => {
                setLecture(true)
                void video.current?.play()
              }}
              className="absolute top-1/2 left-1/2 flex size-14 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/40 bg-black/50 backdrop-blur transition-transform duration-200 hover:scale-105"
            >
              <Play className="ml-1 size-6 fill-white text-white" />
            </button>
          )}
        </div>
        <motion.div variants={staggerVariants} initial="hidden" animate="visible">
          <motion.p variants={badgeVariants} className="truncate text-[17px] font-medium text-white" title={file.name}>
            {file.name}
          </motion.p>
          <motion.div variants={badgeVariants} className="mt-3 flex gap-10 text-[16px] text-white/85">
            <span className="flex items-center gap-2.5">
              <Clock3 className="size-5" />{' '}
              {decoupe ? (
                <>
                  {formatDuree(decoupe.fin - decoupe.debut)} <span className="text-white/50">sur {formatDuree(duree)}</span>
                </>
              ) : (
                formatDuree(duree)
              )}
            </span>
            <span className="flex items-center gap-2.5">
              <Database className="size-5" /> {formatTaille(file.size)}
            </span>
          </motion.div>
        </motion.div>
        {duree != null && Number.isFinite(duree) && duree > 2 && (
          <Raccourcir
            duree={duree}
            decoupe={decoupe}
            onChange={onDecoupe}
            position={() => video.current?.currentTime ?? 0}
            aller={(t) => {
              if (video.current) video.current.currentTime = t
            }}
          />
        )}
      </Panel>
    </motion.div>
  )
}

function EmptyPreview() {
  return (
    <Panel className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center text-white/60">
      <Play className="size-8" />
      <p>L'aperçu de votre vidéo apparaîtra ici.</p>
    </Panel>
  )
}

function DetectedLanguage() {
  return (
    <Panel className="flex h-full items-center gap-6 px-7 py-6">
      <LogoMark className="h-12" />
      <div>
        <p className="text-[17px] text-white/90">Langue détectée automatiquement</p>
        <span className="mt-3 inline-flex items-center gap-2.5 rounded-full border border-white/12 bg-white/[0.03] px-4 py-2 text-[15px] text-white/80">
          <Search className="size-4" /> Détectée pendant l'analyse
        </span>
      </div>
    </Panel>
  )
}

function LanguageChip({ code, actif, onClick }: { code: string; actif: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={actif}
      className={cn(
        'flex h-12 items-center justify-center gap-2.5 rounded-full border px-5 text-[16px] whitespace-nowrap transition-all duration-200',
        actif
          ? 'border-gradient text-white shadow-[0_0_16px_rgba(124,92,255,0.35)] [--gradient-fill:#151231]'
          : 'border-white/12 text-white/90 hover:border-white/30',
      )}
    >
      <Flag langue={code} className="text-[22px]" />
      {nomLangue(code)}
    </button>
  )
}

function LanguagePicker({
  langues,
  value,
  onChange,
  autres,
  onAutres,
}: {
  langues: string[]
  value: string
  onChange: (code: string) => void
  /** Langues supplémentaires (plusieurs langues d'un clic) ; null : une seule langue */
  autres: string[] | null
  onAutres: (codes: string[] | null) => void
}) {
  const [recherche, setRecherche] = useState('')
  const [tout, setTout] = useState(false)
  const choisir = (code: string) => {
    if (autres === null || code === value) return onChange(code)
    onAutres(autres.includes(code) ? autres.filter((c) => c !== code) : [...autres, code])
  }
  const filtre = recherche.trim().toLowerCase()
  const visibles = filtre
    ? langues.filter((code) => nomLangue(code).toLowerCase().includes(filtre) || code.includes(filtre))
    : tout
      ? langues
      : [...new Set([...LANGUES_VISIBLES, value, ...(autres ?? [])])].filter((c) => langues.includes(c))

  return (
    <Panel className="flex h-full flex-col gap-4 px-6 py-5">
      <div className="flex flex-wrap items-center gap-5">
        <p className="text-[20px] font-semibold text-white">{autres === null ? 'Langue cible' : 'Langues cibles'}</p>
        <label className="flex items-center gap-2.5 text-[14px] text-white/85" title="Un projet par langue, traités l'un après l'autre">
          <Switch checked={autres !== null} onChange={(v) => onAutres(v ? [] : null)} label="Plusieurs langues" />
          Plusieurs langues
        </label>
        <label className="relative flex-1">
          <Search className="absolute top-1/2 left-4 size-5 -translate-y-1/2 text-white/70" />
          <input
            value={recherche}
            onChange={(e) => setRecherche(e.target.value)}
            placeholder="Rechercher une langue…"
            className="h-11 w-full rounded-xl border border-white/10 bg-white/[0.03] pr-4 pl-12 text-[15px] text-white outline-none transition-[border-color,box-shadow] duration-200 placeholder:text-white/60 focus:border-violet-400/70 focus:shadow-[0_0_0_3px_rgba(124,92,255,0.15)]"
          />
        </label>
      </div>
      <div className="flex items-start gap-3">
        <div className="flex flex-1 flex-wrap items-center gap-3">
        <AnimatePresence initial={false}>
          {visibles.map((code) => (
            <motion.div key={code} layout initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={voxTransition(0.2)}>
              <LanguageChip code={code} actif={code === value || !!autres?.includes(code)} onClick={() => choisir(code)} />
            </motion.div>
          ))}
        </AnimatePresence>
        {filtre && visibles.length === 0 && <p className="text-white/60">Aucune langue ne correspond.</p>}
        </div>
        {!filtre && (
          <button
            type="button"
            onClick={() => setTout(!tout)}
            aria-label={tout ? 'Afficher moins de langues' : 'Afficher toutes les langues'}
            className="p-3 text-white/80 hover:text-white"
          >
            <ChevronDown className={cn('size-6 transition-transform duration-200', tout && 'rotate-180')} />
          </button>
        )}
      </div>
      {autres !== null && (
        <p className="text-[13px] text-white/60">
          {autres.length === 0
            ? `Cliquez sur d'autres langues pour doubler aussi la vidéo dans ces langues (en plus du ${nomLangue(value).toLowerCase()}).`
            : `${1 + autres.length} doublages : ${[value, ...autres].map((c) => nomLangue(c)).join(', ')}. Un projet par langue, traités l'un après l'autre.`}
        </p>
      )}
    </Panel>
  )
}

function OptionCard({
  icon,
  label,
  aide,
  checked,
  onChange,
}: {
  icon: ReactNode
  label: string
  aide: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <Panel className="flex items-center gap-3.5 px-5 py-4">
      <span className="text-white/90">{icon}</span>
      <span className="text-[16px] leading-snug text-white">{label}</span>
      <span title={aide} className="text-white/60">
        <Info className="size-4" />
      </span>
      <span className="ml-auto">
        <Switch checked={checked} onChange={onChange} label={label} />
      </span>
    </Panel>
  )
}

export function NewDubbing() {
  const queryClient = useQueryClient()
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: getHealth, retry: 1 })
  const [file, setFile] = useState<File | null>(null)
  // Raccourcir la vidéo : passage gardé (null = toute la vidéo)
  const [decoupe, setDecoupe] = useState<Decoupe | null>(null)
  const [langue, setLangue] = useState('fr')
  // Plusieurs langues d'un clic : langues en plus de la principale (null : une seule langue)
  const [autresLangues, setAutresLangues] = useState<string[] | null>(null)
  const [options, setOptions] = useState<JobOptions>({
    conserver_fond: true,
    synchro_labiale: false,
    sous_titres: true,
    volume_fond: 100,
    ajuster_trop_longs: true,
    strategie_duree: 'reformuler',
    voix_bibliotheque: false,
  })
  // Valeurs par défaut choisies dans la page Paramètres (appliquées une seule fois)
  const { data: parametres } = useQuery({ queryKey: ['parametres'], queryFn: getSettings, retry: 0 })
  const [defautsAppliques, setDefautsAppliques] = useState(false)
  if (parametres && !defautsAppliques) {
    setDefautsAppliques(true)
    setOptions((o) => ({
      ...o,
      conserver_fond: parametres.valeurs.conserver_fond,
      sous_titres: parametres.valeurs.sous_titres,
      volume_fond: parametres.valeurs.volume_fond,
      strategie_duree: parametres.valeurs.strategie_duree,
    }))
  }
  const [erreur, setErreur] = useState<string | null>(null)
  const [refus, setRefus] = useState(0)
  const [envoi, setEnvoi] = useState<number | null>(null)

  const setOption = <K extends keyof JobOptions>(cle: K, valeur: JobOptions[K]) =>
    setOptions((o) => ({ ...o, [cle]: valeur }))

  const choisirFichier = (f: File) => {
    const probleme = verifierFichier(f)
    setErreur(probleme)
    if (probleme) setRefus((n) => n + 1)
    else {
      setFile(f)
      setDecoupe(null)
    }
  }

  const lancer = async () => {
    if (!file) return
    setErreur(null)
    setEnvoi(0)
    try {
      const passage = decoupe ? { debut_s: Math.round(decoupe.debut * 100) / 100, fin_s: Math.round(decoupe.fin * 100) / 100 } : {}
      const job = await createJob(file, langue, { ...options, ...passage }, setEnvoi)
      // Les autres langues : même vidéo, mêmes réglages, sans renvoyer le fichier
      const autres = (autresLangues ?? []).filter((c) => c !== langue)
      for (const code of autres) await duplicateJob(job.id, code)
      await queryClient.invalidateQueries({ queryKey: ['jobs'] })
      navigate(autres.length > 0 ? { page: 'projets' } : { page: 'projet', id: job.id })
    } catch (e) {
      setErreur(e instanceof Error ? e.message : String(e))
      setEnvoi(null)
    }
  }

  return (
    <>
      <PageHeader
        back={{ page: 'accueil' }}
        title={
          <>
            Nouveau <Gradient>doublage</Gradient>
          </>
        }
        subtitle="Importez une vidéo, choisissez votre langue cible et laissez VoxLingo faire le reste."
      />

      <motion.div variants={staggerVariants} initial="hidden" animate="visible" className="flex flex-col gap-6">
        <motion.div variants={cardVariants} className="grid gap-6 xl:grid-cols-[1fr_440px]">
          <DropZone key={refus} onFile={choisirFichier} refus={refus > 0 && !!erreur} />
          {file ? (
            <VideoPreview file={file} onRemove={() => setFile(null)} decoupe={decoupe} onDecoupe={setDecoupe} />
          ) : (
            <EmptyPreview />
          )}
        </motion.div>

        <AnimatePresence>
          {erreur && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="rounded-xl border border-destructive/40 bg-destructive/10 px-5 py-3 text-destructive"
            >
              {erreur}
            </motion.p>
          )}
        </AnimatePresence>

        <motion.div variants={cardVariants} className="grid gap-6 xl:grid-cols-[340px_1fr]">
          <DetectedLanguage />
          <LanguagePicker
            langues={health?.langues_cibles ?? LANGUES_VISIBLES}
            value={langue}
            onChange={(code) => {
              setLangue(code)
              setAutresLangues((a) => a && a.filter((c) => c !== code))
            }}
            autres={autresLangues}
            onAutres={setAutresLangues}
          />
        </motion.div>

        <motion.div variants={cardVariants}>
          <p className="mb-3 flex items-center gap-3 text-[17px] font-medium text-white">
            <Settings className="size-5" /> Options avancées
          </p>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            <OptionCard
              icon={<Music className="size-6" />}
              label="Conserver le fond sonore"
              aide="Garde la musique et l'ambiance d'origine sous la nouvelle voix (phase 5)."
              checked={options.conserver_fond}
              onChange={(v) => setOption('conserver_fond', v)}
            />
            <OptionCard
              icon={<LipsIcon className="size-7" />}
              label="Synchronisation labiale"
              aide="Fait correspondre les lèvres au nouvel audio. Sans carte NVIDIA, elle passe par Kaggle (phase 8)."
              checked={options.synchro_labiale}
              onChange={(v) => setOption('synchro_labiale', v)}
            />
            <OptionCard
              icon={<Captions className="size-6" />}
              label="Générer les sous-titres"
              aide="Produit un fichier .srt dans la langue cible (phase 4)."
              checked={options.sous_titres}
              onChange={(v) => setOption('sous_titres', v)}
            />
            <OptionCard
              icon={<Scissors className="size-6" />}
              label="Ajuster les phrases trop longues"
              aide="Après la traduction, chaque phrase trop longue pour son temps est réglée automatiquement : utilisation du silence qui suit, version plus courte proposée par l'IA, ou découpage en morceaux aux pauses naturelles."
              checked={options.ajuster_trop_longs}
              onChange={(v) => setOption('ajuster_trop_longs', v)}
            />
            <OptionCard
              icon={<PenLine className="size-6" />}
              label="Reformuler plutôt qu'accélérer"
              aide="Activé : une phrase trop longue est reformulée plus brièvement par l'IA, dans le même sens, pour garder une voix au débit naturel (l'accélération ne sert qu'en dernier recours). Désactivé : la voix est d'abord accélérée, et l'IA ne reformule qu'au-delà."
              checked={options.strategie_duree === 'reformuler'}
              onChange={(v) => setOption('strategie_duree', v ? 'reformuler' : 'accelerer')}
            />
            <OptionCard
              icon={<Library className="size-6" />}
              label="Voix de la bibliothèque"
              aide="Activé : chaque personne de la vidéo reçoit automatiquement une voix différente de la bibliothèque (même sexe, registre le plus proche). Désactivé : la voix d'origine de chaque personne est clonée. Modifiable ensuite dans la page Locuteurs."
              checked={!!options.voix_bibliotheque}
              onChange={(v) => setOption('voix_bibliotheque', v)}
            />
          </div>
        </motion.div>

        <motion.div variants={cardVariants}>
          <details className="group rounded-2xl border border-white/10 bg-white/[0.02] px-5 py-4">
            <summary className="cursor-pointer list-none text-[16px] text-white marker:hidden">
              <span className="mr-2 inline-block transition-transform group-open:rotate-90">›</span>
              Glossaire et consignes de traduction <span className="text-[14px] text-white/55">(facultatif)</span>
            </summary>
            <div className="mt-4">
              <GlossaireConsignes
                glossaire={options.glossaire ?? ''}
                consignes={options.consignes ?? ''}
                onGlossaire={(v) => setOption('glossaire', v)}
                onConsignes={(v) => setOption('consignes', v)}
              />
            </div>
          </details>
        </motion.div>

        <motion.div variants={cardVariants} className="flex flex-wrap items-center gap-6 border-t border-white/5 pt-5">
          <div className={cn('flex items-center gap-4', !options.conserver_fond && 'opacity-40')}>
            <Volume2 className="size-6 text-white/90" />
            <span className="text-[17px] text-white">Volume du fond :</span>
            <div className="w-64">
              <Slider value={options.volume_fond} onChange={(v) => setOption('volume_fond', v)} label="Volume du fond" />
            </div>
            <span className="w-14 text-[17px] text-white tabular-nums">{options.volume_fond} %</span>
          </div>
          <div className="ml-auto flex items-center gap-5">
            {envoi !== null && (
              <div className="flex w-56 items-center gap-3 text-sm text-white/80">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-brand-gradient transition-[width] duration-300" style={{ width: `${Math.round(envoi * 100)}%` }} />
                </div>
                Envoi {Math.round(envoi * 100)} %
              </div>
            )}
            <PrimaryButton size="lg" disabled={!file || envoi !== null} onClick={lancer} className="min-w-80">
              <Play className="size-5 fill-white" /> Lancer l'analyse
            </PrimaryButton>
          </div>
        </motion.div>
      </motion.div>
    </>
  )
}
