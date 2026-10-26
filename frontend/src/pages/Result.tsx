// Résultat — maquette reference/Doublage vidéo réussi en français.png
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowLeftRight,
  AudioLines,
  ChevronsLeftRight,
  CircleCheck,
  CirclePlay,
  CirclePlus,
  Clock3,
  Download,
  ExternalLink,
  Upload,
  Captions,
  FileAudio,
  Globe,
  Languages,
  ListChecks,
  Maximize,
  Music,
  Pause,
  PencilLine,
  Play,
  TriangleAlert,
  Users,
  Volume2,
  VolumeX,
  Zap,
} from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'

import { LipsIcon, Panel } from '@/components/Controls'
import { LangueTag } from '@/components/Flag'
import { Habillage } from '@/components/Habillage'
import { Gradient, PageHeader, useHealth } from '@/components/Layout'
import { OutlineButton, PrimaryButton } from '@/components/PrimaryButton'
import {
  downloadUrl,
  dubbedUrl,
  importLipsync,
  kaggleUrl,
  lipsyncEngine,
  remixAudio,
  removeLipsync,
  subtitlesVttUrl,
  videoUrl,
  type Job,
} from '@/lib/api'
import { formatDuree } from '@/lib/format'
import { href, navigate } from '@/lib/router'
import { cn } from '@/lib/utils'
import { VOX_EASE, voxTransition } from '@/motion/variants'

type Piste = 'original' | 'double'

/** Version de la vidéo finale (habillée, synchronisée ou doublée) : change à chaque nouveau fichier. */
function versionFinale(job: Job) {
  const h = job.habillage?.etat === 'termine' ? JSON.stringify(job.habillage.reglages ?? {}) : ''
  let somme = 0
  for (const c of h) somme = (somme * 31 + c.charCodeAt(0)) | 0
  return `${job.fin_traitement ?? ''}-${job.synchro_importee ? 's' : 'd'}-${h ? somme.toString(36) : 'brut'}`
}

/** Quelques particules violet → cyan, jouées une seule fois à la première ouverture du résultat. */
function Celebration({ jobId }: { jobId: string }) {
  const [visible] = useState(() => {
    try {
      const cle = `voxlingo-celebration-${jobId}`
      if (sessionStorage.getItem(cle)) return false
      sessionStorage.setItem(cle, '1')
    } catch {
      /* stockage indisponible : on célèbre quand même */
    }
    return true
  })
  if (!visible) return null
  const particules = Array.from({ length: 22 }, (_, i) => ({
    x: (i / 22) * 100 + (i % 3) * 4 - 6,
    y: 20 + ((i * 37) % 60),
    rotation: (i * 47) % 180,
    couleur: i % 3 === 0 ? '#22d3ee' : i % 3 === 1 ? '#a855f7' : '#e45cf0',
    delai: (i % 6) * 0.05,
  }))
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 h-40 overflow-hidden" aria-hidden="true">
      {particules.map((p, i) => (
        <motion.span
          key={i}
          className="absolute h-1.5 w-3 rounded-full"
          style={{ left: `${p.x}%`, top: p.y, background: p.couleur, rotate: p.rotation }}
          initial={{ opacity: 0, y: 24, scale: 0.6 }}
          animate={{ opacity: [0, 1, 0], y: -18, scale: 1 }}
          transition={{ duration: 1.6, delay: p.delai, ease: 'easeOut' }}
        />
      ))}
    </div>
  )
}

/** Anneau qui apparaît puis coche dessinée (animation.txt §I). */
function SuccessIcon() {
  return (
    <svg viewBox="0 0 64 64" className="mx-auto mb-3 size-16" aria-hidden="true">
      <defs>
        <linearGradient id="resultat-anneau" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#a855f7" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <motion.circle
        cx="32"
        cy="32"
        r="28"
        fill="#0b0d16"
        stroke="url(#resultat-anneau)"
        strokeWidth="4"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.5, ease: VOX_EASE }}
        style={{ filter: 'drop-shadow(0 0 8px rgba(124,92,255,0.7))' }}
      />
      <motion.path
        d="M20 33 L28 41 L44 24"
        fill="none"
        stroke="white"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.4, delay: 0.35, ease: VOX_EASE }}
      />
    </svg>
  )
}

function formatTemps(s: number) {
  const m = Math.floor(s / 60)
  const sec = String(Math.floor(s % 60)).padStart(2, '0')
  return `${String(m).padStart(2, '0')}:${sec}`
}

/** Comparateur avant/après : deux lecteurs synchronisés, séparateur glissable, piste audio au choix. */
function Comparateur({ job, piste, setPiste }: { job: Job; piste: Piste; setPiste: (p: Piste) => void }) {
  const original = useRef<HTMLVideoElement>(null)
  const double = useRef<HTMLVideoElement>(null)
  const cadre = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState(50)
  const [lecture, setLecture] = useState(false)
  const [temps, setTemps] = useState(0)
  const [duree, setDuree] = useState(job.duree_s ?? 0)
  const [muet, setMuet] = useState(false)
  const [sousTitres, setSousTitres] = useState(false)

  // Le lecteur doublé est la référence ; l'original le suit
  useEffect(() => {
    const maitre = double.current
    const suiveur = original.current
    if (!maitre || !suiveur) return
    const synchroniser = () => {
      setTemps(maitre.currentTime)
      if (Math.abs(suiveur.currentTime - maitre.currentTime) > 0.15) suiveur.currentTime = maitre.currentTime
    }
    const onPlay = () => {
      setLecture(true)
      void suiveur.play()
    }
    const onPause = () => {
      setLecture(false)
      suiveur.pause()
    }
    maitre.addEventListener('timeupdate', synchroniser)
    maitre.addEventListener('seeked', synchroniser)
    maitre.addEventListener('play', onPlay)
    maitre.addEventListener('pause', onPause)
    return () => {
      maitre.removeEventListener('timeupdate', synchroniser)
      maitre.removeEventListener('seeked', synchroniser)
      maitre.removeEventListener('play', onPlay)
      maitre.removeEventListener('pause', onPause)
    }
  }, [])

  useEffect(() => {
    if (original.current) original.current.muted = muet || piste !== 'original'
    if (double.current) double.current.muted = muet || piste !== 'double'
  }, [piste, muet])

  useEffect(() => {
    const track = double.current?.textTracks[0]
    if (track) track.mode = sousTitres ? 'showing' : 'hidden'
  }, [sousTitres])

  const deplacer = (clientX: number) => {
    const rect = cadre.current!.getBoundingClientRect()
    setPosition(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)))
  }
  const basculer = () => {
    const v = double.current
    if (!v) return
    if (v.paused) void v.play()
    else v.pause()
  }

  return (
    <Panel className="overflow-hidden">
      <div className="grid grid-cols-2 gap-1.5 p-1.5">
        {(['original', 'double'] as const).map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => setPiste(p)}
            className={cn(
              'flex h-12 items-center justify-center gap-3 rounded-xl text-[17px] transition-all duration-250',
              piste === p ? 'bg-brand-gradient text-white shadow-[0_0_18px_rgba(124,92,255,0.45)]' : 'text-white/85 hover:bg-white/5',
            )}
          >
            {p === 'original' ? <Play className="size-5 fill-current" /> : <CirclePlus className="size-5" />}
            {p === 'original' ? 'Original' : 'Doublé'}
          </button>
        ))}
      </div>

      <div
        ref={cadre}
        className="relative aspect-video cursor-ew-resize touch-none bg-black select-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          deplacer(e.clientX)
        }}
        onPointerMove={(e) => e.buttons === 1 && deplacer(e.clientX)}
      >
        <video
          ref={double}
          src={dubbedUrl(job.id, versionFinale(job))}
          preload="metadata"
          crossOrigin="anonymous"
          onLoadedMetadata={(e) => setDuree(e.currentTarget.duration)}
          className="absolute inset-0 size-full object-contain"
        >
          <track kind="subtitles" srcLang={job.langue_cible} label="Sous-titres" src={subtitlesVttUrl(job.id)} />
        </video>
        <video
          ref={original}
          src={videoUrl(job.id)}
          preload="metadata"
          className="absolute inset-0 size-full object-contain"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        />
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90" style={{ left: `${position}%` }}>
          <span className="absolute top-1/2 left-1/2 flex size-14 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-black/60 backdrop-blur transition-transform duration-200 hover:scale-105">
            <ChevronsLeftRight className="size-6 text-white" />
          </span>
        </div>
        <span className="absolute top-3 left-3 rounded-md bg-black/60 px-2 py-0.5 text-[12px] text-white/90">Original</span>
        <span className="absolute top-3 right-3 rounded-md bg-black/60 px-2 py-0.5 text-[12px] text-white/90">Doublé</span>
      </div>

      <div className="px-5 pt-3 pb-4">
        <input
          type="range"
          min={0}
          max={duree || 0}
          step={0.01}
          value={temps}
          aria-label="Position de lecture"
          onChange={(e) => {
            if (double.current) double.current.currentTime = Number(e.target.value)
          }}
          style={{ ['--fill' as string]: `${duree ? (temps / duree) * 100 : 0}%` }}
          className="h-1 w-full cursor-pointer appearance-none rounded-full bg-[linear-gradient(90deg,#ffffff_var(--fill),#ffffff33_var(--fill))] [&::-webkit-slider-thumb]:size-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
        />
        <div className="mt-3 flex items-center gap-5 text-white">
          <button type="button" onClick={basculer} aria-label={lecture ? 'Pause' : 'Lecture'}>
            {lecture ? <Pause className="size-7 fill-current" /> : <Play className="size-7 fill-current" />}
          </button>
          <button type="button" onClick={() => setMuet(!muet)} aria-label={muet ? 'Activer le son' : 'Couper le son'}>
            {muet ? <VolumeX className="size-6" /> : <Volume2 className="size-6" />}
          </button>
          <span className="text-[15px] tabular-nums">
            {formatTemps(temps)} / {formatTemps(duree)}
          </span>
          <span className="flex-1" />
          <button
            type="button"
            onClick={() => setSousTitres(!sousTitres)}
            aria-pressed={sousTitres}
            aria-label="Afficher les sous-titres"
            className={cn('rounded-md p-0.5', sousTitres && 'bg-white/20')}
          >
            <Captions className="size-6" />
          </button>
          <button type="button" onClick={() => void cadre.current?.requestFullscreen()} aria-label="Plein écran">
            <Maximize className="size-6" />
          </button>
        </div>
      </div>
    </Panel>
  )
}

/** Synchronisation labiale : par le moteur s'il a un GPU (MuseTalk), sinon sur Kaggle à la main
 * (paquet à télécharger, vidéo à réimporter). */
function SynchroLabiale({ job }: { job: Job }) {
  const queryClient = useQueryClient()
  const fichier = useRef<HTMLInputElement>(null)
  const { data: sante } = useHealth()
  const parMoteur = !!sante?.modeles.synchro_labiale_locale
  const synchroniser = useMutation({
    mutationFn: () => lipsyncEngine(job.id),
    onSuccess: (j) => queryClient.setQueryData(['job', job.id], j),
  })
  const importer = useMutation({
    mutationFn: (f: File) => importLipsync(job.id, f),
    onSuccess: (j) => queryClient.setQueryData(['job', job.id], j),
  })
  const retirer = useMutation({ mutationFn: () => removeLipsync(job.id), onSuccess: (j) => queryClient.setQueryData(['job', job.id], j) })
  const etapes = [
    { n: 1, texte: 'Téléchargez le paquet (vidéo, voix doublée, carnet prêt à l\'emploi et guide).' },
    { n: 2, texte: 'Sur kaggle.com, importez le carnet et le paquet, choisissez un GPU, puis « Run All ».' },
    { n: 3, texte: 'Importez ici la vidéo « video_synchronisee.mp4 » obtenue.' },
  ]
  return (
    <Panel className="mt-6 px-6 py-5">
      <div className="flex flex-wrap items-start gap-6">
        <div className="min-w-0 flex-1 sm:min-w-72">
          <p className="flex items-center gap-3 text-[20px] font-semibold text-white">
            <LipsIcon className="size-6" /> Synchronisation labiale
            {job.synchro_importee && (
              <span className="rounded-full bg-success px-3 py-0.5 text-[13px] font-medium text-black">
                {job.synchro_moteur ? 'Appliquée par le moteur' : 'Appliquée'}
              </span>
            )}
          </p>
          <p className="mt-1 text-[14px] text-white/75">
            Fait correspondre les lèvres à la nouvelle voix, sur les passages où un visage parle.
            {parMoteur
              ? ' Votre moteur a un GPU : il le fait lui-même avec MuseTalk (quelques minutes par minute de vidéo).'
              : ' Sans carte NVIDIA, le calcul se fait gratuitement sur Kaggle (environ 30 h de GPU par semaine).'}
          </p>
          {parMoteur && (
            <div className="mt-4 flex flex-col gap-2">
              <button
                type="button"
                onClick={() => synchroniser.mutate()}
                disabled={synchroniser.isPending}
                className="flex h-12 w-full max-w-md items-center justify-center gap-2.5 rounded-2xl bg-brand-gradient text-[15px] text-white disabled:opacity-60"
              >
                <LipsIcon className="size-5" />
                {synchroniser.isPending
                  ? 'Lancement…'
                  : job.synchro_moteur && job.synchro_importee
                    ? 'Refaire la synchronisation avec le moteur'
                    : 'Synchroniser les lèvres avec le moteur'}
              </button>
              {(synchroniser.error || job.synchro_erreur) && (
                <p className="text-[13px] text-destructive">
                  {synchroniser.error ? (synchroniser.error as Error).message : `Dernier essai : ${job.synchro_erreur}`}
                </p>
              )}
              <p className="mt-2 text-[13px] text-white/55">Ou, sans GPU, avec le carnet Kaggle :</p>
            </div>
          )}
          <ol className="mt-4 flex flex-col gap-2">
            {etapes.map((e) => (
              <li key={e.n} className="flex items-start gap-3 text-[14px] text-white/90">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-violet-500/30 text-[13px] text-white">{e.n}</span>
                {e.texte}
              </li>
            ))}
          </ol>
        </div>
        <div className="flex w-80 flex-col gap-3">
          <a
            href={kaggleUrl(job.id)}
            className="flex h-12 items-center justify-center gap-2.5 rounded-2xl border border-white/20 text-[15px] text-white transition-colors hover:border-white/40"
          >
            <Download className="size-5" /> 1. Télécharger le paquet Kaggle
          </a>
          <a
            href="https://www.kaggle.com/code"
            target="_blank"
            rel="noreferrer"
            className="flex h-12 items-center justify-center gap-2.5 rounded-2xl border border-white/20 text-[15px] text-white transition-colors hover:border-white/40"
          >
            <ExternalLink className="size-5" /> 2. Ouvrir Kaggle
          </a>
          <button
            type="button"
            onClick={() => fichier.current?.click()}
            disabled={importer.isPending}
            className="flex h-12 items-center justify-center gap-2.5 rounded-2xl bg-brand-gradient text-[15px] text-white disabled:opacity-60"
          >
            <Upload className="size-5" /> {importer.isPending ? 'Import…' : '3. Importer la vidéo synchronisée'}
          </button>
          <input
            ref={fichier}
            type="file"
            accept="video/mp4"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) importer.mutate(f)
              e.target.value = ''
            }}
          />
          {job.synchro_importee && (
            <button type="button" onClick={() => retirer.mutate()} className="text-[13px] text-white/60 hover:text-white">
              Revenir à la vidéo sans synchronisation
            </button>
          )}
          {importer.error && <p className="text-[13px] text-destructive">{(importer.error as Error).message}</p>}
        </div>
      </div>
    </Panel>
  )
}

function Ligne({ icone, libelle, valeur }: { icone: ReactNode; libelle: string; valeur: ReactNode }) {
  return (
    <div className="flex items-center gap-4 py-3">
      <span className="text-white/85">{icone}</span>
      <span className="flex-1 text-[16px] text-white/85">{libelle}</span>
      <span className="text-[16px] text-white">{valeur}</span>
    </div>
  )
}

function tempsTraitement(job: Job) {
  const total = job.etapes.reduce((somme, e) => somme + (e.duree_s ?? 0), 0)
  const m = Math.floor(total / 60)
  const s = Math.round(total % 60)
  return m > 0 ? `≈ ${m} min ${s} s` : `≈ ${s} s`
}

/** Nouveau mixage du son avec les voix existantes (volume, timbre et fond calés sur l'original). */
function RefaireMixage({ job }: { job: Job }) {
  const queryClient = useQueryClient()
  const mixage = useMutation({
    mutationFn: () => remixAudio(job.id),
    onSuccess: (j) => queryClient.setQueryData(['job', job.id], j),
  })
  return (
    <>
      <button
        type="button"
        onClick={() => mixage.mutate()}
        disabled={mixage.isPending}
        title="Volume et timbre de chaque voix calés au décibel près sur la voix d'origine, fond sonore d'origine complet. Les voix ne sont pas régénérées."
        className="flex items-center justify-center gap-2 text-[14px] text-white/70 hover:text-white disabled:opacity-50"
      >
        <AudioLines className="size-4" /> {mixage.isPending ? 'Lancement…' : 'Refaire le mixage du son'}
      </button>
      {mixage.error && <p className="text-center text-[13px] text-destructive">{(mixage.error as Error).message}</p>}
    </>
  )
}

export function Result({ job }: { job: Job }) {
  const [piste, setPiste] = useState<Piste>('double')
  const sousTitres = job.options?.sous_titres ?? true
  const tropLongs = job.segments_trop_longs ?? 0

  return (
    <div className="relative">
      <Celebration jobId={job.id} />
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={voxTransition(0.4)}>
        <SuccessIcon />
        <PageHeader
          align="center"
          title={
            <>
              Votre <Gradient>doublage est prêt !</Gradient>
            </>
          }
          subtitle="Votre vidéo a été traduite et doublée avec la voix d'origine."
        />
      </motion.div>

      <AnimatePresence>
        {tropLongs > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-5 flex flex-wrap items-center gap-4 rounded-2xl border border-warning/50 bg-warning/10 px-5 py-3 text-[15px] text-warning"
          >
            <TriangleAlert className="size-5 shrink-0" />
            <p className="flex-1">
              {tropLongs} phrase{tropLongs > 1 ? 's ont dû être accélérées' : ' a dû être accélérée'} au maximum sans
              tenir entièrement dans leur temps. Raccourcissez-les dans l'éditeur : seules ces phrases seront régénérées.
            </p>
            <OutlineButton onClick={() => navigate({ page: 'edition', id: job.id })} className="border-warning/60 text-warning">
              <PencilLine className="size-4" /> Corriger
            </OutlineButton>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid gap-6 *:min-w-0 xl:grid-cols-[1fr_440px]">
        <div>
          <Comparateur job={job} piste={piste} setPiste={setPiste} />
          <div className="mx-auto mt-5 flex w-fit max-w-full flex-wrap items-center justify-center gap-x-5 gap-y-2 rounded-3xl border border-white/12 bg-white/[0.02] px-6 py-3 text-[15px] text-white">
            <span className="flex items-center gap-2">
              <CirclePlay className="size-5 text-violet-300" /> Audio actuel :
              <LangueTag langue={piste === 'original' ? job.langue_source : job.langue_cible} />
            </span>
            <button
              type="button"
              onClick={() => setPiste(piste === 'original' ? 'double' : 'original')}
              aria-label="Changer de piste audio"
              className="rounded-full p-1.5 hover:bg-white/10"
            >
              <ArrowLeftRight className="size-5" />
            </button>
            <span className="flex items-center gap-2 text-white/80">
              {piste === 'original' ? 'Audio doublé :' : 'Audio original :'}
              <LangueTag langue={piste === 'original' ? job.langue_cible : job.langue_source} />
            </span>
          </div>
        </div>

        <Panel className="flex h-fit flex-col px-6 py-5">
          <p className="mb-2 flex items-center gap-3 border-b border-white/10 pb-4 text-[20px] font-semibold text-white">
            <ListChecks className="size-6" /> Récapitulatif du projet
          </p>
          <Ligne icone={<Globe className="size-6" />} libelle="Langue de départ" valeur={<LangueTag langue={job.langue_source} />} />
          <Ligne icone={<Languages className="size-6" />} libelle="Langue cible" valeur={<LangueTag langue={job.langue_cible} />} />
          <Ligne
            icone={<Users className="size-6" />}
            libelle="Nombre de locuteurs"
            valeur={<a href={href({ page: 'locuteurs', id: job.id })} className="hover:underline">{job.nb_locuteurs ?? 1}</a>}
          />
          <Ligne icone={<Clock3 className="size-6" />} libelle="Durée de la vidéo" valeur={formatDuree(job.duree_s)} />
          <Ligne icone={<Zap className="size-6" />} libelle="Temps de traitement" valeur={tempsTraitement(job)} />

          <div className="mt-4 flex flex-col gap-3">
            <PrimaryButton href={downloadUrl(job.id, 'download')} size="lg" className="h-14 rounded-2xl">
              <Download className="size-5" /> Télécharger la vidéo ({(job.fichier_sortie ?? job.fichier).split(".").pop()?.toUpperCase()}
              {job.habillage?.etat === 'termine' ? ', habillée' : ''})
            </PrimaryButton>
            {job.habillage?.etat === 'en_cours' && (
              <p className="text-center text-[13px] text-white/60">
                Habillage en cours ({job.habillage.progression} %) : la vidéo habillée sera téléchargeable dans un instant.
              </p>
            )}
            <a
              href={sousTitres ? downloadUrl(job.id, 'subtitles') : undefined}
              aria-disabled={!sousTitres}
              title={sousTitres ? undefined : "Option « Générer les sous-titres » désactivée pour ce projet"}
              className={cn(
                'flex h-14 items-center justify-center gap-3 rounded-2xl border border-white/20 text-[16px] text-white transition-colors duration-200 hover:border-white/40',
                !sousTitres && 'pointer-events-none opacity-40',
              )}
            >
              <Download className="size-5" /> Télécharger les sous-titres (SRT)
            </a>
            <a
              href={downloadUrl(job.id, 'audio')}
              className="flex h-14 items-center justify-center gap-3 rounded-2xl border border-white/20 text-[16px] text-white transition-colors duration-200 hover:border-white/40"
            >
              <Music className="size-5" /> Télécharger l'audio seul (MP3)
            </a>
            <a
              href={href({ page: 'edition', id: job.id })}
              className="mt-1 flex items-center justify-center gap-2 text-[14px] text-white/70 hover:text-white"
            >
              <PencilLine className="size-4" /> Modifier la traduction et régénérer
            </a>
            <RefaireMixage job={job} />
          </div>
        </Panel>
      </div>

      <SynchroLabiale job={job} />
      <Habillage job={job} />

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...voxTransition(0.4), delay: 0.3 }}
        className="mt-6 flex flex-wrap items-center gap-5 rounded-2xl border border-success/40 bg-success/[0.06] px-6 py-4"
      >
        <CircleCheck className="size-10 fill-success text-[#0b0d16]" />
        <div className="flex-1">
          <p className="text-[17px] font-medium text-white">Doublage terminé avec succès !</p>
          <p className="text-[15px] text-white/80">Votre vidéo est maintenant disponible dans la langue de votre choix.</p>
        </div>
        <OutlineButton onClick={() => navigate({ page: 'nouveau' })} className="h-12 px-6 text-[15px]">
          <CirclePlus className="size-5" /> Créer un nouveau projet
        </OutlineButton>
      </motion.div>
      <p className="mt-3 flex items-center justify-center gap-2 text-[13px] text-white/50">
        <FileAudio className="size-4" />
        {job.options?.conserver_fond === false
          ? 'Fond sonore non conservé pour ce projet : seule la voix traduite est présente.'
          : `Fond sonore d'origine conservé (volume ${job.options?.volume_fond ?? 100} %).`}
      </p>
    </div>
  )
}
