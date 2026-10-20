// Traitement en cours — maquette reference/VoxLingo _ Doublage IA en cours.png
// Une fois la transcription terminée, la page affiche le texte reconnu
// (l'écran d'édition de la traduction arrive à la phase 3).
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AudioLines,
  Check,
  CircleSlash,
  Clock3,
  Download,
  FileText,
  Languages,
  Mic,
  OctagonAlert,
  RotateCcw,
  Sparkles,
  UserRoundSearch,
  Users,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { LipsIcon, Panel } from '@/components/Controls'
import { LangueTag } from '@/components/Flag'
import { Gradient, PageHeader } from '@/components/Layout'
import { OutlineButton, PrimaryButton } from '@/components/PrimaryButton'
import { Waveform } from '@/components/Waveform'
import { Result } from '@/pages/Result'
import { TranslationEditor } from '@/pages/TranslationEditor'
import { cancelJob, getJob, getSegments, getWaveform, videoUrl, type Job, type Step } from '@/lib/api'
import { formatDuree } from '@/lib/format'
import { href } from '@/lib/router'
import { cn } from '@/lib/utils'
import { cardVariants, staggerVariants, voxTransition } from '@/motion/variants'

const ICONES: Record<string, (p: { className?: string }) => React.ReactNode> = {
  extraction: (p) => <AudioLines {...p} strokeWidth={1.6} />,
  separation: (p) => <Users {...p} strokeWidth={1.6} />,
  transcription: (p) => <FileText {...p} strokeWidth={1.6} />,
  locuteurs: (p) => <UserRoundSearch {...p} strokeWidth={1.6} />,
  traduction: (p) => <Languages {...p} strokeWidth={1.6} />,
  voix: (p) => <Mic {...p} strokeWidth={1.6} />,
  synchro_labiale: (p) => <LipsIcon {...p} />,
  export: (p) => <Download {...p} strokeWidth={1.6} />,
}

const enCours = (job?: Job) => job?.etat === 'en_attente' || job?.etat === 'en_cours'

/** Coche dessinée (animation.txt §F). */
function DrawnCheck({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <motion.path
        d="M5 12.5 10 17 19 7"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={voxTransition(0.4)}
      />
    </svg>
  )
}

/** Anneau lumineux de l'étape active : arc réel si le backend donne un pourcentage. */
function ActiveRing({ value }: { value: number }) {
  const r = 17
  const c = 2 * Math.PI * r
  const determine = value > 0
  return (
    <span className="relative flex size-11 items-center justify-center">
      <span className="absolute inset-0 animate-pulse rounded-full bg-violet-500/20 blur-md [animation-duration:2s]" />
      <svg viewBox="0 0 40 40" className={cn('relative size-11 -rotate-90', !determine && 'animate-spin [animation-duration:1.4s]')}>
        <defs>
          <linearGradient id="anneau-actif" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#a855f7" />
            <stop offset="100%" stopColor="#22d3ee" />
          </linearGradient>
        </defs>
        <circle cx="20" cy="20" r={r} fill="#0b0d16" stroke="#262b3a" strokeWidth="3.5" />
        <circle
          cx="20"
          cy="20"
          r={r}
          fill="none"
          stroke="url(#anneau-actif)"
          strokeWidth="3.5"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={determine ? c * (1 - value / 100) : c * 0.7}
          className="transition-[stroke-dashoffset] duration-300 ease-linear"
        />
      </svg>
    </span>
  )
}

function StepIndicator({ step, progression }: { step: Step; progression: number }) {
  switch (step.etat) {
    case 'termine':
      return (
        <span className="flex size-11 items-center justify-center">
          <span className="flex size-8 items-center justify-center rounded-full bg-success text-[#0b0d16] shadow-[0_0_14px_rgba(34,197,94,0.5)]">
            <DrawnCheck className="size-5" />
          </span>
        </span>
      )
    case 'en_cours':
      return <ActiveRing value={progression} />
    case 'erreur':
      return (
        <span className="flex size-11 items-center justify-center">
          <span className="flex size-8 items-center justify-center rounded-full bg-destructive text-white">!</span>
        </span>
      )
    default:
      return (
        <span className="flex size-11 items-center justify-center">
          <span className="size-7 rounded-full border-2 border-white/30 bg-[#0b0d16]" />
        </span>
      )
  }
}

function sousTitre(step: Step, job: Job) {
  switch (step.etat) {
    case 'termine':
      return `Terminé en ${step.duree_s} s`
    case 'en_cours':
      return job.progression_etape > 0 ? `En cours… ${job.progression_etape} %` : 'En cours…'
    case 'erreur':
      return 'Échec'
    case 'a_venir':
      return 'Prochaine phase'
    case 'ignore':
      return 'Désactivée pour ce projet'
    case 'kaggle':
      return 'À faire sur Kaggle (écran Résultat)'
    default:
      return enCours(job) ? 'En attente…' : 'Non lancée'
  }
}

function StepsTimeline({ job }: { job: Job }) {
  return (
    <Panel className="px-5 py-6">
      {job.etapes.map((step, i) => {
        const Icon = ICONES[step.cle] ?? ((p) => <Sparkles {...p} />)
        const suivant = job.etapes[i + 1]
        return (
          <div key={step.cle} className="flex gap-3">
            <div className="flex flex-col items-center">
              <StepIndicator step={step} progression={job.progression_etape} />
              {suivant && (
                <span
                  className={cn(
                    'w-0.5 flex-1 transition-colors duration-500',
                    step.etat === 'termine' ? 'bg-success' : 'bg-white/15',
                  )}
                />
              )}
            </div>
            <motion.div
              layout
              transition={voxTransition(0.3)}
              className={cn(
                'mb-2 flex min-h-[60px] flex-1 items-center gap-4 rounded-xl px-4 py-2',
                step.etat === 'en_cours' &&
                  'border-gradient shadow-[0_0_24px_rgba(124,92,255,0.25)] [--gradient-fill:#141532]',
                (step.etat === 'a_venir' || step.etat === 'ignore') && 'opacity-55',
              )}
            >
              <Icon className="size-8 shrink-0 text-white/90" />
              <div className="flex-1">
                <p className="text-[16px] font-medium text-white">{step.libelle}</p>
                <p className="text-[14px] text-white/70">{sousTitre(step, job)}</p>
              </div>
              {step.etat === 'termine' && <Check className="size-6 text-success" strokeWidth={2.5} />}
            </motion.div>
          </div>
        )
      })}
    </Panel>
  )
}

/** Temps restant estimé à partir du temps écoulé et de la progression réelle. */
function useTempsRestant(job?: Job): number | null {
  const [maintenant, setMaintenant] = useState(() => Date.now())
  const actif = enCours(job)
  useEffect(() => {
    if (!actif) return
    const timer = setInterval(() => setMaintenant(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [actif])
  if (!job || !actif || !job.debut_traitement || job.progression < 3) return null
  const ecoule = (maintenant - new Date(job.debut_traitement).getTime()) / 1000
  return Math.max(0, (ecoule * (100 - job.progression)) / job.progression)
}

function formatRestant(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.round(s % 60)
  return m > 0 ? `≈ ${m} min ${sec} s` : `≈ ${sec} s`
}

function GlobalProgress({ job }: { job: Job }) {
  const queryClient = useQueryClient()
  const restant = useTempsRestant(job)
  const annuler = useMutation({
    mutationFn: () => cancelJob(job.id),
    onSuccess: (data) => queryClient.setQueryData(['job', job.id], data),
  })
  const actif = enCours(job)

  return (
    <Panel className="flex flex-wrap items-center gap-x-8 gap-y-4 px-8 py-6">
      <p className="flex items-center gap-3 text-[18px] text-white">
        <Sparkles className="size-7 text-violet-300" strokeWidth={1.5} /> Progression globale
      </p>
      <div className="h-3.5 min-w-60 flex-1 overflow-hidden rounded-full bg-[#1e2331]">
        <motion.div
          className="h-full rounded-full bg-[linear-gradient(90deg,#a855f7,#3b82f6_70%,#22d3ee)] shadow-[0_0_14px_rgba(124,92,255,0.6)]"
          initial={false}
          animate={{ width: `${job.progression}%` }}
          transition={voxTransition(0.3)}
        />
      </div>
      <p className="w-20 text-right text-[32px] font-semibold text-white tabular-nums">{job.progression} %</p>
      {actif && (
        <>
          <span className="hidden h-12 w-px bg-white/15 lg:block" />
          <div className="flex items-center gap-3">
            <Clock3 className="size-7 text-white/90" strokeWidth={1.5} />
            <div>
              <p className="text-[14px] text-white/80">Temps restant estimé</p>
              <p className="text-[20px] font-medium text-white">
                {restant == null ? 'Calcul…' : formatRestant(restant)}
              </p>
            </div>
          </div>
          <OutlineButton
            onClick={() => annuler.mutate()}
            disabled={annuler.isPending || job.annulation_demandee}
            className="h-12 min-w-32 text-[16px]"
          >
            {job.annulation_demandee ? 'Annulation…' : 'Annuler'}
          </OutlineButton>
        </>
      )}
    </Panel>
  )
}

function formatHorodatage(s: number) {
  const t = Math.floor(s)
  const h = String(Math.floor(t / 3600)).padStart(2, '0')
  const m = String(Math.floor((t % 3600) / 60)).padStart(2, '0')
  return `${h}:${m}:${String(t % 60).padStart(2, '0')}`
}

function TranscriptView({ job, onSeek }: { job: Job; onSeek: (t: number) => void }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['segments', job.id],
    queryFn: () => getSegments(job.id),
  })

  return (
    <motion.div variants={cardVariants} initial="hidden" animate="visible">
      <Panel className="overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-6 py-4">
          <p className="flex items-center gap-3 text-[18px] font-medium text-white">
            <FileText className="size-5" /> Texte original
          </p>
          {data && (
            <span className="flex items-center gap-2 rounded-full border border-success/40 bg-success/10 px-4 py-1.5 text-[14px] text-success">
              <LangueTag langue={data.langue} /> détecté · {Math.round(data.probabilite * 100)} %
            </span>
          )}
        </div>
        {isLoading && <p className="px-6 py-4 text-white/70">Chargement…</p>}
        {error && <p className="px-6 py-4 text-destructive">{(error as Error).message}</p>}
        {data && data.segments.length === 0 && (
          <p className="px-6 py-4 text-white/70">Aucune parole détectée dans cette vidéo.</p>
        )}
        <motion.div variants={staggerVariants} initial="hidden" animate="visible">
          {data?.segments.map((seg) => (
            <motion.button
              key={seg.id}
              variants={cardVariants}
              type="button"
              onClick={() => onSeek(seg.debut)}
              className="flex w-full items-center gap-6 border-b border-white/5 px-6 py-4 text-left transition-colors duration-200 last:border-0 hover:bg-white/[0.03]"
            >
              <span className="size-3.5 shrink-0 rounded-full bg-violet-500" />
              <span className="w-20 shrink-0 text-[15px] text-white/85 tabular-nums">{formatHorodatage(seg.debut)}</span>
              <span className="text-[15px] text-white">{seg.texte}</span>
            </motion.button>
          ))}
        </motion.div>
      </Panel>
    </motion.div>
  )
}

function Titre({ job }: { job: Job }) {
  switch (job.etat) {
    case 'termine':
      return (
        <>
          Transcription <Gradient>terminée !</Gradient>
        </>
      )
    case 'erreur':
      return <>Le traitement a échoué</>
    case 'annule':
      return <>Traitement annulé</>
    default:
      return (
        <>
          Votre vidéo est en cours de <Gradient>traitement…</Gradient>
        </>
      )
  }
}

const SOUS_TITRES: Record<Job['etat'], string> = {
  en_attente: "L'IA de VoxLingo analyse, traduit et prépare votre doublage. Cela peut prendre quelques minutes.",
  en_cours: "L'IA de VoxLingo analyse, traduit et prépare votre doublage. Cela peut prendre quelques minutes.",
  a_valider: 'La traduction est prête à être relue.',
  termine: 'La langue a été détectée et chaque phrase transcrite. Cliquez sur une phrase pour la réécouter.',
  erreur: 'Une étape du traitement a rencontré un problème.',
  annule: 'Le traitement a été arrêté à votre demande.',
}

export function Project({ id, edition = false }: { id: string; edition?: boolean }) {
  const video = useRef<HTMLVideoElement>(null)
  const { data: job, error } = useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id),
    // Pendant l'habillage, le projet reste « terminé » : on suit quand même sa progression
    refetchInterval: (query) =>
      enCours(query.state.data) || query.state.data?.habillage?.etat === 'en_cours' ? 2000 : false,
  })
  const extractionFaite = job?.etapes.find((s) => s.cle === 'extraction')?.etat === 'termine'
  const { data: peaks } = useQuery({
    queryKey: ['waveform', id],
    queryFn: () => getWaveform(id),
    enabled: extractionFaite,
  })

  if (error) {
    return (
      <>
        <PageHeader back={{ page: 'accueil' }} title="Projet introuvable" />
        <p className="text-destructive">{(error as Error).message}</p>
      </>
    )
  }
  if (!job) return <PageHeader back={{ page: 'accueil' }} backLabel="Traitement en cours" title="Chargement…" />

  const traitementFini = job.etat === 'a_valider' || job.etat === 'termine'
  // Vidéo doublée prête : écran Résultat (sauf si l'on revient corriger la traduction)
  if (job.etat === 'termine' && job.fichier_sortie && !edition) {
    return <Result job={job} />
  }
  // Traduction prête, ou retour volontaire à l'édition (même après une erreur de génération des voix)
  if ((traitementFini || (edition && !enCours(job))) && job.moteur_traduction) {
    return <TranslationEditor job={job} />
  }

  const seek = (t: number) => {
    if (!video.current) return
    video.current.currentTime = t
    void video.current.play()
  }

  return (
    <>
      <PageHeader
        back={{ page: 'accueil' }}
        backLabel={enCours(job) ? 'Traitement en cours' : 'Retour'}
        align="center"
        title={<Titre job={job} />}
        subtitle={SOUS_TITRES[job.etat]}
      />

      <AnimatePresence>
        {(job.etat === 'erreur' || job.etat === 'annule') && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
              'mb-6 flex items-center gap-4 rounded-2xl border px-6 py-4',
              job.etat === 'erreur' ? 'border-destructive/40 bg-destructive/10 text-destructive' : 'border-warning/40 bg-warning/10 text-warning',
            )}
          >
            {job.etat === 'erreur' ? <OctagonAlert className="size-6 shrink-0" /> : <CircleSlash className="size-6 shrink-0" />}
            <p className="flex-1">{job.erreur ?? 'Erreur inconnue.'}</p>
            {job.moteur_traduction ? (
              <PrimaryButton href={href({ page: 'edition', id: job.id })}>
                <RotateCcw className="size-4" /> Revenir à l'édition
              </PrimaryButton>
            ) : (
              <PrimaryButton href={href({ page: 'nouveau' })}>
                <RotateCcw className="size-4" /> Recommencer
              </PrimaryButton>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mb-6 grid gap-6 xl:grid-cols-[1fr_470px]">
        <div className="flex flex-col">
          <Panel className="p-4">
            <video ref={video} src={videoUrl(job.id)} controls preload="metadata" className="aspect-video w-full rounded-xl bg-black" />
          </Panel>
          <div className="mt-5 border-b border-white/10 px-2 pb-5">
            {peaks ? (
              <Waveform peaks={peaks} videoRef={video} onSeek={(f) => seek(f * (job.duree_s ?? 0))} />
            ) : (
              <div className="h-20 animate-pulse rounded-xl bg-white/[0.03]" />
            )}
          </div>
        </div>
        <StepsTimeline job={job} />
      </div>

      <div className="mb-6">
        <GlobalProgress job={job} />
      </div>

      {job.etat === 'termine' && (
        <>
          <p className="mb-3 flex flex-wrap items-center gap-3 text-white/80">
            <LangueTag langue={job.langue_source} /> → <LangueTag langue={job.langue_cible} />
            <span className="text-white/50">· vidéo de {formatDuree(job.duree_s)}</span>
          </p>
          <TranscriptView job={job} onSeek={seek} />
        </>
      )}
    </>
  )
}
