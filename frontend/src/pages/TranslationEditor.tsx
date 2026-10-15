// Édition de la traduction — maquette reference/VoxLingo French Translation Editor.png
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CircleCheck,
  EllipsisVertical,
  Pause,
  Play,
  PenLine,
  RotateCcw,
  Save,
  Scissors,
  Sparkles,
  TriangleAlert,
  Languages,
  Undo2,
  UsersRound,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Panel, Switch } from '@/components/Controls'
import { EditorTimeline, type PisteLocuteur } from '@/components/EditorTimeline'
import { BottomPanel, SystemStatus, UserFooter } from '@/components/Layout'
import { PrimaryButton } from '@/components/PrimaryButton'
import {
  filmstripUrl,
  getSegments,
  getWaveform,
  resetTranslations,
  retranslate,
  saveTranslations,
  validateTranslations,
  adjustTranslations,
  type BilanAjustement,
  type StrategieDuree,
  setStrategy,
  setSegmentSpeaker,
  setSegmentTone,
  type Ton,
  videoUrl,
  type Job,
  type Segment,
  type Transcription,
} from '@/lib/api'
import { href } from '@/lib/router'
import { cn } from '@/lib/utils'
import { formatHorodatage, useVideoClock } from '@/lib/useVideoClock'
import { cardVariants, staggerVariants, voxTransition } from '@/motion/variants'

const ETAPES = ['Analyse', 'Traduction & Édition', 'Clonage vocal', 'Génération', 'Export']

/** Fil d'étapes du doublage (en-tête de la maquette). */
function Stepper({ actuelle }: { actuelle: number }) {
  return (
    <ol className="flex items-center gap-2">
      {ETAPES.map((libelle, i) => {
        const fait = i < actuelle
        const actif = i === actuelle
        return (
          <li key={libelle} className="flex items-center gap-3">
            {i > 0 && <span className={cn('h-0.5 w-5 rounded-full 2xl:w-10', i <= actuelle ? 'bg-violet-500' : 'bg-white/20')} />}
            <span
              className={cn(
                'flex size-7 items-center justify-center rounded-full text-[14px]',
                fait && 'bg-blue-500 text-white',
                actif && 'bg-violet-500 text-white shadow-[0_0_12px_rgba(139,92,246,0.7)]',
                !fait && !actif && 'border border-white/40 text-white/80',
              )}
            >
              {fait ? <Check className="size-4" strokeWidth={3} /> : i + 1}
            </span>
            <span className={cn('text-[14px] whitespace-nowrap 2xl:text-[15px]', actif ? 'font-medium text-white' : 'text-white/80')}>{libelle}</span>
          </li>
        )
      })}
    </ol>
  )
}

/** Zone de texte qui s'agrandit avec son contenu. */
function AutoTextarea({
  value,
  onChange,
  onFocus,
  label,
}: {
  value: string
  onChange: (v: string) => void
  onFocus: () => void
  label: string
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${el.scrollHeight}px`
  }, [value])
  return (
    <textarea
      ref={ref}
      rows={1}
      value={value}
      aria-label={label}
      onFocus={onFocus}
      onChange={(e) => onChange(e.target.value)}
      className="w-full resize-none overflow-hidden rounded-lg border border-transparent bg-transparent px-2 py-1 text-[15px] leading-snug text-white outline-none transition-[border-color,background-color] duration-200 hover:border-white/10 focus:border-violet-400/60 focus:bg-white/[0.03]"
    />
  )
}

/** Estimation des syllabes, identique à backend/pipeline/translate.py, pour l'alerte en direct. */
function syllabes(texte: string) {
  const cjk = texte.match(/[぀-ヿ一-鿿가-힯]/g)
  if (cjk) return cjk.length
  return Math.max(1, (texte.match(/[aeiouyàâäéèêëîïôöùûüœæ]+/gi) ?? []).length)
}

// Débit moyen de parole en syllabes/s (même table que DEBIT_SYLLABES côté serveur)
const DEBIT: Record<string, number> = {
  en: 6.19, fr: 7.18, es: 7.82, it: 6.99, de: 5.97, ja: 7.84, zh: 5.18, 'zh-cn': 5.18, pt: 7.0, nl: 6.0,
  pl: 6.5, cs: 6.5, ru: 6.5, tr: 7.0, hu: 6.5, ko: 7.0, ar: 6.5, hi: 6.5,
}
const debit = (langue?: string | null) => (langue && DEBIT[langue]) || 6.5

// Mêmes valeurs que backend/pipeline/segments.py
const FACTEUR_DEBIT_XTTS = 0.77 // XTTS parle ~23 % plus lentement que le débit moyen de la langue
const CAPACITE: Record<StrategieDuree, number> = {
  reformuler: 1.08, // voix au débit naturel, accélération imperceptible tolérée
  accelerer: 1.35 * 1.25, // XTTS ×1,35 puis calage atempo ×1,25
}

/**
 * Vrai si la voix doublée ne tient pas dans le temps disponible (durée de la phrase + silence
 * avant la suivante) sans dépasser l'accélération permise par la stratégie choisie.
 */
function tropLong(seg: Segment, traduction: string, strategie: StrategieDuree, cible?: string | null) {
  const dureeVoix = syllabes(traduction) / (debit(cible) * FACTEUR_DEBIT_XTTS)
  const disponible = Math.max((seg.fin_max ?? seg.fin) - seg.debut, 0.1)
  return dureeVoix / disponible > CAPACITE[strategie]
}

// Ton de la phrase d'origine : couleurs des badges de la maquette
const TONS: Record<Ton, { libelle: string; classe: string; aide: string }> = {
  energique: { libelle: 'Énergique', classe: 'border-violet-400/70 bg-violet-500/15 text-violet-200', aide: 'Phrase animée : voix doublée plus expressive.' },
  intense: { libelle: 'Intense', classe: 'border-orange-400/70 bg-orange-500/15 text-orange-200', aide: 'Phrase dite fort : voix doublée plus forte.' },
  calme: { libelle: 'Calme', classe: 'border-sky-400/70 bg-sky-500/15 text-sky-200', aide: 'Phrase posée : voix doublée plus calme.' },
  doux: { libelle: 'Doux', classe: 'border-teal-400/70 bg-teal-500/15 text-teal-200', aide: 'Phrase dite doucement : voix doublée plus douce.' },
  neutre: { libelle: 'Neutre', classe: 'border-white/25 bg-white/5 text-white/80', aide: 'Ton habituel du locuteur.' },
}

/** Badge du ton de la phrase (mesuré sur l'original) ; un clic permet de le corriger. */
function BadgeTon({ ton, corrige, onChange }: { ton: Ton; corrige: boolean; onChange: (t: Ton) => void }) {
  const [ouvert, setOuvert] = useState(false)
  return (
    <div className="relative">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          setOuvert(!ouvert)
        }}
        title={TONS[ton].aide + (corrige ? ' (corrigé à la main)' : ' Cliquez pour corriger.')}
        className={cn('rounded-full border px-3 py-1 text-[13px] whitespace-nowrap', TONS[ton].classe)}
      >
        {TONS[ton].libelle}
      </button>
      <AnimatePresence>
        {ouvert && (
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={voxTransition(0.2)}
            onMouseLeave={() => setOuvert(false)}
            className="absolute top-8 right-0 z-30 w-44 rounded-xl border border-white/12 bg-popover p-1.5 shadow-2xl"
          >
            {(Object.keys(TONS) as Ton[]).map((t) => (
              <button
                key={t}
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  setOuvert(false)
                  if (t !== ton) onChange(t)
                }}
                className="flex w-full items-center justify-between rounded-lg px-3 py-1.5 text-left text-[14px] text-white hover:bg-white/5"
              >
                {TONS[t].libelle}
                {t === ton && <Check className="size-4" />}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

const LIBELLES_AJUSTEMENT = {
  raccourcie: { texte: 'Reformulée', aide: "Traduction reformulée plus brièvement par l'IA, dans le même sens, pour tenir dans le temps." },
  decoupee: { texte: 'Découpée', aide: 'Phrase redécoupée aux pauses naturelles, chaque morceau calé sur son moment.' },
  a_revoir: { texte: 'À revoir', aide: 'Aucun ajustement automatique ne suffit : raccourcissez cette traduction à la main.' },
} as const

/** Bilan du dernier ajustement automatique des phrases trop longues. */
function BilanBandeau({ bilan, onClose }: { bilan: BilanAjustement; onClose: () => void }) {
  const parties = [
    bilan.silence > 0 && `${bilan.silence} tiennent grâce au silence suivant`,
    bilan.raccourcies > 0 && `${bilan.raccourcies} reformulée${bilan.raccourcies > 1 ? 's' : ''} par l'IA`,
    bilan.decoupees > 0 && `${bilan.decoupees} découpée${bilan.decoupees > 1 ? 's' : ''} en morceaux`,
  ].filter(Boolean)
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={voxTransition(0.3)}
      className="mb-4 flex items-center gap-3 rounded-xl border border-cyan-400/40 bg-cyan-400/[0.07] px-4 py-2.5 text-[14px] text-white"
    >
      <Scissors className="size-4 shrink-0 text-cyan-300" />
      <p className="flex-1">
        <span className="font-medium">Ajustement automatique :</span>{' '}
        {parties.length ? parties.join(', ') : 'aucune phrase modifiée'}.
        {bilan.a_revoir > 0 && (
          <span className="text-warning">
            {' '}
            {bilan.a_revoir} phrase{bilan.a_revoir > 1 ? 's restent' : ' reste'} à raccourcir à la main.
          </span>
        )}
      </p>
      <button type="button" onClick={onClose} aria-label="Fermer" className="text-white/70 hover:text-white">
        ✕
      </button>
    </motion.div>
  )
}

/** Pastille de couleur du locuteur ; un clic permet de corriger qui parle dans la phrase. */
function ChoixLocuteurLigne({
  seg,
  locuteurs,
  onSpeaker,
}: {
  seg: Segment
  locuteurs: PisteLocuteur[]
  onSpeaker: (code: string) => void
}) {
  const [ouvert, setOuvert] = useState(false)
  const actuel = locuteurs.find((l) => l.code === (seg.locuteur ?? 'SPEAKER_00')) ?? locuteurs[0]
  return (
    <div className="relative">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          if (locuteurs.length > 1) setOuvert(!ouvert)
        }}
        title={actuel ? actuel.nom + (locuteurs.length > 1 ? ' : cliquez pour changer de locuteur' : '') : undefined}
        aria-label="Locuteur de la phrase"
        className="block size-4 rounded-full ring-offset-2 ring-offset-[#0e1119] transition-shadow hover:ring-2 hover:ring-white/50"
        style={{ background: actuel?.couleur ?? '#a855f7' }}
      />
      <AnimatePresence>
        {ouvert && (
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={voxTransition(0.2)}
            onMouseLeave={() => setOuvert(false)}
            className="absolute top-6 left-0 z-30 w-52 rounded-xl border border-white/12 bg-popover p-1.5 shadow-2xl"
          >
            <p className="px-3 py-1 text-[12px] text-white/60">Qui parle dans cette phrase ?</p>
            {locuteurs.map((l) => (
              <button
                key={l.code}
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  setOuvert(false)
                  if (l.code !== actuel?.code) onSpeaker(l.code)
                }}
                className="flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left text-[14px] text-white hover:bg-white/5"
              >
                <span className="size-3 rounded-full" style={{ background: l.couleur }} /> {l.nom}
                {l.code === actuel?.code && <Check className="ml-auto size-4" />}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function SegmentRow({
  seg,
  langues,
  traduction,
  selectionne,
  enLecture,
  modifie,
  onChange,
  onSelect,
  onPlay,
  onRevert,
  onAdjust,
  onTone,
  original,
  originalModifie,
  onChangeOriginal,
  locuteurs,
  onSpeaker,
}: {
  seg: Segment
  langues: { source?: string | null; cible?: string | null; strategie: StrategieDuree }
  traduction: string
  selectionne: boolean
  enLecture: boolean
  modifie: boolean
  onChange: (v: string) => void
  onSelect: () => void
  onPlay: () => void
  onRevert: () => void
  onAdjust: () => void
  locuteurs: PisteLocuteur[]
  onSpeaker: (code: string) => void
  onTone: (ton: Ton) => void
  original: string
  originalModifie: boolean
  onChangeOriginal: (v: string) => void
}) {
  const [menu, setMenu] = useState(false)
  const alerte = tropLong(seg, traduction, langues.strategie, langues.cible)
  const ajustement = seg.ajustement && seg.ajustement !== 'a_revoir' && !modifie ? LIBELLES_AJUSTEMENT[seg.ajustement] : null
  return (
    <motion.div
      variants={cardVariants}
      onClick={onSelect}
      className={cn(
        'grid grid-cols-[20px_76px_1fr_20px_1fr_104px_32px_24px] items-center gap-2 border-b border-white/[0.06] px-5 py-3 transition-colors duration-200 last:border-0',
        selectionne ? 'bg-violet-500/[0.08]' : 'hover:bg-white/[0.02]',
      )}
    >
      <ChoixLocuteurLigne seg={seg} locuteurs={locuteurs} onSpeaker={onSpeaker} />
      <span className="text-[15px] text-white/90 tabular-nums">{formatHorodatage(seg.debut)}</span>
      <div className="pr-2">
        <AutoTextarea
          value={original}
          onChange={onChangeOriginal}
          onFocus={onSelect}
          label={`Texte original du segment ${seg.id + 1}`}
        />
        {(originalModifie || seg.a_retraduire) && (
          <p className="px-2 text-[12px] text-warning">Texte corrigé : pensez à retraduire</p>
        )}
        {!originalModifie && !seg.a_retraduire && seg.texte_corrige && (
          <p className="px-2 text-[12px] text-white/45">Transcription corrigée</p>
        )}
      </div>
      <ArrowRight className="size-4 text-white/60" />
      <AutoTextarea value={traduction} onChange={onChange} onFocus={onSelect} label={`Traduction du segment ${seg.id + 1}`} />
      <div className="flex justify-center">
        <AnimatePresence>
          {alerte && (
            <motion.span
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={voxTransition(0.25)}
              title={
                langues.strategie === 'reformuler'
                  ? "Lue à un débit naturel, la voix ne tiendrait pas dans son temps : « Ajuster » la fera reformuler plus brièvement. À défaut, elle sera accélérée en dernier recours."
                  : 'Même accélérée au maximum, la voix ne tiendrait pas dans son temps : utilisez « Ajuster » ou raccourcissez le texte.'
              }
              className="flex items-center gap-1.5 rounded-full border border-warning/70 bg-warning/10 px-3 py-1 text-[13px] whitespace-nowrap text-warning"
            >
              <TriangleAlert className="size-3.5" /> Trop long
            </motion.span>
          )}
          {!alerte && ajustement && (
            <motion.span
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              title={ajustement.aide}
              className="flex items-center gap-1.5 rounded-full border border-cyan-400/50 bg-cyan-400/10 px-3 py-1 text-[13px] whitespace-nowrap text-cyan-300"
            >
              <Scissors className="size-3.5" /> {ajustement.texte}
            </motion.span>
          )}
          {!alerte && !ajustement && !modifie && seg.ton && (
            <BadgeTon ton={seg.ton} corrige={!!seg.ton_corrige} onChange={onTone} />
          )}
          {!alerte && !ajustement && modifie && (
            <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-[12px] text-white/50">
              Modifié
            </motion.span>
          )}
        </AnimatePresence>
      </div>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onPlay()
        }}
        aria-label={enLecture ? 'Pause' : 'Écouter le segment original'}
        title="Écoute la phrase d'origine. La voix traduite sera générée à l'étape « Clonage vocal »."
        className="flex justify-center text-white hover:text-violet-300"
      >
        {enLecture ? <Pause className="size-6" /> : <Play className="size-6" />}
      </button>
      <div className="relative flex justify-center">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            setMenu(!menu)
          }}
          aria-label="Plus d'actions"
          className="text-white hover:text-violet-300"
        >
          <EllipsisVertical className="size-5" />
        </button>
        <AnimatePresence>
          {menu && (
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 4 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={voxTransition(0.2)}
              className="absolute top-7 right-0 z-20 w-64 rounded-xl border border-white/10 bg-popover p-1.5 shadow-xl"
              onMouseLeave={() => setMenu(false)}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onRevert()
                  setMenu(false)
                }}
                className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[14px] text-white hover:bg-white/5"
              >
                <Undo2 className="size-4" /> Revenir à la traduction de l'IA
              </button>
              <button
                type="button"
                disabled={!alerte}
                onClick={(e) => {
                  e.stopPropagation()
                  onAdjust()
                  setMenu(false)
                }}
                title={alerte ? undefined : "Cette phrase tient déjà dans son temps."}
                className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[14px] text-white hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Scissors className="size-4" /> Ajuster cette phrase
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export function TranslationEditor({ job }: { job: Job }) {
  const queryClient = useQueryClient()
  const video = useRef<HTMLVideoElement>(null)
  const clock = useVideoClock(video)
  const { data } = useQuery({ queryKey: ['segments', job.id], queryFn: () => getSegments(job.id) })
  const { data: peaks } = useQuery({ queryKey: ['waveform', job.id], queryFn: () => getWaveform(job.id) })
  const [brouillon, setBrouillon] = useState<Record<number, string>>({})
  // Corrections de la transcription d'origine, pas encore enregistrées
  const [originaux, setOriginaux] = useState<Record<number, string>>({})
  const [selection, setSelection] = useState<number | null>(null)
  const [confirmation, setConfirmation] = useState(false)
  const [message, setMessage] = useState<{ type: 'succes' | 'erreur'; texte: string } | null>(null)

  const segments = useMemo(() => data?.segments ?? [], [data])
  // Locuteurs détectés (phase 6) : une couleur et une piste par personne
  const pistes: PisteLocuteur[] = useMemo(
    () => Object.entries(data?.locuteurs ?? {}).map(([code, l]) => ({ code, nom: l.nom, couleur: l.couleur })),
    [data],
  )
  const [bilanVu, setBilanVu] = useState<string | null>(null)
  const cleBilan = job.dernier_ajustement ? `${job.fin_traitement}` : null
  const traductions = useMemo(
    () => Object.fromEntries(segments.map((s) => [s.id, brouillon[s.id] ?? s.traduction ?? ''])),
    [segments, brouillon],
  )
  const originalModifie = (s: Segment) => originaux[s.id] !== undefined && originaux[s.id].trim() !== s.texte
  const modifications = segments
    .filter((s) => (brouillon[s.id] !== undefined && brouillon[s.id] !== s.traduction) || originalModifie(s))
    .map((s) => ({ id: s.id, traduction: traductions[s.id], ...(originalModifie(s) ? { texte: originaux[s.id].trim() } : {}) }))
  // Phrases dont le texte d'origine a été corrigé : leur traduction est à refaire
  const aRetraduire = segments.filter((s) => s.a_retraduire || originalModifie(s)).map((s) => s.id)
  const strategie: StrategieDuree = job.options?.strategie_duree ?? 'reformuler'
  const langues = { source: data?.langue ?? job.langue_source, cible: job.langue_cible, strategie }
  const nbTropLongs = segments.filter((s) => tropLong(s, traductions[s.id], strategie, langues.cible)).length
  const vide = segments.some((s) => !traductions[s.id]?.trim())

  const appliquer = (nouvelles: Transcription) => {
    queryClient.setQueryData(['segments', job.id], nouvelles)
    setBrouillon({})
    setOriginaux({})
  }
  const notifier = (type: 'succes' | 'erreur', texte: string) => {
    setMessage({ type, texte })
    if (type === 'succes') setTimeout(() => setMessage(null), 3500)
  }

  const enregistrer = useMutation({
    mutationFn: () => saveTranslations(job.id, modifications),
    onSuccess: (d) => {
      appliquer(d)
      notifier('succes', 'Brouillon enregistré.')
    },
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const valider = useMutation({
    mutationFn: () =>
      validateTranslations(
        job.id,
        segments.map((s) => ({ id: s.id, traduction: traductions[s.id], ...(originalModifie(s) ? { texte: originaux[s.id].trim() } : {}) })),
      ),
    onSuccess: async (j) => {
      queryClient.setQueryData(['job', job.id], j)
      await queryClient.invalidateQueries({ queryKey: ['segments', job.id] })
      setBrouillon({})
      setOriginaux({})
      notifier('succes', 'Traduction validée : génération des voix lancée.')
    },
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const reinitialiser = useMutation({
    mutationFn: () => resetTranslations(job.id),
    onSuccess: (d) => {
      appliquer(d)
      notifier('succes', "Traductions de l'IA rétablies.")
    },
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const retraduire = useMutation({
    mutationFn: () => retranslate(job.id),
    onSuccess: (j) => {
      setBrouillon({})
      setOriginaux({})
      queryClient.setQueryData(['job', job.id], j)
      void queryClient.invalidateQueries({ queryKey: ['job', job.id] })
    },
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const retraduireCorrigees = useMutation({
    // Les corrections en cours sont enregistrées, puis seules ces phrases sont retraduites
    mutationFn: async () => {
      if (modifications.length > 0) await saveTranslations(job.id, modifications)
      return retranslate(job.id, aRetraduire)
    },
    onSuccess: (j) => {
      setBrouillon({})
      setOriginaux({})
      queryClient.setQueryData(['job', job.id], j)
      void queryClient.invalidateQueries({ queryKey: ['job', job.id] })
    },
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const ajuster = useMutation({
    // Les modifications en cours sont enregistrées avant l'ajustement
    mutationFn: async (ids?: number[]) => {
      if (modifications.length > 0) await saveTranslations(job.id, modifications)
      return adjustTranslations(job.id, ids)
    },
    onSuccess: (j) => {
      setBrouillon({})
      queryClient.setQueryData(['job', job.id], j)
      void queryClient.invalidateQueries({ queryKey: ['job', job.id] })
    },
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const changerStrategie = useMutation({
    mutationFn: (s: StrategieDuree) => setStrategy(job.id, s),
    onSuccess: (j) => {
      queryClient.setQueryData(['job', job.id], j)
      void queryClient.invalidateQueries({ queryKey: ['segments', job.id] })
    },
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const changerTon = useMutation({
    mutationFn: ({ segId, ton }: { segId: number; ton: Ton }) => setSegmentTone(job.id, segId, ton),
    onSuccess: (d) => queryClient.setQueryData(['segments', job.id], d),
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const changerLocuteur = useMutation({
    mutationFn: ({ segId, code }: { segId: number; code: string }) => setSegmentSpeaker(job.id, segId, code),
    onSuccess: (d) => queryClient.setQueryData(['segments', job.id], d),
    onError: (e) => notifier('erreur', (e as Error).message),
  })
  const occupe = retraduireCorrigees.isPending || changerStrategie.isPending || ajuster.isPending || enregistrer.isPending || valider.isPending || reinitialiser.isPending || retraduire.isPending

  // Le segment sous le curseur de lecture est mis en évidence dans le tableau
  const segmentLu = segments.find((s) => clock.temps >= s.debut && clock.temps < s.fin)
  const actif = selection ?? segmentLu?.id ?? null

  return (
    <>
      <header className="mb-5 flex items-center gap-6 border-b border-white/10 pb-5">
        <a href={href({ page: 'accueil' })} className="flex shrink-0 items-center gap-3 text-[21px] font-medium whitespace-nowrap text-white">
          <ArrowLeft className="size-6" strokeWidth={1.6} /> Édition de la traduction
        </a>
        <div className="min-w-0 flex-1 overflow-hidden">
          <Stepper actuelle={job.traduction_validee ? 2 : 1} />
        </div>
        <SystemStatus />
      </header>

      <div className="mb-4 flex items-center justify-end gap-3">
        <AnimatePresence>
          {message && (
            <motion.p
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={voxTransition(0.3)}
              className={cn(
                'mr-auto flex items-center gap-2 rounded-xl border px-4 py-2 text-[14px]',
                message.type === 'succes' ? 'border-success/40 bg-success/10 text-success' : 'border-destructive/40 bg-destructive/10 text-destructive',
              )}
            >
              {message.type === 'succes' ? <CircleCheck className="size-4" /> : <TriangleAlert className="size-4" />}
              {message.texte}
            </motion.p>
          )}
        </AnimatePresence>
        {confirmation ? (
          <span className="flex items-center gap-2 rounded-xl border border-warning/50 bg-warning/10 px-3 py-1.5 text-[14px] text-warning">
            Toutes vos modifications seront remplacées.
            <button
              type="button"
              onClick={() => {
                setConfirmation(false)
                retraduire.mutate()
              }}
              className="rounded-lg bg-warning px-3 py-1 font-medium text-black"
            >
              Retraduire
            </button>
            <button type="button" onClick={() => setConfirmation(false)} className="px-2 text-white/80 hover:text-white">
              Annuler
            </button>
          </span>
        ) : (
          <button
            type="button"
            disabled={occupe}
            onClick={() => setConfirmation(true)}
            className="flex h-11 items-center gap-2.5 rounded-xl border-gradient px-5 text-[15px] text-white transition-shadow duration-200 hover:shadow-[0_0_18px_rgba(124,92,255,0.35)] disabled:opacity-50 [--gradient-fill:#151231]"
          >
            <Sparkles className="size-5 text-violet-300" /> Traduction assistée par IA
          </button>
        )}
        <label
          className="flex h-11 items-center gap-3 rounded-xl border border-white/12 px-4 text-[14px] text-white"
          title="Activé : une phrase trop longue est reformulée plus brièvement, dans le même sens, pour garder une voix naturelle. Désactivé : la voix est d'abord accélérée."
        >
          <PenLine className="size-4 text-violet-300" /> Reformuler plutôt qu'accélérer
          <Switch
            checked={strategie === 'reformuler'}
            onChange={(v) => changerStrategie.mutate(v ? 'reformuler' : 'accelerer')}
            label="Reformuler plutôt qu'accélérer"
          />
        </label>
        {aRetraduire.length > 0 && (
          <button
            type="button"
            disabled={occupe}
            onClick={() => retraduireCorrigees.mutate()}
            title="Retraduit uniquement les phrases dont vous avez corrigé le texte d'origine (les autres traductions sont conservées)"
            className="flex h-11 items-center gap-2.5 rounded-xl border border-warning/60 bg-warning/10 px-5 text-[15px] text-white transition-colors duration-200 hover:border-warning disabled:opacity-50"
          >
            <Languages className="size-5 text-warning" /> Retraduire les phrases corrigées
            <span className="rounded-full bg-warning px-2 text-[12px] font-semibold text-black">{aRetraduire.length}</span>
          </button>
        )}
        <button
          type="button"
          disabled={occupe || nbTropLongs === 0}
          onClick={() => ajuster.mutate(undefined)}
          title={
            nbTropLongs === 0
              ? 'Aucune phrase trop longue'
              : strategie === 'reformuler'
                ? "L'IA reformule plus brièvement, dans le même sens, les phrases qui ne tiennent pas au débit naturel"
                : "Silence suivant, version plus courte proposée par l'IA, ou découpage aux pauses naturelles"
          }
          className="flex h-11 items-center gap-2.5 rounded-xl border border-cyan-400/40 px-5 text-[15px] text-white transition-colors duration-200 hover:border-cyan-300/80 disabled:opacity-45"
        >
          <Scissors className="size-5 text-cyan-300" /> Ajuster automatiquement
          {nbTropLongs > 0 && (
            <span className="rounded-full bg-warning px-2 text-[12px] font-semibold text-black">{nbTropLongs}</span>
          )}
        </button>
        <button
          type="button"
          disabled={occupe}
          onClick={() => reinitialiser.mutate()}
          title="Revenir aux traductions proposées par l'IA"
          className="flex h-11 items-center gap-2.5 rounded-xl border border-white/15 px-5 text-[15px] text-white transition-colors duration-200 hover:border-white/30 disabled:opacity-50"
        >
          <RotateCcw className="size-5" /> Réinitialiser
        </button>
        <a
          href={href({ page: 'locuteurs', id: job.id })}
          className="flex h-11 items-center gap-2.5 rounded-xl border border-white/15 px-5 text-[15px] text-white transition-colors duration-200 hover:border-white/30"
        >
          <UsersRound className="size-5" /> Locuteurs
        </a>
      </div>

      <AnimatePresence>
        {job.dernier_ajustement && cleBilan !== bilanVu && (
          <BilanBandeau bilan={job.dernier_ajustement} onClose={() => setBilanVu(cleBilan)} />
        )}
      </AnimatePresence>

      <div className="grid gap-5 xl:grid-cols-[320px_1fr] 2xl:grid-cols-[420px_1fr]">
        <div>
          <video ref={video} src={videoUrl(job.id)} controls preload="metadata" className="aspect-video w-full rounded-xl bg-black" />
        </div>
        <Panel className="overflow-visible">
          <div className="grid grid-cols-[20px_76px_1fr_20px_1fr_104px_32px_24px] gap-2 border-b border-white/[0.08] px-5 py-3.5 text-[16px] font-medium text-white">
            <span />
            <span />
            <span>Texte original</span>
            <span />
            <span>Traduction (éditable)</span>
          </div>
          {/* Apparition progressive seulement pour les vidéos courtes : sur des centaines de
              phrases, le décalage de 40 ms rendrait l'affichage interminable. */}
          <motion.div variants={staggerVariants} initial={segments.length <= 30 ? 'hidden' : false} animate="visible">
            {segments.map((seg) => (
              <SegmentRow
                key={seg.id}
                seg={seg}
                langues={langues}
                traduction={traductions[seg.id]}
                selectionne={actif === seg.id}
                enLecture={clock.lecture && segmentLu?.id === seg.id}
                modifie={brouillon[seg.id] !== undefined && brouillon[seg.id] !== seg.traduction}
                onChange={(v) => setBrouillon((b) => ({ ...b, [seg.id]: v }))}
                onSelect={() => setSelection(seg.id)}
                onPlay={() => {
                  if (clock.lecture && segmentLu?.id === seg.id) clock.toggle()
                  else {
                    setSelection(seg.id)
                    clock.playRange(seg.debut, seg.fin)
                  }
                }}
                onRevert={() => setBrouillon((b) => ({ ...b, [seg.id]: seg.traduction_ia ?? seg.traduction ?? '' }))}
                onAdjust={() => ajuster.mutate([seg.id])}
                locuteurs={pistes}
                onSpeaker={(code) => changerLocuteur.mutate({ segId: seg.id, code })}
                onTone={(ton) => changerTon.mutate({ segId: seg.id, ton })}
                original={originaux[seg.id] ?? seg.texte}
                originalModifie={originalModifie(seg)}
                onChangeOriginal={(v) => setOriginaux((o) => ({ ...o, [seg.id]: v }))}
              />
            ))}
          </motion.div>
        </Panel>
      </div>

      <BottomPanel>
        <EditorTimeline
          duree={job.duree_s ?? data?.duree_s ?? 0}
          temps={clock.temps}
          lecture={clock.lecture}
          segments={segments}
          locuteurs={pistes}
          traductions={traductions}
          selection={actif}
          peaks={peaks}
          filmstrip={filmstripUrl(job.id)}
          onSeek={clock.seek}
          onToggle={clock.toggle}
          onSelect={setSelection}
        />
        <div className="flex flex-wrap items-center gap-5 px-6 pt-1 pb-4">
          <div className="w-[214px]">
            <UserFooter bordered={false} />
          </div>
          <span className="flex items-center gap-2.5 rounded-full border border-white/12 bg-white/[0.02] px-5 py-2.5 text-[15px] text-white/90">
            <CircleCheck className="size-5 fill-success text-[#0e1119]" />
            {job.traduction_validee ? 'Traduction validée' : 'Traduction prête'}
            <span className="text-white/50">•</span> {segments.length} segment{segments.length > 1 ? 's' : ''} détecté{segments.length > 1 ? 's' : ''}
            <span className="text-white/50">•</span>
            <a href={href({ page: 'locuteurs', id: job.id })} className="hover:text-white hover:underline">
              {Math.max(1, pistes.length)} locuteur{pistes.length > 1 ? 's' : ''}
            </a>
            {nbTropLongs > 0 && (
              <>
                <span className="text-white/50">•</span>
                <span className="text-warning">
                  {nbTropLongs} trop long{nbTropLongs > 1 ? 's' : ''}
                </span>
              </>
            )}
          </span>
          <div className="ml-auto flex items-center gap-4">
            <button
              type="button"
              disabled={occupe || modifications.length === 0 || vide}
              onClick={() => enregistrer.mutate()}
              className="flex h-14 items-center gap-3 rounded-2xl border border-white/20 px-6 text-[16px] text-white transition-colors duration-200 hover:border-white/40 disabled:opacity-45"
            >
              <Save className="size-5" /> Enregistrer comme brouillon
            </button>
            <PrimaryButton
              size="lg"
              disabled={occupe || vide || segments.length === 0}
              onClick={() => valider.mutate()}
              className="h-14 min-w-80 rounded-2xl"
            >
              Valider et générer les voix <ArrowRight className="size-5" />
            </PrimaryButton>
          </div>
        </div>
      </BottomPanel>
    </>
  )
}
