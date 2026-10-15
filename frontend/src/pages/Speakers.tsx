// Locuteurs détectés — maquette reference/VoxLingo Speaker Management Dashboard.png
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowRight,
  Clock3,
  Eye,
  Info,
  Library,
  Mic,
  PencilLine,
  RefreshCw,
  Sparkles,
  UserRound,
  UsersRound,
  Wand2,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Panel } from '@/components/Controls'
import { Gradient, PageHeader } from '@/components/Layout'
import { MiniPlayer } from '@/components/MiniPlayer'
import { OutlineButton, PrimaryButton } from '@/components/PrimaryButton'
import {
  LIBELLES_AGE,
  autoAssignVoices,
  detectSpeakers,
  getJob,
  getSpeakers,
  listVoices,
  mergeSpeakers,
  regenerateVoices,
  speakerAudioUrl,
  speakerAvatarUrl,
  updateSpeaker,
  voicePreviewUrl,
  type Job,
  type Locuteur,
  type Voix,
} from '@/lib/api'
import { formatDuree, nomLangue } from '@/lib/format'
import { href, navigate } from '@/lib/router'
import { cn } from '@/lib/utils'
import { cardVariants, staggerVariants, voxTransition } from '@/motion/variants'

function Avatar({ jobId, loc }: { jobId: string; loc: Locuteur }) {
  const [ok, setOk] = useState(true)
  return (
    <div className="relative shrink-0">
      <div
        className="size-[132px] overflow-hidden rounded-full border-[5px] bg-[#141821]"
        style={{ borderColor: loc.couleur, boxShadow: `0 0 26px ${loc.couleur}66` }}
      >
        {ok ? (
          <img src={speakerAvatarUrl(jobId, loc.code)} alt="" onError={() => setOk(false)} className="size-full object-cover" />
        ) : (
          <UserRound className="m-auto mt-8 size-14 text-white/50" />
        )}
      </div>
      <span
        className="absolute right-1 bottom-1 flex size-10 items-center justify-center rounded-full border-4 border-[#0e1119]"
        style={{ background: loc.couleur }}
      >
        <Mic className="size-4 text-white" />
      </span>
    </div>
  )
}

function libelleVoix(v: Voix) {
  return `${v.nom} · ${v.sexe === 'femme' ? 'femme' : 'homme'} · ${LIBELLES_AGE[v.age].toLowerCase()}${v.registre ? ` · ${v.registre}` : ''}`
}

function ChoixVoix({
  loc,
  voix,
  onChange,
  disabled,
}: {
  loc: Locuteur
  voix: Voix[]
  onChange: (v: string) => void
  disabled: boolean
}) {
  const groupes: [string, Voix[]][] = [
    ['Voix de femmes', voix.filter((v) => v.source === 'xtts' && v.sexe === 'femme')],
    ["Voix d'hommes", voix.filter((v) => v.source === 'xtts' && v.sexe === 'homme')],
    ['Mes voix', voix.filter((v) => v.source === 'perso')],
  ]
  const conseil = loc.sexe_estime ? ` (voix d'origine : ${loc.sexe_estime}${loc.registre ? `, ${loc.registre}` : ''})` : ''
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex items-center gap-2 text-[13px] text-white/75">
        <Library className="size-3.5" /> Voix utilisée pour le doublage{conseil}
      </span>
      <select
        value={loc.voix}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="h-10 w-full cursor-pointer rounded-xl border border-white/12 bg-[#0b0d16] px-3 text-[14px] text-white outline-none focus:border-violet-400/70 disabled:opacity-50"
      >
        <option value="clone">Voix d'origine clonée</option>
        {groupes.map(
          ([titre, liste]) =>
            liste.length > 0 && (
              <optgroup key={titre} label={titre}>
                {liste.map((v) => (
                  <option key={v.id} value={v.id}>
                    {libelleVoix(v)}
                  </option>
                ))}
              </optgroup>
            ),
        )}
      </select>
    </label>
  )
}

function CarteLocuteur({
  job,
  loc,
  index,
  autres,
  voix,
  occupe,
  onRenommer,
  onVoix,
  onFusion,
}: {
  job: Job
  loc: Locuteur
  index: number
  autres: Locuteur[]
  voix: Voix[]
  occupe: boolean
  onRenommer: (nom: string) => void
  onVoix: (v: string) => void
  onFusion: (cible: string) => void
}) {
  const champ = useRef<HTMLInputElement>(null)
  const [nom, setNom] = useState(loc.nom)
  const [fusion, setFusion] = useState(false)
  useEffect(() => setNom(loc.nom), [loc.nom])
  const choisie = voix.find((v) => v.id === loc.voix)
  const aJour = !!loc.voix_generee && loc.voix_generee === loc.voix
  // Voix doublée à jour : la phrase générée ; sinon, l'aperçu de la voix choisie dans la bibliothèque
  const srcDoublee = aJour
    ? speakerAudioUrl(job.id, loc.code, 'doublee')
    : choisie?.apercu
      ? voicePreviewUrl(choisie.id)
      : null

  return (
    <motion.div variants={cardVariants} whileHover={{ y: -2, transition: voxTransition(0.25) }}>
      <Panel className="relative flex flex-wrap items-center gap-6 px-6 py-5">
        <Avatar jobId={job.id} loc={loc} />
        <div className="flex w-[230px] flex-col gap-3">
          <span className="flex items-center gap-3">
            <span className="rounded-lg px-3 py-1 text-[14px] font-medium text-white" style={{ background: loc.couleur }}>
              Locuteur {index + 1}
            </span>
            <button type="button" onClick={() => champ.current?.focus()} aria-label="Renommer" className="text-white/80 hover:text-white">
              <PencilLine className="size-5" />
            </button>
          </span>
          <input
            ref={champ}
            value={nom}
            onChange={(e) => setNom(e.target.value)}
            onBlur={() => nom.trim() && nom !== loc.nom && onRenommer(nom)}
            onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
            aria-label={`Nom du locuteur ${index + 1}`}
            className="h-11 rounded-full border border-white/15 bg-transparent px-5 text-[18px] font-medium text-white outline-none transition-colors focus:border-violet-400/70"
          />
          <p className="flex items-center gap-2 text-[14px] text-white/85">
            <Clock3 className="size-4" /> Durée de parole : {formatDuree(loc.duree_s)} ({loc.part} %)
          </p>
        </div>
        <div className="grid min-w-[300px] flex-1 gap-4 lg:grid-cols-2">
          <MiniPlayer src={speakerAudioUrl(job.id, loc.code, 'original')} couleur="#a855f7" libelle="Voix originale" />
          <MiniPlayer
            src={srcDoublee}
            couleur={loc.couleur}
            libelle={aJour ? `${loc.voix === 'clone' ? 'Voix clonée' : `Voix ${choisie?.nom ?? ''}`} (${nomLangue(job.langue_cible)})` : `Aperçu : ${choisie?.nom ?? "voix d'origine clonée"}`}
            vide={loc.voix === 'clone' ? 'Cliquez sur « Générer les voix »' : 'Aperçu à générer (Bibliothèque)'}
          />
          <div className="lg:col-span-2">
            <ChoixVoix loc={loc} voix={voix} onChange={onVoix} disabled={occupe} />
            {!aJour && (
              <p className="mt-2 flex items-center gap-2 text-[13px] text-warning">
                <Sparkles className="size-4" /> Voix pas encore générée : cliquez sur « Générer les voix » en bas de page.
              </p>
            )}
          </div>
        </div>
        <div className="flex flex-col gap-2 self-start">
          <OutlineButton onClick={() => champ.current?.focus()} className="h-9 justify-start px-4 text-[14px]">
            <PencilLine className="size-4" /> Renommer
          </OutlineButton>
          <div className="relative">
            <OutlineButton
              disabled={occupe || autres.length === 0}
              onClick={() => setFusion(!fusion)}
              className="h-9 w-full justify-start px-4 text-[14px]"
              title="Le même personnage a été détecté deux fois : regroupez-le"
            >
              <UsersRound className="size-4" /> Fusionner
            </OutlineButton>
            <AnimatePresence>
              {fusion && (
                <motion.div
                  initial={{ opacity: 0, y: 4, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0 }}
                  transition={voxTransition(0.2)}
                  onMouseLeave={() => setFusion(false)}
                  className="absolute right-0 z-20 mt-1 w-64 rounded-xl border border-white/12 bg-popover p-1.5 shadow-2xl"
                >
                  <p className="px-3 py-1.5 text-[12px] text-white/60">Fusionner « {loc.nom} » avec :</p>
                  {autres.map((a) => (
                    <button
                      key={a.code}
                      type="button"
                      onClick={() => {
                        setFusion(false)
                        onFusion(a.code)
                      }}
                      className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-[14px] text-white hover:bg-white/5"
                    >
                      <span className="size-3 rounded-full" style={{ background: a.couleur }} /> {a.nom}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </Panel>
    </motion.div>
  )
}

export function Speakers({ id }: { id: string }) {
  const queryClient = useQueryClient()
  const { data: job } = useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id),
    refetchInterval: (q) => (q.state.data?.etat === 'en_cours' || q.state.data?.etat === 'en_attente' ? 2000 : false),
  })
  const enCours = job?.etat === 'en_cours' || job?.etat === 'en_attente'
  const { data: locuteurs, error } = useQuery({
    queryKey: ['locuteurs', id, job?.maj_le],
    queryFn: () => getSpeakers(id),
    enabled: !!job && !enCours,
  })
  const { data: bibliotheque } = useQuery({ queryKey: ['voix'], queryFn: listVoices })
  const [message, setMessage] = useState<string | null>(null)

  const appliquer = (liste: Locuteur[]) => {
    queryClient.setQueryData(['locuteurs', id, job?.maj_le], liste)
    void queryClient.invalidateQueries({ queryKey: ['segments', id] })
  }
  const erreur = (e: unknown) => setMessage((e as Error).message)
  const modifier = useMutation({
    mutationFn: ({ code, modif }: { code: string; modif: { nom?: string; voix?: string } }) => updateSpeaker(id, code, modif),
    onSuccess: appliquer,
    onError: erreur,
  })
  const fusionner = useMutation({ mutationFn: ({ source, cible }: { source: string; cible: string }) => mergeSpeakers(id, source, cible), onSuccess: appliquer, onError: erreur })
  const attribuer = useMutation({ mutationFn: () => autoAssignVoices(id), onSuccess: appliquer, onError: erreur })
  const generer = useMutation({
    mutationFn: () => regenerateVoices(id),
    onSuccess: (j) => {
      queryClient.setQueryData(['job', id], j)
      navigate({ page: 'projet', id })
    },
    onError: erreur,
  })
  const detecter = useMutation({
    mutationFn: () => detectSpeakers(id),
    onSuccess: (j) => queryClient.setQueryData(['job', id], j),
    onError: erreur,
  })
  const occupe = enCours || modifier.isPending || fusionner.isPending || attribuer.isPending

  const voix = bibliotheque?.voix ?? []
  const nb = locuteurs?.length ?? 0

  return (
    <>
      <div className="relative">
        <PageHeader
          back={{ page: 'edition', id }}
          title={
            <>
              Locuteurs <Gradient>détectés</Gradient>
            </>
          }
          subtitle={
            enCours
              ? 'Identification des locuteurs en cours…'
              : `Nous avons identifié ${nb} locuteur${nb > 1 ? 's' : ''} dans votre vidéo. Gérez leurs voix et leurs profils.`
          }
        />
        <div className="absolute right-0 bottom-0 hidden gap-3 lg:flex">
          <OutlineButton onClick={() => detecter.mutate()} disabled={occupe} className="h-11">
            <RefreshCw className={cn('size-4', enCours && 'animate-spin')} /> Détecter à nouveau
          </OutlineButton>
          <OutlineButton onClick={() => attribuer.mutate()} disabled={occupe || nb === 0} className="h-11" title="Une voix différente de la bibliothèque pour chaque locuteur, du même sexe et du registre le plus proche">
            <Wand2 className="size-4" /> Attribuer automatiquement
          </OutlineButton>
        </div>
      </div>

      {message && <p className="mb-4 rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-2 text-destructive">{message}</p>}
      {error && <p className="text-destructive">{(error as Error).message}</p>}
      {enCours && <div className="h-40 animate-pulse rounded-2xl bg-white/[0.04]" />}

      {job && locuteurs && (
        <motion.div variants={staggerVariants} initial="hidden" animate="visible" className="flex flex-col gap-5">
          {locuteurs.map((loc, i) => (
            <CarteLocuteur
              key={loc.code}
              job={job}
              loc={loc}
              index={i}
              autres={locuteurs.filter((l) => l.code !== loc.code)}
              voix={voix}
              occupe={occupe}
              onRenommer={(nom) => modifier.mutate({ code: loc.code, modif: { nom } })}
              onVoix={(v) => modifier.mutate({ code: loc.code, modif: { voix: v } })}
              onFusion={(cible) => fusionner.mutate({ source: loc.code, cible })}
            />
          ))}
        </motion.div>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-4">
        <Panel className="flex flex-1 items-center gap-3 px-5 py-4 text-[14px] text-white/85">
          <Info className="size-5 shrink-0" />
          Vous pouvez modifier les noms, fusionner des locuteurs ou leur donner une voix de la{' '}
          <a href={href({ page: 'voix' })} className="text-violet-300 underline-offset-2 hover:underline">
            bibliothèque
          </a>{' '}
          à tout moment : seules les phrases concernées seront régénérées.
        </Panel>
        {job?.fichier_sortie && (
          <OutlineButton onClick={() => navigate({ page: 'projet', id })} className="h-14 px-6 text-[15px]">
            <Eye className="size-5" /> Voir l'aperçu complet
          </OutlineButton>
        )}
        <OutlineButton onClick={() => navigate({ page: 'edition', id })} className="h-14 px-6 text-[15px]">
          Modifier la traduction <ArrowRight className="size-5" />
        </OutlineButton>
        <PrimaryButton
          size="lg"
          disabled={occupe || generer.isPending || !job?.moteur_traduction}
          onClick={() => generer.mutate()}
          className="h-14 min-w-64"
          title="Seules les phrases dont la voix ou le texte a changé sont refaites"
        >
          <Mic className="size-5" /> Générer les voix
        </PrimaryButton>
      </div>
    </>
  )
}
