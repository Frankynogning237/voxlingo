// Mes projets — maquette reference/VoxLingo Neon Dashboard Split Screen.png (partie gauche)
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowRight,
  CalendarDays,
  ChevronDown,
  CircleCheck,
  CircleX,
  Copy,
  EllipsisVertical,
  Film,
  FolderOpen,
  Layers,
  LayoutGrid,
  List,
  Play,
  Search,
  Trash2,
} from 'lucide-react'
import { useMemo, useState } from 'react'

import { Flag } from '@/components/Flag'
import { Gradient, PageHeader } from '@/components/Layout'
import { PrimaryButton } from '@/components/PrimaryButton'
import { deleteJob, duplicateJob, listJobs, thumbnailUrl, type Job } from '@/lib/api'
import { formatDuree, nomLangue, titreProjet } from '@/lib/format'
import { href, navigate } from '@/lib/router'
import { cn } from '@/lib/utils'
import { cardHover, cardVariants, staggerVariants, voxTransition } from '@/motion/variants'

type Filtre = 'tous' | 'termines' | 'en_cours' | 'erreurs'
type Tri = 'recent' | 'ancien' | 'nom' | 'duree'

const enCours = (j: Job) => j.etat === 'en_attente' || j.etat === 'en_cours' || j.etat === 'a_valider'
const FILTRES: { cle: Filtre; libelle: string; garde: (j: Job) => boolean }[] = [
  { cle: 'tous', libelle: 'Tous', garde: () => true },
  { cle: 'termines', libelle: 'Terminés', garde: (j) => j.etat === 'termine' },
  { cle: 'en_cours', libelle: 'En cours', garde: enCours },
  { cle: 'erreurs', libelle: 'Erreurs', garde: (j) => j.etat === 'erreur' || j.etat === 'annule' },
]
const TRIS: Record<Tri, string> = {
  recent: 'Date récente',
  ancien: 'Date ancienne',
  nom: 'Nom',
  duree: 'Durée',
}

function formatDateProjet(iso: string) {
  const d = new Date(iso)
  const jours = Math.floor((new Date().setHours(0, 0, 0, 0) - new Date(iso).setHours(0, 0, 0, 0)) / 86_400_000)
  const heure = d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
  if (jours === 0) return `Aujourd'hui, ${heure}`
  if (jours === 1) return `Hier, ${heure}`
  return d.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' })
}

function BadgeEtat({ job }: { job: Job }) {
  const [libelle, classe] =
    job.etat === 'termine'
      ? ['Terminé', 'bg-success text-black']
      : job.etat === 'erreur'
        ? ['Erreur', 'bg-destructive text-white']
        : job.etat === 'annule'
          ? ['Annulé', 'bg-warning text-black']
          : job.etat === 'a_valider'
            ? ['À relire', 'bg-violet-500 text-white']
            : ['En cours', 'bg-blue-500 text-white']
  return <span className={cn('rounded-full px-3 py-0.5 text-[13px] font-medium shadow', classe)}>{libelle}</span>
}

function MenuProjet({ job, onClose }: { job: Job; onClose: () => void }) {
  const queryClient = useQueryClient()
  const [confirmer, setConfirmer] = useState(false)
  const actualiser = () => queryClient.invalidateQueries({ queryKey: ['jobs'] })
  const dupliquer = useMutation({ mutationFn: () => duplicateJob(job.id), onSuccess: () => { void actualiser(); onClose() } })
  const supprimer = useMutation({ mutationFn: () => deleteJob(job.id), onSuccess: () => { void actualiser(); onClose() } })
  const occupe = job.etat === 'en_attente' || job.etat === 'en_cours'
  const erreur = (dupliquer.error ?? supprimer.error) as Error | null
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98, y: 2 }}
      transition={voxTransition(0.2)}
      onClick={(e) => e.preventDefault()}
      onMouseLeave={onClose}
      className="absolute right-2 bottom-10 z-30 w-56 rounded-xl border border-white/12 bg-popover p-1.5 shadow-2xl"
    >
      <button type="button" onClick={() => navigate({ page: 'projet', id: job.id })} className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-[15px] text-white hover:bg-white/5">
        <FolderOpen className="size-4" /> Ouvrir
      </button>
      <button
        type="button"
        disabled={dupliquer.isPending}
        onClick={() => dupliquer.mutate()}
        title="Crée un nouveau projet avec la même vidéo et les mêmes réglages"
        className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-[15px] text-white hover:bg-white/5 disabled:opacity-50"
      >
        <Copy className="size-4" /> {dupliquer.isPending ? 'Duplication…' : 'Dupliquer'}
      </button>
      {confirmer ? (
        <div className="mt-1 rounded-lg bg-destructive/10 p-2 text-[13px] text-destructive">
          Supprimer définitivement ce projet et ses fichiers ?
          <div className="mt-2 flex gap-2">
            <button type="button" onClick={() => supprimer.mutate()} disabled={supprimer.isPending} className="rounded-md bg-destructive px-3 py-1 font-medium text-white">
              {supprimer.isPending ? '…' : 'Supprimer'}
            </button>
            <button type="button" onClick={() => setConfirmer(false)} className="px-2 text-white/80">
              Annuler
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          disabled={occupe}
          title={occupe ? 'Annulez le traitement en cours avant de supprimer' : undefined}
          onClick={() => setConfirmer(true)}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-[15px] text-destructive hover:bg-destructive/10 disabled:opacity-40"
        >
          <Trash2 className="size-4" /> Supprimer
        </button>
      )}
      {erreur && <p className="px-3 py-1 text-[12px] text-destructive">{erreur.message}</p>}
    </motion.div>
  )
}

function Miniature({ job, className }: { job: Job; className?: string }) {
  const [ok, setOk] = useState(true)
  return (
    <div className={cn('relative overflow-hidden bg-muted', className)}>
      {ok ? (
        <img src={thumbnailUrl(job.id)} alt="" onError={() => setOk(false)} className="size-full object-cover" />
      ) : (
        <div className="flex size-full items-center justify-center">
          <Film className="size-8 text-muted-foreground" />
        </div>
      )}
    </div>
  )
}

function CarteProjet({ job }: { job: Job }) {
  const [menu, setMenu] = useState(false)
  return (
    <motion.a
      href={href({ page: 'projet', id: job.id })}
      variants={cardVariants}
      whileHover={cardHover}
      exit={{ opacity: 0, scale: 0.97, transition: { duration: 0.25, ease: 'easeIn' } }}
      layout
      className="relative flex flex-col overflow-visible rounded-2xl border border-border bg-card/85"
    >
      <div className="relative">
        <Miniature job={job} className="aspect-[16/8] rounded-t-2xl" />
        <span className="absolute top-3 right-3">
          <BadgeEtat job={job} />
        </span>
        {job.duree_s != null && (
          <span className="absolute right-3 bottom-3 flex items-center gap-1.5 rounded-full bg-black/75 px-2.5 py-0.5 text-[14px] font-medium text-white tabular-nums">
            <Play className="size-3.5 fill-white" /> {formatDuree(job.duree_s)}
          </span>
        )}
      </div>
      <div className="flex flex-col gap-2 px-4 pt-3 pb-4">
        <p className="truncate text-[18px] font-medium text-white" title={job.nom_fichier}>
          {titreProjet(job.nom_fichier)}
        </p>
        <p className="flex items-center gap-2 text-[14px] text-white/85">
          <Flag langue={job.langue_source} /> {job.langue_source ? nomLangue(job.langue_source) : '—'}
          <ArrowRight className="size-3.5 text-white/60" />
          <Flag langue={job.langue_cible} /> {nomLangue(job.langue_cible)}
        </p>
        <div className="flex items-center justify-between">
          <p className="flex items-center gap-2 text-[14px] text-white/80">
            <CalendarDays className="size-4" /> {formatDateProjet(job.cree_le)}
          </p>
          <button
            type="button"
            aria-label="Actions du projet"
            onClick={(e) => {
              e.preventDefault()
              setMenu(!menu)
            }}
            className="rounded-md p-1 text-white/85 hover:bg-white/10"
          >
            <EllipsisVertical className="size-5" />
          </button>
        </div>
      </div>
      <AnimatePresence>{menu && <MenuProjet job={job} onClose={() => setMenu(false)} />}</AnimatePresence>
    </motion.a>
  )
}

function LigneProjet({ job }: { job: Job }) {
  const [menu, setMenu] = useState(false)
  return (
    <motion.a
      href={href({ page: 'projet', id: job.id })}
      variants={cardVariants}
      layout
      exit={{ opacity: 0 }}
      className="relative flex items-center gap-5 rounded-xl border border-border bg-card/85 p-2 pr-4 transition-colors hover:border-white/20"
    >
      <Miniature job={job} className="h-16 w-28 shrink-0 rounded-lg" />
      <p className="min-w-0 flex-1 truncate text-[16px] font-medium text-white">{titreProjet(job.nom_fichier)}</p>
      <p className="hidden w-56 items-center gap-2 text-[14px] text-white/85 lg:flex">
        <Flag langue={job.langue_source} /> <ArrowRight className="size-3.5" /> <Flag langue={job.langue_cible} /> {nomLangue(job.langue_cible)}
      </p>
      <span className="w-16 text-[14px] text-white/80 tabular-nums">{formatDuree(job.duree_s)}</span>
      <span className="hidden w-40 text-[14px] text-white/70 md:block">{formatDateProjet(job.cree_le)}</span>
      <BadgeEtat job={job} />
      <button
        type="button"
        aria-label="Actions du projet"
        onClick={(e) => {
          e.preventDefault()
          setMenu(!menu)
        }}
        className="rounded-md p-1 text-white/85 hover:bg-white/10"
      >
        <EllipsisVertical className="size-5" />
      </button>
      <AnimatePresence>{menu && <MenuProjet job={job} onClose={() => setMenu(false)} />}</AnimatePresence>
    </motion.a>
  )
}

export function Projects() {
  const { data: jobs, isLoading, error } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    refetchInterval: (q) => (q.state.data?.some((j) => j.etat === 'en_cours' || j.etat === 'en_attente') ? 3000 : false),
  })
  const [filtre, setFiltre] = useState<Filtre>('tous')
  const [recherche, setRecherche] = useState('')
  const [tri, setTri] = useState<Tri>('recent')
  const [vue, setVue] = useState<'grille' | 'liste'>('grille')

  const visibles = useMemo(() => {
    const garde = FILTRES.find((f) => f.cle === filtre)!.garde
    const texte = recherche.trim().toLowerCase()
    const liste = (jobs ?? []).filter(
      (j) => garde(j) && (!texte || titreProjet(j.nom_fichier).toLowerCase().includes(texte)),
    )
    const comparer: Record<Tri, (a: Job, b: Job) => number> = {
      recent: (a, b) => b.cree_le.localeCompare(a.cree_le),
      ancien: (a, b) => a.cree_le.localeCompare(b.cree_le),
      nom: (a, b) => titreProjet(a.nom_fichier).localeCompare(titreProjet(b.nom_fichier), 'fr'),
      duree: (a, b) => (b.duree_s ?? 0) - (a.duree_s ?? 0),
    }
    return [...liste].sort(comparer[tri])
  }, [jobs, filtre, recherche, tri])

  return (
    <>
      <div className="relative">
        <PageHeader
          title={
            <>
              Mes <Gradient>projets</Gradient>
            </>
          }
          subtitle="Retrouvez et gérez tous vos doublages en un seul endroit."
        />
        <label className="absolute right-0 bottom-1 hidden w-[380px] lg:block">
          <Search className="absolute top-1/2 left-4 size-5 -translate-y-1/2 text-white/70" />
          <input
            value={recherche}
            onChange={(e) => setRecherche(e.target.value)}
            placeholder="Rechercher un projet…"
            className="h-12 w-full rounded-xl border border-white/12 bg-white/[0.03] pr-4 pl-12 text-[15px] text-white outline-none transition-[border-color,box-shadow] duration-200 placeholder:text-white/60 focus:border-violet-400/70 focus:shadow-[0_0_0_3px_rgba(124,92,255,0.15)]"
          />
        </label>
      </div>

      <div className="mb-6 flex flex-wrap items-center gap-3">
        {FILTRES.map((f) => {
          const nombre = (jobs ?? []).filter(f.garde).length
          const actif = filtre === f.cle
          const Icone = { tous: Layers, termines: CircleCheck, en_cours: Play, erreurs: CircleX }[f.cle]
          return (
            <button
              key={f.cle}
              type="button"
              onClick={() => setFiltre(f.cle)}
              className={cn(
                'relative flex h-11 items-center gap-2.5 rounded-full px-5 text-[15px] transition-colors duration-250',
                actif ? 'text-white' : 'border border-white/12 text-white/85 hover:border-white/30',
              )}
            >
              {actif && (
                <motion.span
                  layoutId="filtre-actif"
                  transition={voxTransition(0.25)}
                  className="absolute inset-0 rounded-full bg-brand-gradient shadow-[0_0_16px_rgba(124,92,255,0.45)]"
                />
              )}
              <Icone
                className={cn(
                  'relative size-4',
                  !actif && f.cle === 'termines' && 'text-success',
                  !actif && f.cle === 'en_cours' && 'fill-blue-500 text-blue-500',
                  !actif && f.cle === 'erreurs' && 'text-destructive',
                )}
              />
              <span className="relative">{f.libelle}</span>
              <span className={cn('relative rounded-full px-1.5 text-[12px]', actif ? 'bg-white/25' : 'border border-white/25')}>{nombre}</span>
            </button>
          )
        })}
        <span className="flex-1" />
        <label className="relative flex h-11 items-center rounded-full border border-white/12 pr-3 pl-5 text-[14px] text-white">
          Trier par :
          <select
            value={tri}
            onChange={(e) => setTri(e.target.value as Tri)}
            className="ml-1 cursor-pointer appearance-none bg-transparent pr-6 font-medium text-white outline-none"
          >
            {Object.entries(TRIS).map(([cle, libelle]) => (
              <option key={cle} value={cle} className="bg-popover">
                {libelle}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-3 size-4" />
        </label>
        <div className="flex rounded-xl border border-white/12 p-1">
          {(['grille', 'liste'] as const).map((v) => (
            <button
              key={v}
              type="button"
              aria-label={v === 'grille' ? 'Affichage en grille' : 'Affichage en liste'}
              aria-pressed={vue === v}
              onClick={() => setVue(v)}
              className={cn('rounded-lg p-1.5', vue === v ? 'bg-violet-500/30 text-white' : 'text-white/70 hover:text-white')}
            >
              {v === 'grille' ? <LayoutGrid className="size-5" /> : <List className="size-5" />}
            </button>
          ))}
        </div>
      </div>

      {isLoading && (
        <div className="grid gap-5 *:min-w-0 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }, (_, i) => (
            <div key={i} className="h-72 animate-pulse rounded-2xl bg-white/[0.04]" />
          ))}
        </div>
      )}
      {error && <p className="text-destructive">{(error as Error).message}</p>}
      {jobs && visibles.length === 0 && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-center gap-4 py-16 text-center">
          <Film className="size-10 text-white/50" />
          <p className="text-white/80">{jobs.length === 0 ? "Vous n'avez encore aucun projet." : 'Aucun projet ne correspond.'}</p>
          {jobs.length === 0 && <PrimaryButton href={href({ page: 'nouveau' })}>Créer mon premier doublage</PrimaryButton>}
        </motion.div>
      )}

      <motion.div
        key={vue}
        variants={staggerVariants}
        initial={visibles.length <= 30 ? 'hidden' : false}
        animate="visible"
        className={vue === 'grille' ? 'grid gap-5 *:min-w-0 md:grid-cols-2 xl:grid-cols-3' : 'flex flex-col gap-3'}
      >
        <AnimatePresence>
          {visibles.map((job) => (vue === 'grille' ? <CarteProjet key={job.id} job={job} /> : <LigneProjet key={job.id} job={job} />))}
        </AnimatePresence>
      </motion.div>
    </>
  )
}
