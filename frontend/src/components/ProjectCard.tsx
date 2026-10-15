// Carte « Projet récent » (maquette Accueil) : miniature, langues, statut.
import { motion } from 'framer-motion'
import { ArrowRight, CircleCheck, PencilLine, CircleSlash, EllipsisVertical, Film, OctagonAlert } from 'lucide-react'
import { useState } from 'react'

import { Flag } from '@/components/Flag'
import { thumbnailUrl, type Job } from '@/lib/api'
import { formatDuree, formatIlYa, nomLangue, titreProjet } from '@/lib/format'
import { href } from '@/lib/router'
import { cn } from '@/lib/utils'
import { cardHover, cardVariants } from '@/motion/variants'

/** Anneau de progression réel (pourcentage envoyé par le backend). */
function ProgressRing({ value }: { value: number }) {
  const r = 8
  const c = 2 * Math.PI * r
  return (
    <svg viewBox="0 0 20 20" className="size-5 -rotate-90">
      <circle cx="10" cy="10" r={r} fill="none" stroke="#2a3040" strokeWidth="2" />
      <circle
        cx="10"
        cy="10"
        r={r}
        fill="none"
        stroke="#3b82f6"
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - value / 100)}
        className="transition-[stroke-dashoffset] duration-300"
      />
    </svg>
  )
}

function Statut({ job }: { job: Job }) {
  switch (job.etat) {
    case 'termine':
      return (
        <span className="flex flex-wrap items-center gap-x-2 text-[15px]">
          <CircleCheck className="size-5 text-success" />
          <span className="text-success">Terminé</span>
          <span className="text-white/60">•</span>
          <span className="whitespace-nowrap text-white/80">{formatIlYa(job.fin_traitement ?? job.maj_le)}</span>
        </span>
      )
    case 'a_valider':
      return (
        <span className="flex items-center gap-2 text-[15px] text-violet-300">
          <PencilLine className="size-5" /> Traduction à relire
        </span>
      )
    case 'erreur':
      return (
        <span className="flex items-center gap-2 text-[15px] text-destructive">
          <OctagonAlert className="size-5" /> Erreur
        </span>
      )
    case 'annule':
      return (
        <span className="flex items-center gap-2 text-[15px] text-warning">
          <CircleSlash className="size-5" /> Annulé
        </span>
      )
    default:
      return (
        <span className="flex items-center gap-2 text-[15px] text-white/85">
          <ProgressRing value={job.progression} /> En cours <span className="text-white/60">•</span> {job.progression} %
        </span>
      )
  }
}

export function ProjectCard({ job }: { job: Job }) {
  const [imageOk, setImageOk] = useState(true)
  return (
    <motion.a
      href={href({ page: 'projet', id: job.id })}
      variants={cardVariants}
      whileHover={cardHover}
      className="flex gap-5 rounded-2xl border border-border bg-card/80 p-2.5 pr-4"
    >
      <div className="relative size-[124px] shrink-0 2xl:size-[140px] overflow-hidden rounded-xl bg-muted">
        {imageOk ? (
          <img
            src={thumbnailUrl(job.id)}
            alt=""
            onError={() => setImageOk(false)}
            className="size-full object-cover animate-in fade-in duration-300"
          />
        ) : (
          <div className="flex size-full items-center justify-center">
            <Film className="size-8 text-muted-foreground" />
          </div>
        )}
        {job.duree_s != null && (
          <span className="absolute bottom-2 left-2 rounded-md bg-black/70 px-2 py-0.5 text-sm font-medium text-white tabular-nums">
            {formatDuree(job.duree_s)}
          </span>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col pt-3">
        <div className="flex items-start justify-between gap-2">
          <p className="truncate text-[18px] font-medium text-white" title={job.nom_fichier}>
            {titreProjet(job.nom_fichier)}
          </p>
          <EllipsisVertical className="size-5 shrink-0 text-white/80" aria-hidden="true" />
        </div>
        <p className="mt-3 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[15px] text-white/90">
          <span className="flex items-center gap-2 whitespace-nowrap">
            <Flag langue={job.langue_source} />
            {job.langue_source ? nomLangue(job.langue_source) : job.etat === 'en_cours' || job.etat === 'en_attente' ? 'Détection…' : 'Non détectée'}
          </span>
          <ArrowRight className="size-4 text-white/70" />
          <span className="flex items-center gap-2 whitespace-nowrap">
            <Flag langue={job.langue_cible} />
            {nomLangue(job.langue_cible)}
          </span>
        </p>
        <div className={cn('mt-4 border-t border-white/10 pt-4')}>
          <Statut job={job} />
        </div>
      </div>
    </motion.a>
  )
}
