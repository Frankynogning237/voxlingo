// Accueil — maquette reference/VoxLingo Neon Translation Dashboard.png
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { ArrowRight, Clock3, Plus } from 'lucide-react'

import { HeroVoice, FilmStack } from '@/components/Illustrations'
import { SystemStatus, useHealth } from '@/components/Layout'
import { OutlineButton, PrimaryButton } from '@/components/PrimaryButton'
import { ProjectCard } from '@/components/ProjectCard'
import { listJobs } from '@/lib/api'
import { href } from '@/lib/router'
import { cardVariants, staggerVariants } from '@/motion/variants'

function RecentProjects() {
  const { data: jobs } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    // Rafraîchit tant qu'un projet est en cours, pour suivre sa progression
    refetchInterval: (query) =>
      query.state.data?.some((j) => j.etat === 'en_cours' || j.etat === 'en_attente') ? 3000 : false,
  })

  return (
    <section>
      <div className="mb-5 flex items-center justify-between">
        <h2 className="flex items-center gap-3 text-[26px] font-medium text-white">
          <Clock3 className="size-7" strokeWidth={1.6} /> Projets récents
        </h2>
        <OutlineButton disabled title="La page « Mes projets » arrive à la phase 9">
          Voir tous mes projets <ArrowRight className="size-4" />
        </OutlineButton>
      </div>
      {jobs && jobs.length === 0 && (
        <motion.p variants={cardVariants} initial="hidden" animate="visible" className="text-white/70">
          Aucun projet pour le moment. Importez votre première vidéo pour commencer.
        </motion.p>
      )}
      <motion.div
        variants={staggerVariants}
        initial="hidden"
        animate="visible"
        className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3"
      >
        {jobs?.slice(0, 3).map((job) => <ProjectCard key={job.id} job={job} />)}
      </motion.div>
    </section>
  )
}

export function Home() {
  const { data } = useHealth()
  const prenom = data?.nom_utilisateur?.trim()

  return (
    <>
      <div className="relative mb-6 min-h-[250px]">
        <div className="absolute top-0 right-0 z-10">
          <SystemStatus />
        </div>
        <HeroVoice className="pointer-events-none absolute top-6 right-0 hidden h-[230px] w-[640px] xl:block" />
        <div className="relative pt-14">
          <h1 className="text-[52px] leading-[1.15] font-semibold tracking-tight">
            <span className="text-white">{prenom ? `Bonjour ${prenom},` : 'Bonjour,'}</span>
            <br />
            <span className="text-gradient">Bienvenue sur VoxLingo !</span>
          </h1>
          <p className="mt-6 text-[20px] leading-relaxed text-white/80">
            Transformez vos vidéos en plusieurs langues grâce à l'IA.
            <br />
            Doublage, traduction et synchronisation labiale, en toute simplicité.
          </p>
        </div>
      </div>

      <motion.section
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="relative mb-10 overflow-hidden rounded-3xl border-gradient p-10 shadow-[0_0_40px_rgba(124,92,255,0.12)] [--gradient-fill:#0d0f19]"
      >
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_80%_120%,rgba(34,211,238,0.10),transparent_60%),radial-gradient(ellipse_at_10%_0%,rgba(139,92,246,0.10),transparent_55%)]" />
        <div className="relative flex flex-col items-center gap-10 lg:flex-row">
          <FilmStack />
          <span className="hidden h-28 w-[3px] rounded-full bg-linear-to-b from-violet-500 to-fuchsia-500 lg:block" />
          <div className="flex-1">
            <p className="text-[16px] font-medium tracking-wide text-violet-400 uppercase">
              Commencer un nouveau doublage
            </p>
            <h2 className="mt-2 text-[40px] font-semibold text-white">Nouveau doublage</h2>
            <p className="mt-2 max-w-2xl text-[18px] leading-relaxed text-white/80">
              Importez une vidéo, choisissez vos langues, et laissez VoxLingo faire le reste : transcription,
              traduction, clonage vocal et synchronisation labiale.
            </p>
            <PrimaryButton href={href({ page: 'nouveau' })} size="lg" className="mt-7 min-w-80">
              <Plus className="size-5" strokeWidth={2.2} /> Importer une vidéo
            </PrimaryButton>
          </div>
        </div>
      </motion.section>

      <RecentProjects />
    </>
  )
}
