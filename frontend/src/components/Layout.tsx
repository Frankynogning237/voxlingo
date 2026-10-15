import { useQuery } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowLeft, ChevronDown, CirclePlus, FolderClosed, House, Info, Settings, type LucideIcon } from 'lucide-react'
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

import { Logo, NeonWaves } from '@/components/Brand'
import { getHealth } from '@/lib/api'
import { href, type Route } from '@/lib/router'
import { cn } from '@/lib/utils'
import { pageVariants, voxTransition } from '@/motion/variants'

type NavItem = { label: string; icon: LucideIcon; route: Route; disponible: boolean }

const NAV: NavItem[] = [
  { label: 'Accueil', icon: House, route: { page: 'accueil' }, disponible: true },
  { label: 'Nouveau doublage', icon: CirclePlus, route: { page: 'nouveau' }, disponible: true },
  { label: 'Mes projets', icon: FolderClosed, route: { page: 'projets' }, disponible: true },
  { label: 'Paramètres', icon: Settings, route: { page: 'parametres' }, disponible: true },
]

function isActive(item: Route, current: Route) {
  // Un projet en cours de traitement appartient au parcours « Nouveau doublage »
  if (current.page === 'projet' || current.page === 'edition') return item.page === 'nouveau'
  return item.page === current.page
}

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: getHealth, retry: 1 })
}

function NavLink({ item, active }: { item: NavItem; active: boolean }) {
  const { label, icon: Icon, route, disponible } = item
  const content = (
    <>
      {active && (
        <motion.span
          layoutId="nav-actif"
          transition={voxTransition(0.25)}
          className="absolute inset-0 rounded-xl border-gradient shadow-[0_0_22px_rgba(124,92,255,0.35)] [--gradient-fill:#1b1532]"
        >
          <span className="absolute inset-0 rounded-xl bg-linear-to-r from-violet-500/25 to-cyan-400/5" />
        </motion.span>
      )}
      <Icon className="relative size-6 shrink-0" strokeWidth={1.6} />
      <span className="relative">{label}</span>
    </>
  )
  const classes = cn(
    'relative flex h-14 items-center gap-3.5 rounded-xl px-5 text-[16px] whitespace-nowrap transition-colors duration-200',
    active ? 'text-white' : 'text-[#d6d9e3] hover:text-white',
  )
  if (!disponible) {
    return (
      <span title="Disponible dans une prochaine phase" className={cn(classes, 'cursor-not-allowed opacity-45')}>
        {content}
      </span>
    )
  }
  return (
    <a href={href(route)} className={classes} aria-current={active ? 'page' : undefined}>
      {content}
    </a>
  )
}

export function UserFooter({ bordered = true }: { bordered?: boolean }) {
  const { data } = useHealth()
  const nom = data?.nom_utilisateur?.trim()
  return (
    <div className={cn('relative px-2', bordered && 'mt-auto border-t border-white/10 pt-6')}>
      <div className="flex items-center gap-3">
        <span className="flex size-11 items-center justify-center rounded-full bg-linear-to-br from-violet-600 to-indigo-600 text-lg font-medium text-white">
          {(nom?.[0] ?? 'V').toUpperCase()}
        </span>
        <div className="flex-1">
          <p className="text-[15px] text-white">Mon espace</p>
          <p className="flex items-center gap-1.5 text-sm text-success">
            <span className="size-2 rounded-full bg-success" /> En ligne
          </p>
        </div>
        <ChevronDown className="size-5 text-muted-foreground" />
      </div>
    </div>
  )
}

function Sidebar({ current, footer }: { current: Route; footer: boolean }) {
  return (
    <motion.aside
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={voxTransition(0.4)}
      className="relative flex w-[260px] shrink-0 flex-col overflow-hidden border-r border-sidebar-border bg-sidebar px-4 pt-10 pb-7"
    >
      <NeonWaves className="absolute bottom-24 -left-10 h-72 w-[340px] opacity-90" />
      <a href="#/" className="relative mb-14 px-3">
        <Logo />
      </a>
      <nav className="relative flex flex-col gap-2">
        {NAV.map((item) => (
          <NavLink key={item.label} item={item} active={isActive(item.route, current)} />
        ))}
      </nav>
      {footer && <UserFooter />}
    </motion.aside>
  )
}

export function SystemStatus() {
  const { data, isError } = useHealth()
  const etat = isError ? 'erreur' : !data ? 'attente' : data.device === 'cuda' ? 'gpu' : 'cpu'
  const libelle = {
    erreur: 'Serveur local injoignable',
    attente: 'Vérification…',
    gpu: 'GPU NVIDIA prêt',
    cpu: 'Mode CPU',
  }[etat]
  const couleur = { erreur: 'bg-destructive', attente: 'bg-muted-foreground', gpu: 'bg-success', cpu: 'bg-warning' }[etat]
  return (
    <div className="flex items-center gap-4">
      <span
        title={data?.device === 'cuda' ? `${data.vram_totale_go} Go de VRAM` : 'Aucune carte NVIDIA : traitement sur le processeur'}
        className="flex h-10 items-center gap-2.5 rounded-full border border-white/12 bg-[#0d1017] px-4 text-[15px] text-white"
      >
        <span className={cn('size-2.5 rounded-full', couleur, etat === 'gpu' && 'shadow-[0_0_8px_#22c55e]')} />
        {libelle}
      </span>
      <Info className="size-7 text-white/80" strokeWidth={1.5} aria-label="Informations" />
    </div>
  )
}

/** En-tête de page des maquettes : « ← Retour », grand titre avec mot en dégradé, sous-titre. */
export function PageHeader({
  title,
  subtitle,
  back,
  backLabel = 'Retour',
  align = 'left',
}: {
  title: ReactNode
  subtitle?: ReactNode
  back?: Route
  backLabel?: string
  align?: 'left' | 'center'
}) {
  return (
    <header className="relative mb-8">
      <div className="mb-2 flex items-center justify-between gap-4">
        {back ? (
          <a href={href(back)} className="flex items-center gap-3 text-[19px] text-white/90 hover:text-white">
            <ArrowLeft className="size-6" strokeWidth={1.6} /> {backLabel}
          </a>
        ) : (
          <span />
        )}
        <SystemStatus />
      </div>
      <div className={cn(align === 'center' && 'text-center')}>
        <h1 className="text-[44px] leading-[1.15] font-semibold tracking-tight text-white">{title}</h1>
        {subtitle && <p className="mt-2 text-[18px] text-white/80">{subtitle}</p>}
      </div>
    </header>
  )
}

/** Mot mis en valeur dans un titre (dégradé rose → violet → cyan). */
export function Gradient({ children }: { children: ReactNode }) {
  return <span className="text-gradient">{children}</span>
}

/**
 * Zone pleine largeur sous le menu et le contenu (timeline de l'éditeur, maquette
 * « Translation Editor »). Une page y affiche son contenu avec <BottomPanel>.
 */
const BottomSlotContext = createContext<{
  slot: HTMLElement | null
  setActive: (active: boolean) => void
}>({ slot: null, setActive: () => {} })

export function BottomPanel({ children }: { children: ReactNode }) {
  const { slot, setActive } = useContext(BottomSlotContext)
  useEffect(() => {
    setActive(true)
    return () => setActive(false)
  }, [setActive])
  return slot ? createPortal(children, slot) : null
}

export function Layout({ current, children }: { current: Route; children: ReactNode }) {
  const pageKey = current.page === 'projet' || current.page === 'edition' ? `${current.page}-${current.id}` : current.page
  const [slot, setSlot] = useState<HTMLElement | null>(null)
  const [bottomActive, setActive] = useState(false)
  return (
    <BottomSlotContext.Provider value={{ slot, setActive }}>
    <div className="flex h-svh flex-col overflow-hidden bg-background text-foreground">
    <div className="flex min-h-0 flex-1">
      <Sidebar current={current} footer={!bottomActive} />
      <main className="relative flex-1 overflow-y-auto">
        <NeonWaves flip className="absolute -top-16 right-0 h-64 w-[520px] opacity-40" />
        <AnimatePresence mode="wait">
          <motion.div
            key={pageKey}
            variants={pageVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="relative mx-auto max-w-[1400px] px-10 pt-8 pb-10"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
    <div ref={setSlot} className={cn(bottomActive && 'border-t border-sidebar-border bg-sidebar')} />
    </div>
    </BottomSlotContext.Provider>
  )
}
