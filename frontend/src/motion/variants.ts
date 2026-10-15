// Système de mouvement VoxLingo (reference/animation.txt).
import type { Transition, Variants } from 'framer-motion'

export const VOX_EASE = [0.22, 1, 0.36, 1] as const

export const DUREES = {
  fast: 0.2,
  base: 0.3,
  page: 0.4,
  emphasis: 0.5,
  celebration: 0.6,
} as const

export const voxTransition = (duration: number = DUREES.base): Transition => ({
  duration,
  ease: VOX_EASE,
})

export const pageVariants: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: voxTransition(DUREES.page) },
  exit: { opacity: 0, y: -4, transition: { duration: DUREES.fast, ease: 'easeIn' } },
}

/** Conteneur qui fait apparaître ses enfants avec un décalage de 40 ms. */
export const staggerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.04 } },
}

export const cardVariants: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: voxTransition(DUREES.base) },
}

export const badgeVariants: Variants = {
  hidden: { opacity: 0, y: 4 },
  visible: { opacity: 1, y: 0, transition: voxTransition(DUREES.base) },
}

/** Survol des cartes de projet : élévation de 3 px, ombre douce, bordure plus lumineuse. */
export const cardHover = {
  y: -3,
  boxShadow: '0 12px 32px rgba(0,0,0,0.32)',
  borderColor: '#45405F',
  transition: voxTransition(0.25),
}
