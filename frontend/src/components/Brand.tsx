// Éléments graphiques de la marque : logo et ondes néon décoratives.
import { useId } from 'react'

import { cn } from '@/lib/utils'

/** Logo VoxLingo : barres d'onde sonore en dégradé violet → cyan. */
export function LogoMark({ className }: { className?: string }) {
  const id = useId()
  const bars = [
    { x: 2, h: 14 },
    { x: 8, h: 26 },
    { x: 14, h: 38 },
    { x: 20, h: 26 },
    { x: 26, h: 34 },
  ]
  return (
    <svg viewBox="0 0 30 40" className={cn('h-10 w-auto', className)} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#c084fc" />
          <stop offset="55%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      {bars.map((b) => (
        <rect key={b.x} x={b.x} y={(40 - b.h) / 2} width="3.2" height={b.h} rx="1.6" fill={`url(#${id})`} />
      ))}
    </svg>
  )
}

export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark />
      <span className="text-[22px] font-semibold tracking-tight text-white">VoxLingo</span>
    </span>
  )
}

/** Vagues néon floues (bas du menu, coins de page), purement décoratives. */
export function NeonWaves({ className, flip = false }: { className?: string; flip?: boolean }) {
  const id = useId()
  return (
    <svg
      viewBox="0 0 400 300"
      preserveAspectRatio="none"
      className={cn('pointer-events-none', flip && '-scale-x-100', className)}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`${id}-a`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0%" stopColor="#7c3aed" stopOpacity="0.9" />
          <stop offset="60%" stopColor="#3b82f6" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </linearGradient>
        <filter id={`${id}-glow`} x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" />
        </filter>
      </defs>
      {[0, 10, 20, 30, 42, 56].map((d, i) => (
        <path
          key={d}
          d={`M-20 ${300 - d} C 90 ${260 - d}, 150 ${120 + d}, 260 ${90 + d * 0.6} S 380 ${10 + d}, 420 ${-10 + d}`}
          fill="none"
          stroke={`url(#${id}-a)`}
          strokeWidth={i === 0 ? 3 : 1}
          opacity={1 - i * 0.13}
          filter={i === 0 ? `url(#${id}-glow)` : undefined}
        />
      ))}
      <path
        d="M-20 300 C 90 260, 150 120, 260 90 S 380 10, 420 -10 L 420 300 Z"
        fill={`url(#${id}-a)`}
        opacity="0.08"
      />
    </svg>
  )
}
