// Contrôles stylés comme les maquettes : interrupteur, curseur, carte « verre ».
import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'
import { voxTransition } from '@/motion/variants'

export function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('rounded-2xl border border-border bg-card/85', className)}>{children}</div>
}

export function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative h-8 w-14 shrink-0 rounded-full transition-colors duration-250 ease-vox focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-400',
        checked ? 'bg-linear-to-r from-violet-500 to-indigo-500 shadow-[0_0_14px_rgba(124,92,255,0.45)]' : 'bg-[#2a2f3d]',
      )}
    >
      <motion.span
        layout
        transition={voxTransition(0.25)}
        className={cn('absolute top-1 size-6 rounded-full bg-white shadow', checked ? 'right-1' : 'left-1')}
      />
    </button>
  )
}

export function Slider({
  value,
  onChange,
  label,
}: {
  value: number
  onChange: (value: number) => void
  label: string
}) {
  return (
    <input
      type="range"
      min={0}
      max={100}
      value={value}
      aria-label={label}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ ['--fill' as string]: `${value}%` }}
      className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-[linear-gradient(90deg,#8b5cf6,#6366f1_var(--fill),#2a2f3d_var(--fill))] [&::-webkit-slider-thumb]:size-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(124,92,255,0.6)]"
    />
  )
}

/** Lèvres (icône « Synchronisation labiale » des maquettes, absente de Lucide). */
export function LipsIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <path d="M2 12c2.5-3 4.5-4.5 6.5-4.5 1.3 0 2.4.8 3.5.8s2.2-.8 3.5-.8c2 0 4 1.5 6.5 4.5-3 3.6-6 5.5-10 5.5S5 15.6 2 12Z" />
      <path d="M2 12c4 1.2 7 1.6 10 1.6s6-.4 10-1.6" />
    </svg>
  )
}
