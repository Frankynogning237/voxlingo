// Bouton principal en dégradé violet → cyan, avec reflet au survol (animation.txt §D).
import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

type Props = {
  children: ReactNode
  className?: string
  size?: 'md' | 'lg'
} & (
  | ({ href: string } & Omit<HTMLMotionProps<'a'>, 'children'>)
  | ({ href?: undefined } & Omit<HTMLMotionProps<'button'>, 'children'>)
)

const base =
  'group relative isolate inline-flex items-center justify-center gap-2.5 overflow-hidden rounded-full bg-brand-gradient font-medium text-white shadow-lg shadow-violet-500/25 outline-none transition-[box-shadow,opacity] duration-200 hover:shadow-[0_0_28px_rgba(124,92,255,0.45)] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cyan-400 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none'

const sizes = {
  md: 'h-11 px-6 text-[15px]',
  lg: 'h-13 px-8 text-base',
}

function Shine() {
  return (
    <span
      aria-hidden="true"
      className="pointer-events-none absolute inset-y-0 -left-1/2 w-1/3 -skew-x-12 bg-linear-to-r from-transparent via-white/30 to-transparent opacity-0 transition-all duration-500 ease-in-out group-hover:left-full group-hover:opacity-100"
    />
  )
}

export function PrimaryButton({ children, className, size = 'md', ...props }: Props) {
  const classes = cn(base, sizes[size], className)
  const content = (
    <>
      <Shine />
      <span className="relative z-10 flex items-center gap-2.5">{children}</span>
    </>
  )
  if (props.href !== undefined) {
    return (
      <motion.a whileHover={{ y: -1 }} whileTap={{ scale: 0.98 }} className={classes} {...(props as HTMLMotionProps<'a'>)}>
        {content}
      </motion.a>
    )
  }
  const disabled = (props as HTMLMotionProps<'button'>).disabled
  return (
    <motion.button
      type="button"
      whileHover={disabled ? undefined : { y: -1 }}
      whileTap={disabled ? undefined : { scale: 0.98 }}
      className={classes}
      {...(props as HTMLMotionProps<'button'>)}
    >
      {content}
    </motion.button>
  )
}

/** Bouton secondaire en contour arrondi (« Voir tous mes projets », « Annuler »…). */
export function OutlineButton({
  children,
  className,
  ...props
}: { children: ReactNode; className?: string } & Omit<HTMLMotionProps<'button'>, 'children'>) {
  return (
    <motion.button
      type="button"
      whileTap={{ scale: 0.98 }}
      className={cn(
        'inline-flex h-10 items-center justify-center gap-2 rounded-full border border-white/15 bg-white/[0.02] px-5 text-sm text-foreground transition-colors duration-200 hover:border-white/30 hover:bg-white/[0.05] disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
      {...props}
    >
      {children}
    </motion.button>
  )
}
