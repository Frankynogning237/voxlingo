// Illustrations néon des maquettes (accueil, nouveau doublage), en SVG pur.
import { Languages, Play } from 'lucide-react'
import { useId } from 'react'

/** Onde sonore qui se transforme en bulle de dialogue avec bouton lecture (en-tête de l'accueil). */
export function HeroVoice({ className }: { className?: string }) {
  const id = useId()
  // Hauteurs fixes : illustration décorative, pas une mesure audio
  const hauteurs = [6, 10, 16, 12, 26, 18, 34, 22, 48, 30, 62, 40, 74, 52, 88, 60, 70, 44, 56, 34, 42, 26, 30, 18, 20, 12, 10, 8, 6, 5, 4, 4, 3, 3]
  return (
    <svg viewBox="0 0 640 220" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={`${id}-w`} gradientUnits="userSpaceOnUse" x1="20" y1="0" x2="330" y2="0">
          <stop offset="0%" stopColor="#a855f7" stopOpacity="0.2" />
          <stop offset="35%" stopColor="#a855f7" />
          <stop offset="70%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
        <linearGradient id={`${id}-b`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="60%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
        <linearGradient id={`${id}-p`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#a855f7" />
          <stop offset="100%" stopColor="#6366f1" />
        </linearGradient>
        <filter id={`${id}-g`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="6" result="flou" />
          <feMerge>
            <feMergeNode in="flou" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <ellipse cx="500" cy="108" rx="150" ry="88" fill="none" stroke="#3b82f6" strokeOpacity="0.25" transform="rotate(-18 500 108)" />
      <ellipse cx="500" cy="108" rx="128" ry="70" fill="none" stroke="#8b5cf6" strokeOpacity="0.18" transform="rotate(-18 500 108)" />
      <g filter={`url(#${id}-g)`}>
        {hauteurs.map((h, i) => (
          <rect key={i} x={20 + i * 9.5} y={110 - h / 2} width="4" height={h} rx="2" fill={`url(#${id}-w)`} />
        ))}
        <circle cx="352" cy="110" r="2.5" fill="#22d3ee" />
        <circle cx="364" cy="110" r="2.5" fill="#22d3ee" />
        <path
          d="M430 40 h120 a34 34 0 0 1 34 34 v62 a34 34 0 0 1 -34 34 h-78 l-48 30 l12 -30 h-6 a34 34 0 0 1 -34 -34 v-62 a34 34 0 0 1 34 -34 z"
          fill="#0b0f1a"
          stroke={`url(#${id}-b)`}
          strokeWidth="7"
          strokeLinejoin="round"
        />
        <path d="M478 82 L478 138 L526 110 Z" fill={`url(#${id}-p)`} stroke={`url(#${id}-p)`} strokeWidth="6" strokeLinejoin="round" />
      </g>
    </svg>
  )
}

/** Pellicules superposées + tuile de traduction (carte « Nouveau doublage » de l'accueil). */
export function FilmStack() {
  return (
    <div className="relative h-[190px] w-[300px]" aria-hidden="true">
      <div className="absolute top-4 left-8 h-[150px] w-[230px] -rotate-6 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-sm" />
      <div className="absolute top-2 left-14 h-[150px] w-[230px] -rotate-3 rounded-2xl border border-white/10 bg-white/[0.04]" />
      <div className="absolute top-6 left-10 flex h-[150px] w-[230px] -rotate-6 overflow-hidden rounded-xl border border-white/15 shadow-2xl shadow-black/60">
        <div className="flex w-5 flex-col justify-around bg-[#151a26] py-2 pl-1.5">
          {Array.from({ length: 7 }).map((_, i) => (
            <span key={i} className="h-2.5 w-2.5 rounded-[2px] bg-[#2b3142]" />
          ))}
        </div>
        <div className="relative flex-1 bg-[linear-gradient(170deg,#3b2a7a_0%,#7c4dbb_32%,#f08a5d_58%,#3a2f4d_75%,#151826_100%)]">
          <div className="absolute inset-x-0 bottom-0 h-1/2 bg-[linear-gradient(160deg,transparent_30%,#20243a_31%,#1a1d2c_100%)] opacity-90" />
          <Play className="absolute top-1/2 left-1/2 size-9 -translate-x-1/2 -translate-y-1/2 fill-white text-white drop-shadow" />
        </div>
      </div>
      <div className="absolute right-0 bottom-3 flex size-[78px] rotate-6 items-center justify-center rounded-2xl border border-white/15 bg-linear-to-br from-violet-500/80 to-indigo-600/80 shadow-xl shadow-violet-900/50 backdrop-blur">
        <Languages className="size-10 text-white" strokeWidth={1.8} />
      </div>
    </div>
  )
}
