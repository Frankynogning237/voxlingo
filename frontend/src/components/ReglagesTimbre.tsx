// Réglages de timbre d'un locuteur : variantes prêtes à l'emploi, réglages fins et écoute immédiate.
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, Headphones, RotateCcw, SlidersHorizontal } from 'lucide-react'
import { useState } from 'react'

import { MiniPlayer } from '@/components/MiniPlayer'
import { speakerTimbreUrl, type Timbre } from '@/lib/api'
import { VARIANTES, memeTimbre } from '@/lib/timbre'
import { cn } from '@/lib/utils'
import { voxTransition } from '@/motion/variants'

type Reglage = { cle: keyof Timbre; nom: string; min: number; max: number; pas: number; neutre: number; gauche: string; droite: string; format: (v: number) => string }

const REGLAGES: Reglage[] = [
  { cle: 'ton', nom: 'Hauteur', min: -12, max: 12, pas: 1, neutre: 0, gauche: 'Grave', droite: 'Aiguë', format: (v) => `${v > 0 ? '+' : ''}${v} demi-ton${Math.abs(v) > 1 ? 's' : ''}` },
  { cle: 'formant', nom: 'Timbre', min: 0.8, max: 1.25, pas: 0.01, neutre: 1, gauche: 'Sombre', droite: 'Clair', format: (v) => `${v > 1 ? '+' : ''}${Math.round((v - 1) * 100)} %` },
  { cle: 'graves', nom: 'Chaleur (graves)', min: -12, max: 12, pas: 1, neutre: 0, gauche: '−', droite: '+', format: (v) => `${v > 0 ? '+' : ''}${v} dB` },
  { cle: 'aigus', nom: 'Brillance (aigus)', min: -12, max: 12, pas: 1, neutre: 0, gauche: '−', droite: '+', format: (v) => `${v > 0 ? '+' : ''}${v} dB` },
  { cle: 'espace', nom: 'Espace (réverbération)', min: 0, max: 1, pas: 0.05, neutre: 0, gauche: 'Sec', droite: 'Ample', format: (v) => `${Math.round(v * 100)} %` },
  { cle: 'vibrato', nom: 'Tremblement', min: 0, max: 0.5, pas: 0.01, neutre: 0, gauche: 'Stable', droite: 'Tremblant', format: (v) => `${Math.round(v * 200)} %` },
]

function valeur(timbre: Timbre, r: Reglage) {
  return timbre[r.cle] ?? r.neutre
}

function Curseur({ r, v, onChange, disabled }: { r: Reglage; v: number; onChange: (v: number) => void; disabled: boolean }) {
  const part = ((v - r.min) / (r.max - r.min)) * 100
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex items-center justify-between text-[13px]">
        <span className="text-white/85">{r.nom}</span>
        <span className={cn('tabular-nums', v === r.neutre ? 'text-white/45' : 'text-violet-300')}>{r.format(v)}</span>
      </span>
      <input
        type="range"
        min={r.min}
        max={r.max}
        step={r.pas}
        value={v}
        disabled={disabled}
        aria-label={r.nom}
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={() => onChange(r.neutre)}
        style={{ ['--fill' as string]: `${part}%` }}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-[linear-gradient(90deg,#8b5cf6,#6366f1_var(--fill),#2a2f3d_var(--fill))] disabled:opacity-50 [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(124,92,255,0.6)]"
      />
      <span className="flex justify-between text-[11px] text-white/45">
        <span>{r.gauche}</span>
        <span>{r.droite}</span>
      </span>
    </label>
  )
}

/**
 * Panneau « Timbre de la voix » d'un locuteur. Les réglages s'écoutent aussitôt (filtres appliqués
 * à un extrait de la voix choisie) ; « Appliquer » les enregistre pour la prochaine génération.
 */
export function ReglagesTimbre({
  jobId,
  code,
  enregistre,
  genere,
  disabled,
  onAppliquer,
}: {
  jobId: string
  code: string
  /** Réglages enregistrés pour ce locuteur */
  enregistre: Timbre | undefined
  /** Réglages utilisés lors de la dernière génération des voix */
  genere: Timbre | undefined
  disabled: boolean
  onAppliquer: (t: Timbre) => void
}) {
  const [ouvert, setOuvert] = useState(false)
  const [timbre, setTimbre] = useState<Timbre>(enregistre ?? {})
  const [ecoute, setEcoute] = useState<string | null>(null)
  const modifie = !memeTimbre(timbre, enregistre)
  const actif = Object.keys(enregistre ?? {}).length > 0
  const variante = VARIANTES.find((v) => memeTimbre(v.timbre, timbre))

  const changer = (t: Timbre) => {
    setTimbre(t)
    setEcoute(null)
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02]">
      <button
        type="button"
        onClick={() => setOuvert(!ouvert)}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left text-[14px] text-white/90"
        aria-expanded={ouvert}
      >
        <SlidersHorizontal className="size-4 text-violet-300" />
        Timbre de la voix
        <span className="text-[13px] text-white/55">
          · {actif ? (VARIANTES.find((v) => memeTimbre(v.timbre, enregistre))?.nom ?? 'personnalisé') : 'naturel'}
        </span>
        {!memeTimbre(enregistre, genere) && <span className="text-[12px] text-warning">· à générer</span>}
        <ChevronDown className={cn('ml-auto size-4 transition-transform', ouvert && 'rotate-180')} />
      </button>
      <AnimatePresence initial={false}>
        {ouvert && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={voxTransition(0.25)}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-4 border-t border-white/10 px-4 pt-3 pb-4">
              <div>
                <p className="mb-2 text-[13px] text-white/70">Variantes</p>
                <div className="flex flex-wrap gap-2">
                  {VARIANTES.map((v) => (
                    <button
                      key={v.nom}
                      type="button"
                      title={v.aide}
                      disabled={disabled}
                      onClick={() => changer(v.timbre)}
                      className={cn(
                        'h-8 rounded-full px-3.5 text-[13px] transition-colors duration-200 disabled:opacity-50',
                        variante?.nom === v.nom
                          ? 'bg-brand-gradient text-white shadow-[0_0_12px_rgba(124,92,255,0.4)]'
                          : 'border border-white/12 text-white/85 hover:border-white/30',
                      )}
                    >
                      {v.nom}
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
                {REGLAGES.map((r) => (
                  <Curseur key={r.cle} r={r} v={valeur(timbre, r)} disabled={disabled} onChange={(v) => changer({ ...timbre, [r.cle]: v })} />
                ))}
              </div>
              <label className="flex items-center gap-2.5 text-[13px] text-white/85">
                <input
                  type="checkbox"
                  checked={!!timbre.telephone}
                  disabled={disabled}
                  onChange={(e) => changer({ ...timbre, telephone: e.target.checked ? 1 : 0 })}
                  className="size-4 accent-violet-500"
                />
                Effet téléphone (voix filtrée, comme au bout du fil)
              </label>
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => setEcoute(speakerTimbreUrl(jobId, code, timbre))}
                  className="flex h-9 items-center gap-2 rounded-full border border-white/15 px-4 text-[14px] text-white transition-colors hover:border-violet-400/60"
                >
                  <Headphones className="size-4" /> Écouter ce réglage
                </button>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => changer({})}
                  className="flex h-9 items-center gap-2 rounded-full px-3 text-[13px] text-white/65 hover:text-white disabled:opacity-50"
                >
                  <RotateCcw className="size-4" /> Réinitialiser
                </button>
                <button
                  type="button"
                  disabled={disabled || !modifie}
                  onClick={() => onAppliquer(timbre)}
                  className="ml-auto flex h-9 items-center rounded-full bg-brand-gradient px-5 text-[14px] font-medium text-white disabled:opacity-40"
                >
                  Appliquer au locuteur
                </button>
              </div>
              {ecoute && <MiniPlayer key={ecoute} src={ecoute} couleur="#8b5cf6" libelle="Écoute du réglage (voix choisie, avant doublage)" />}
              <p className="text-[12px] text-white/50">
                Double-cliquez sur un curseur pour le remettre au neutre. Après « Appliquer », cliquez sur « Générer les voix » : seules les
                phrases de ce locuteur seront refaites.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
