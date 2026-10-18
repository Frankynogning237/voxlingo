// Fiche de la vidéo : ce que l'IA a compris en regardant et en écoutant la vidéo avant de la traduire.
import { Clapperboard, Plus, X } from 'lucide-react'

import type { FicheVideo as Fiche, Scene } from '@/lib/api'
import { formatHorodatage } from '@/lib/useVideoClock'

const champ =
  'w-full resize-y rounded-xl border border-white/12 bg-[#0b0d16] px-3.5 py-2.5 text-[14px] leading-relaxed text-white outline-none placeholder:text-white/35 focus:border-violet-400/70 disabled:opacity-50'

export const FICHE_VIDE: Fiche = { genre: '', resume: '', registre: '', ton: '', personnages: [] }

function Champ({
  libelle,
  valeur,
  onChange,
  disabled,
  lignes = 1,
  aide,
}: {
  libelle: string
  valeur: string
  onChange: (v: string) => void
  disabled: boolean
  lignes?: number
  aide: string
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[14px] text-white/90">{libelle}</span>
      <textarea rows={lignes} value={valeur} disabled={disabled} maxLength={2000} placeholder={aide} onChange={(e) => onChange(e.target.value)} className={champ} />
    </label>
  )
}

/**
 * Fiche éditable (genre, résumé, registre, ton, personnages) et description des scènes vues par
 * l'IA. La traduction s'appuie dessus pour rester dans la scène au lieu de traduire mot à mot.
 */
export function FicheVideo({
  fiche,
  scenes,
  onChange,
  disabled = false,
}: {
  fiche: Fiche
  scenes: Scene[]
  onChange: (f: Fiche) => void
  disabled?: boolean
}) {
  const maj = (champs: Partial<Fiche>) => onChange({ ...fiche, ...champs })
  const majPerso = (i: number, champs: Partial<Fiche['personnages'][number]>) =>
    maj({ personnages: fiche.personnages.map((p, k) => (k === i ? { ...p, ...champs } : p)) })
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 md:grid-cols-2">
        <Champ libelle="Genre" valeur={fiche.genre} onChange={(genre) => maj({ genre })} disabled={disabled} aide="Sketch comique, dispute, interview…" />
        <Champ libelle="Ton général" valeur={fiche.ton} onChange={(ton) => maj({ ton })} disabled={disabled} aide="Humour, tension, émotion…" />
        <Champ libelle="Résumé" valeur={fiche.resume} onChange={(resume) => maj({ resume })} disabled={disabled} lignes={3} aide="Ce qui se passe, en 2 ou 3 phrases." />
        <Champ libelle="Registre de langue" valeur={fiche.registre} onChange={(registre) => maj({ registre })} disabled={disabled} lignes={3} aide="Argot entre jeunes, familier, soutenu…" />
      </div>
      <div className="flex flex-col gap-2">
        <span className="text-[14px] text-white/90">Personnages</span>
        {fiche.personnages.map((p, i) => (
          <div key={i} className="flex flex-col gap-2 sm:flex-row">
            <input
              value={p.locuteur}
              disabled={disabled}
              maxLength={100}
              onChange={(e) => majPerso(i, { locuteur: e.target.value })}
              aria-label="Locuteur"
              className={`${champ} sm:w-44`}
            />
            <input
              value={p.description}
              disabled={disabled}
              maxLength={1000}
              onChange={(e) => majPerso(i, { description: e.target.value })}
              placeholder="Qui il est, âge, caractère, relation avec les autres"
              aria-label="Description du personnage"
              className={`${champ} flex-1`}
            />
            <button
              type="button"
              disabled={disabled}
              onClick={() => maj({ personnages: fiche.personnages.filter((_, k) => k !== i) })}
              title="Retirer ce personnage"
              className="flex size-10 shrink-0 items-center justify-center self-end rounded-xl border border-white/12 text-white/60 hover:text-white disabled:opacity-45 sm:self-auto"
            >
              <X className="size-4" />
            </button>
          </div>
        ))}
        <button
          type="button"
          disabled={disabled || fiche.personnages.length >= 30}
          onClick={() => maj({ personnages: [...fiche.personnages, { locuteur: '', description: '' }] })}
          className="flex w-fit items-center gap-2 rounded-xl border border-white/12 px-3 py-2 text-[13px] text-white/80 hover:border-white/30 disabled:opacity-45"
        >
          <Plus className="size-4" /> Ajouter un personnage
        </button>
      </div>
      {scenes.length > 0 && (
        <details className="rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3">
          <summary className="flex cursor-pointer items-center gap-2 text-[14px] text-white/90">
            <Clapperboard className="size-4 text-violet-300" /> Ce que l'IA a vu ({scenes.length} scène{scenes.length > 1 ? 's' : ''})
          </summary>
          <ul className="mt-3 flex max-h-56 flex-col gap-2 overflow-y-auto pr-1">
            {scenes.map((s) => (
              <li key={s.debut} className="flex gap-3 text-[13px] text-white/75">
                <span className="shrink-0 text-white/45 tabular-nums">{formatHorodatage(s.debut)}</span>
                {s.description}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}
