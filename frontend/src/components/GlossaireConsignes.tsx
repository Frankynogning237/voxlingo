// Glossaire et consignes de traduction d'un projet : termes imposés et style souhaité.
import { BookText } from 'lucide-react'

const champ =
  'w-full resize-y rounded-xl border border-white/12 bg-[#0b0d16] px-3.5 py-2.5 text-[14px] leading-relaxed text-white outline-none placeholder:text-white/35 focus:border-violet-400/70 disabled:opacity-50'

/**
 * Deux champs : le glossaire (« terme = traduction », une ligne par terme) et les consignes de
 * style. Ils sont suivis par l'IA à chaque traduction et reformulation du projet.
 */
export function GlossaireConsignes({
  glossaire,
  consignes,
  onGlossaire,
  onConsignes,
  disabled = false,
}: {
  glossaire: string
  consignes: string
  onGlossaire: (v: string) => void
  onConsignes: (v: string) => void
  disabled?: boolean
}) {
  const nbTermes = glossaire.split('\n').filter((l) => /[=→:]|->/.test(l) && l.trim().length > 2).length
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <label className="flex flex-col gap-1.5">
        <span className="flex items-center gap-2 text-[14px] text-white/90">
          <BookText className="size-4 text-violet-300" /> Glossaire
          <span className="text-[12px] text-white/50">{nbTermes > 0 ? `${nbTermes} terme${nbTermes > 1 ? 's' : ''}` : 'une ligne par terme'}</span>
        </span>
        <textarea
          rows={4}
          value={glossaire}
          disabled={disabled}
          onChange={(e) => onGlossaire(e.target.value)}
          placeholder={'ma meuf = meine Freundin\nbaraque = Bude\nYanis = Yanis'}
          className={champ}
        />
        <span className="text-[12px] text-white/50">Noms, expressions, mots d'argot : « terme = traduction voulue ».</span>
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="text-[14px] text-white/90">Consignes de style</span>
        <textarea
          rows={4}
          value={consignes}
          disabled={disabled}
          maxLength={2000}
          onChange={(e) => onConsignes(e.target.value)}
          placeholder={"Registre familier entre jeunes, tutoiement.\nGarder les jurons mais sans vulgarité excessive."}
          className={champ}
        />
        <span className="text-[12px] text-white/50">Ton, registre, public visé, ce qu'il faut éviter…</span>
      </label>
    </div>
  )
}
