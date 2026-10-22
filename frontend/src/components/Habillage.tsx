// Habillage de la vidéo exportée : sous-titres incrustés animés, flou des anciens sous-titres, logo discret,
// amélioration légère de l'image et suppression des métadonnées. Réexport sans régénérer les voix.
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import { Captions, Eraser, ImageUpscale, RotateCcw, ScanText, ShieldCheck, Sparkles, Stamp, TriangleAlert, Wand2 } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'

import { Panel, Switch } from '@/components/Controls'
import { PrimaryButton } from '@/components/PrimaryButton'
import {
  detectOldSubtitles,
  dressingPreview,
  HABILLAGE_DEFAUT,
  setDressing,
  type AnimationSousTitres,
  type CoinLogo,
  type Habillage as Reglages,
  type Job,
  type StyleSousTitres,
} from '@/lib/api'
import { cn } from '@/lib/utils'
import { voxTransition } from '@/motion/variants'

const STYLES: { cle: StyleSousTitres; nom: string; aide: string }[] = [
  { cle: 'reseaux', nom: 'Réseaux', aide: 'Quelques mots à la fois, le mot prononcé ressort' },
  { cle: 'impact', nom: 'Impact', aide: 'Majuscules énormes, deux mots par écran' },
  { cle: 'karaoke', nom: 'Karaoké', aide: 'La phrase se colore au fil de la voix' },
  { cle: 'classique', nom: 'Classique', aide: 'Sous-titres de film, sobres et lisibles' },
]
const ANIMATIONS: { cle: AnimationSousTitres; nom: string }[] = [
  { cle: 'aucune', nom: 'Aucune' },
  { cle: 'fondu', nom: 'Fondu' },
  { cle: 'pop', nom: 'Pop' },
  { cle: 'glisse', nom: 'Glissement' },
]
const TAILLES: { cle: Reglages['taille']; nom: string }[] = [
  { cle: 'petite', nom: 'Petite' },
  { cle: 'moyenne', nom: 'Moyenne' },
  { cle: 'grande', nom: 'Grande' },
]
const POSITIONS: { cle: Reglages['position']; nom: string }[] = [
  { cle: 'haut', nom: 'Haut' },
  { cle: 'milieu', nom: 'Milieu' },
  { cle: 'bas', nom: 'Bas' },
]
const COINS: { cle: CoinLogo; nom: string }[] = [
  { cle: 'haut_gauche', nom: '↖ Haut gauche' },
  { cle: 'haut_droite', nom: '↗ Haut droite' },
  { cle: 'bas_gauche', nom: '↙ Bas gauche' },
  { cle: 'bas_droite', nom: '↘ Bas droite' },
]
const COULEURS = ['#facc15', '#a855f7', '#22d3ee', '#4ade80', '#f472b6', '#ffffff']

const meme = (a: Reglages, b: Reglages) => JSON.stringify(a) === JSON.stringify(b)
const actif = (r: Reglages) => r.sous_titres || r.flou || r.logo || r.qualite || r.sans_metadonnees

function formatTemps(s: number) {
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

/** Pastilles de choix (une seule active). */
function Choix<T extends string>({
  options,
  valeur,
  onChange,
  label,
}: {
  options: { cle: T; nom: string }[]
  valeur: T
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-1.5">
      {options.map((o) => (
        <button
          key={o.cle}
          type="button"
          role="radio"
          aria-checked={valeur === o.cle}
          onClick={() => onChange(o.cle)}
          className={cn(
            'h-8 rounded-full border px-3 text-[13px] transition-colors duration-200',
            valeur === o.cle
              ? 'border-transparent bg-brand-gradient text-white shadow-[0_0_12px_rgba(124,92,255,0.4)]'
              : 'border-white/15 text-white/80 hover:border-white/35 hover:text-white',
          )}
        >
          {o.nom}
        </button>
      ))}
    </div>
  )
}

/** Exemple miniature de chaque style de sous-titres, dans la couleur choisie. */
function ExempleStyle({ style, couleur }: { style: StyleSousTitres; couleur: string }) {
  const contour = { textShadow: '0 0 3px #000, 0 0 3px #000, 0 0 2px #000' }
  if (style === 'reseaux')
    return (
      <span className="text-[15px] font-bold text-white" style={contour}>
        le mot <span style={{ color: couleur }}>actif</span>
      </span>
    )
  if (style === 'impact')
    return (
      <span className="text-[17px] font-black tracking-tight uppercase" style={{ ...contour, color: couleur, fontFamily: 'Impact, sans-serif' }}>
        GROS TEXTE
      </span>
    )
  if (style === 'karaoke')
    return (
      <span className="text-[14px] font-bold" style={contour}>
        <span style={{ color: couleur }}>Une phrase </span>
        <span className="text-white">qui se remplit</span>
      </span>
    )
  return (
    <span className="text-[14px] text-white" style={{ textShadow: '0 1px 2px #000, 0 0 2px #000' }}>
      Sous-titre sobre
    </span>
  )
}

/** Ligne d'option : icône, titre, aide, interrupteur ; réglages détaillés dépliés quand elle est active. */
function Option({
  icone,
  titre,
  aide,
  active,
  onChange,
  children,
}: {
  icone: ReactNode
  titre: string
  aide: string
  active: boolean
  onChange: (v: boolean) => void
  children?: ReactNode
}) {
  return (
    <div className="border-b border-white/10 py-3.5 last:border-b-0">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 text-violet-300">{icone}</span>
        <div className="min-w-0 flex-1">
          <p className="text-[15px] font-medium text-white">{titre}</p>
          <p className="text-[13px] text-white/60">{aide}</p>
        </div>
        <Switch checked={active} onChange={onChange} label={titre} />
      </div>
      <AnimatePresence initial={false}>
        {active && children && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={voxTransition(0.25)}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-3 pt-3 pl-8">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function Etiquette({ children }: { children: ReactNode }) {
  return <p className="mb-1.5 text-[12px] tracking-wide text-white/50 uppercase">{children}</p>
}

/** Aperçu d'une image de la vidéo habillée ; la zone à flouter se trace directement dessus. */
function Apercu({ job, reglages, onZone }: { job: Job; reglages: Reglages; onZone: (z: Reglages['zone']) => void }) {
  const [image, setImage] = useState<string | null>(null)
  const [chargement, setChargement] = useState(false)
  const [erreur, setErreur] = useState<string | null>(null)
  const [instant, setInstant] = useState<number | null>(null)
  const [trace, setTrace] = useState<{ x0: number; y0: number; zone: Reglages['zone'] } | null>(null)
  const cadre = useRef<HTMLDivElement>(null)
  // Pendant le tracé, l'aperçu garde l'ancienne zone : il n'est recalculé qu'une fois la zone posée
  const cle = JSON.stringify({ reglages, instant })

  useEffect(() => {
    const controle = new AbortController()
    const minuterie = window.setTimeout(() => {
      setChargement(true)
      dressingPreview(job.id, reglages, instant, controle.signal)
        .then((url) => {
          setImage((ancienne) => {
            if (ancienne) URL.revokeObjectURL(ancienne)
            return url
          })
          setErreur(null)
        })
        .catch((e: Error) => {
          if (e.name !== 'AbortError') setErreur(e.message)
        })
        .finally(() => {
          if (!controle.signal.aborted) setChargement(false)
        })
    }, 350)
    return () => {
      window.clearTimeout(minuterie)
      controle.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id, cle])

  useEffect(() => () => {
    setImage((ancienne) => {
      if (ancienne) URL.revokeObjectURL(ancienne)
      return null
    })
  }, [])

  const point = (e: React.PointerEvent) => {
    const r = cadre.current!.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    }
  }
  const zone = trace?.zone ?? reglages.zone
  const duree = job.duree_s ?? 0

  return (
    <div>
      <div className="relative flex min-h-48 items-center justify-center overflow-hidden rounded-xl bg-black">
        {image ? (
          <div ref={cadre} className="relative w-fit">
            <img
              src={image}
              alt="Aperçu de la vidéo habillée"
              draggable={false}
              className={cn('block max-h-[440px] w-auto max-w-full transition-opacity duration-200', chargement && 'opacity-70')}
            />
            {reglages.flou && (
              <div
                className="group absolute inset-0 cursor-crosshair touch-none"
                onPointerDown={(e) => {
                  e.currentTarget.setPointerCapture(e.pointerId)
                  const p = point(e)
                  setTrace({ x0: p.x, y0: p.y, zone: { x: p.x, y: p.y, l: 0, h: 0 } })
                }}
                onPointerMove={(e) => {
                  if (!trace) return
                  const p = point(e)
                  setTrace({
                    ...trace,
                    zone: {
                      x: Math.min(trace.x0, p.x),
                      y: Math.min(trace.y0, p.y),
                      l: Math.abs(p.x - trace.x0),
                      h: Math.abs(p.y - trace.y0),
                    },
                  })
                }}
                onPointerUp={() => {
                  if (trace && trace.zone.l > 0.02 && trace.zone.h > 0.02) onZone(trace.zone)
                  setTrace(null)
                }}
              >
                <div
                  // Repère de réglage seulement (au survol ou pendant le tracé) : rien n'est dessiné dans la vidéo
                  className={cn(
                    'pointer-events-none absolute rounded-sm border border-dashed border-cyan-300/80 transition-opacity duration-200',
                    trace ? 'opacity-100' : 'opacity-0 group-hover:opacity-100',
                  )}
                  style={{ left: `${zone.x * 100}%`, top: `${zone.y * 100}%`, width: `${zone.l * 100}%`, height: `${zone.h * 100}%` }}
                />
              </div>
            )}
          </div>
        ) : (
          <p className="px-6 py-16 text-[14px] text-white/60">{erreur ?? 'Préparation de l\'aperçu…'}</p>
        )}
        {chargement && image && (
          <span className="absolute top-2 right-2 rounded-md bg-black/70 px-2 py-0.5 text-[12px] text-white/85">Mise à jour…</span>
        )}
      </div>
      {erreur && image && <p className="mt-2 text-[13px] text-destructive">{erreur}</p>}
      {reglages.flou && (
        <p className="mt-2 text-[13px] text-cyan-200/80">Tracez sur l'image la zone où apparaissent les anciens sous-titres. Les images sans texte ne sont pas modifiées.</p>
      )}
      {duree > 0 && (
        <label className="mt-3 flex items-center gap-3 text-[13px] text-white/70">
          <span className="shrink-0">Instant de l'aperçu</span>
          <input
            type="range"
            min={0}
            max={duree}
            step={0.1}
            value={instant ?? 0}
            aria-label="Instant de l'aperçu"
            onChange={(e) => setInstant(Number(e.target.value))}
            style={{ ['--fill' as string]: `${((instant ?? 0) / duree) * 100}%` }}
            className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-[linear-gradient(90deg,#8b5cf6_var(--fill),#ffffff26_var(--fill))] [&::-webkit-slider-thumb]:size-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
          />
          <span className="w-20 shrink-0 text-right tabular-nums">{instant === null ? '1re phrase' : formatTemps(instant)}</span>
        </label>
      )}
    </div>
  )
}

export function Habillage({ job }: { job: Job }) {
  const queryClient = useQueryClient()
  const enregistre: Reglages = { ...HABILLAGE_DEFAUT, ...job.options?.habillage }
  const [reglages, setReglages] = useState<Reglages>(enregistre)
  const changer = (modif: Partial<Reglages>) => setReglages((r) => ({ ...r, ...modif }))
  const appliquer = useMutation({
    mutationFn: () => setDressing(job.id, reglages),
    onSuccess: (j) => queryClient.setQueryData(['job', job.id], j),
  })
  const detecter = useMutation({
    mutationFn: () => detectOldSubtitles(job.id),
    onSuccess: ({ zone }) => zone && changer({ zone }),
  })

  const etat = job.habillage
  const enCours = etat?.etat === 'en_cours'
  const modifie = !meme(reglages, enregistre)
  const aJour = !modifie && (etat?.etat === 'termine' || (!actif(reglages) && !etat))
  const libelleBouton = enCours
    ? `Habillage en cours… ${etat.progression} %`
    : appliquer.isPending
      ? 'Lancement…'
      : aJour
        ? 'Vidéo à jour'
        : actif(reglages)
          ? 'Appliquer et réexporter la vidéo'
          : 'Retirer l\'habillage de la vidéo'

  return (
    <Panel className="mt-6 px-6 py-5">
      <div className="flex flex-wrap items-center gap-3">
        <p className="flex items-center gap-3 text-[20px] font-semibold text-white">
          <Sparkles className="size-6 text-violet-300" /> Habillage de la vidéo
        </p>
        {etat?.etat === 'termine' && (
          <span className="rounded-full bg-success px-3 py-0.5 text-[13px] font-medium text-black">Appliqué</span>
        )}
        {enCours && (
          <span className="rounded-full bg-violet-500/30 px-3 py-0.5 text-[13px] text-white">En cours · {etat.progression} %</span>
        )}
        <span className="flex-1" />
        {modifie && !enCours && (
          <button
            type="button"
            onClick={() => setReglages(enregistre)}
            className="flex items-center gap-1.5 text-[13px] text-white/60 hover:text-white"
          >
            <RotateCcw className="size-4" /> Annuler les changements
          </button>
        )}
      </div>
      <p className="mt-1 text-[14px] text-white/75">
        Facultatif : sous-titres traduits incrustés, effacement des anciens sous-titres, petit logo, image plus nette et
        fichier sans métadonnées. La vidéo est refaite en quelques instants, sans régénérer les voix.
      </p>

      <div className="mt-5 grid gap-6 *:min-w-0 xl:grid-cols-[1fr_420px]">
        <Apercu job={job} reglages={reglages} onZone={(zone) => changer({ zone })} />

        <div className="flex flex-col">
          <Option
            icone={<Captions className="size-5" />}
            titre="Sous-titres traduits incrustés"
            aide="Gravés dans l'image, calés sur la voix doublée."
            active={reglages.sous_titres}
            onChange={(v) => changer({ sous_titres: v })}
          >
            <div>
              <Etiquette>Style</Etiquette>
              <div role="radiogroup" aria-label="Style des sous-titres" className="grid grid-cols-2 gap-2">
                {STYLES.map((s) => (
                  <button
                    key={s.cle}
                    type="button"
                    role="radio"
                    aria-checked={reglages.style === s.cle}
                    title={s.aide}
                    onClick={() => changer({ style: s.cle })}
                    className={cn(
                      'flex flex-col items-center gap-1.5 rounded-xl border px-2 py-2.5 transition-colors duration-200',
                      reglages.style === s.cle
                        ? 'border-violet-400 bg-violet-500/15 shadow-[0_0_14px_rgba(124,92,255,0.35)]'
                        : 'border-white/12 hover:border-white/30',
                    )}
                  >
                    <span className="flex h-7 items-center rounded-md bg-[linear-gradient(135deg,#334155,#1e293b)] px-2">
                      <ExempleStyle style={s.cle} couleur={reglages.couleur} />
                    </span>
                    <span className="text-[13px] text-white">{s.nom}</span>
                  </button>
                ))}
              </div>
            </div>
            <div>
              <Etiquette>Animation</Etiquette>
              <Choix options={ANIMATIONS} valeur={reglages.animation} onChange={(animation) => changer({ animation })} label="Animation" />
            </div>
            <div className="flex flex-wrap gap-x-6 gap-y-3">
              <div>
                <Etiquette>Taille</Etiquette>
                <Choix options={TAILLES} valeur={reglages.taille} onChange={(taille) => changer({ taille })} label="Taille" />
              </div>
              <div>
                <Etiquette>Position</Etiquette>
                <Choix options={POSITIONS} valeur={reglages.position} onChange={(position) => changer({ position })} label="Position" />
              </div>
            </div>
            <div>
              <Etiquette>Couleur d'accent</Etiquette>
              <div role="radiogroup" aria-label="Couleur d'accent" className="flex gap-2">
                {COULEURS.map((c) => (
                  <button
                    key={c}
                    type="button"
                    role="radio"
                    aria-checked={reglages.couleur === c}
                    aria-label={c}
                    onClick={() => changer({ couleur: c })}
                    className={cn(
                      'size-7 rounded-full border-2 transition-transform duration-200 hover:scale-110',
                      reglages.couleur === c ? 'border-white shadow-[0_0_10px_rgba(255,255,255,0.5)]' : 'border-white/20',
                    )}
                    style={{ background: c }}
                  />
                ))}
              </div>
            </div>
          </Option>

          <Option
            icone={<Eraser className="size-5" />}
            titre="Effacer les anciens sous-titres"
            aide="Retire les sous-titres incrustés d'origine dans la zone choisie : seules les lettres sont reconstruites, sans flou ni cadre."
            active={reglages.flou}
            onChange={(v) => changer({ flou: v })}
          >
            <div className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() => detecter.mutate()}
                disabled={detecter.isPending}
                className="flex h-8 items-center gap-1.5 rounded-full border border-cyan-400/50 px-3 text-[13px] text-cyan-200 hover:border-cyan-300 disabled:opacity-60"
              >
                <ScanText className="size-4" /> {detecter.isPending ? 'Recherche…' : 'Détecter automatiquement'}
              </button>
              <button
                type="button"
                onClick={() => changer({ zone: HABILLAGE_DEFAUT.zone })}
                className="h-8 rounded-full border border-white/15 px-3 text-[13px] text-white/80 hover:border-white/35 hover:text-white"
              >
                Bande du bas
              </button>
              <button
                type="button"
                onClick={() => changer({ zone: { x: 0.08, y: 0.04, l: 0.84, h: 0.18 } })}
                className="h-8 rounded-full border border-white/15 px-3 text-[13px] text-white/80 hover:border-white/35 hover:text-white"
              >
                Bande du haut
              </button>
            </div>
            {detecter.data && !detecter.data.zone && (
              <p className="text-[13px] text-white/60">Aucun sous-titre trouvé : tracez la zone sur l'aperçu.</p>
            )}
            {detecter.error && <p className="text-[13px] text-destructive">{(detecter.error as Error).message}</p>}
          </Option>

          <Option
            icone={<Stamp className="size-5" />}
            titre="Petit logo VoxLingo"
            aide="Discret et semi-transparent, dans un coin."
            active={reglages.logo}
            onChange={(v) => changer({ logo: v })}
          >
            <Choix options={COINS} valeur={reglages.coin} onChange={(coin) => changer({ coin })} label="Coin du logo" />
          </Option>

          <Option
            icone={<ImageUpscale className="size-5" />}
            titre="Améliorer légèrement la qualité"
            aide="Débruitage doux et netteté, sans changer le format ni la taille de la vidéo."
            active={reglages.qualite}
            onChange={(v) => changer({ qualite: v })}
          />

          <Option
            icone={<ShieldCheck className="size-5" />}
            titre="Supprimer les métadonnées"
            aide="Ni titre, ni date, ni lieu, ni appareil, ni logiciel dans le fichier exporté."
            active={reglages.sans_metadonnees}
            onChange={(v) => changer({ sans_metadonnees: v })}
          />

          <div className="mt-4 flex flex-col gap-2">
            {enCours && (
              <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                <motion.div
                  className="h-full bg-brand-gradient"
                  animate={{ width: `${etat.progression}%` }}
                  transition={voxTransition(0.4)}
                />
              </div>
            )}
            <PrimaryButton
              onClick={() => appliquer.mutate()}
              disabled={enCours || appliquer.isPending || aJour}
              className="h-12 rounded-2xl"
            >
              <Wand2 className="size-5" /> {libelleBouton}
            </PrimaryButton>
            <p className="text-center text-[12px] text-white/50">Les voix ne sont pas régénérées : seule l'image est refaite.</p>
            {(appliquer.error || etat?.etat === 'erreur') && (
              <p className="flex items-start gap-2 text-[13px] text-destructive">
                <TriangleAlert className="mt-0.5 size-4 shrink-0" />
                {appliquer.error ? (appliquer.error as Error).message : etat?.erreur}
              </p>
            )}
          </div>
        </div>
      </div>
    </Panel>
  )
}
