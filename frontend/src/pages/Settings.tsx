// Paramètres — maquette reference/VoxLingo Neon Dashboard Split Screen.png (partie droite)
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowRight,
  Box,
  Library,
  Check,
  CircleCheck,
  Cpu,
  Database,
  FolderOpen,
  Info,
  Palette,
  RotateCcw,
  SlidersHorizontal,
  TriangleAlert,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'

import { Panel, Slider, Switch } from '@/components/Controls'
import { Flag } from '@/components/Flag'
import { Gradient, PageHeader } from '@/components/Layout'
import { OutlineButton, PrimaryButton } from '@/components/PrimaryButton'
import { getSettings, openProjectsFolder, saveSettings, type Parametres, type ValeursParametres } from '@/lib/api'
import { href } from '@/lib/router'
import { cn } from '@/lib/utils'
import { cardVariants, staggerVariants, voxTransition } from '@/motion/variants'

function Carte({ icone, titre, children }: { icone: ReactNode; titre: string; children: ReactNode }) {
  return (
    <motion.div variants={cardVariants}>
      <Panel className="h-full px-6 py-5">
        <p className="mb-5 flex items-center gap-3 text-[22px] font-semibold text-white">
          {icone} {titre}
        </p>
        {children}
      </Panel>
    </motion.div>
  )
}

function Aide({ texte }: { texte: string }) {
  return (
    <span title={texte} className="text-white/60">
      <Info className="size-4" />
    </span>
  )
}

function Jauge({ valeur, max, couleur = 'bg-brand-gradient' }: { valeur: number; max: number; couleur?: string }) {
  return (
    <div className="h-2 flex-1 overflow-hidden rounded-full bg-[#232836]">
      <motion.div
        className={cn('h-full rounded-full', couleur)}
        initial={{ width: 0 }}
        animate={{ width: `${max ? Math.min(100, (valeur / max) * 100) : 0}%` }}
        transition={voxTransition(0.4)}
      />
    </div>
  )
}

function Choix({
  libelle,
  aide,
  valeur,
  options,
  onChange,
}: {
  libelle: string
  aide: string
  valeur: string
  options: { valeur: string; libelle: string }[]
  onChange: (v: string) => void
}) {
  return (
    <label className="flex items-center gap-3 py-2.5">
      <span className="flex flex-1 items-center gap-2 text-[15px] text-white">
        {libelle} <Aide texte={aide} />
      </span>
      <select
        value={valeur}
        onChange={(e) => onChange(e.target.value)}
        className="h-10 w-56 cursor-pointer rounded-xl border border-white/12 bg-[#0b0d16] px-3 text-[14px] text-white outline-none focus:border-violet-400/70"
      >
        {options.map((o) => (
          <option key={o.valeur} value={o.valeur}>
            {o.libelle}
          </option>
        ))}
      </select>
    </label>
  )
}

const formatGo = (o: number) => `${(o / 1024 ** 3).toFixed(1)} Go`

export function Settings() {
  const queryClient = useQueryClient()
  const { data, error } = useQuery({ queryKey: ['parametres'], queryFn: getSettings })
  const [valeurs, setValeurs] = useState<ValeursParametres | null>(null)
  const [message, setMessage] = useState<{ ok: boolean; texte: string } | null>(null)

  useEffect(() => {
    if (data) setValeurs(data.valeurs)
  }, [data])

  const enregistrer = useMutation({
    mutationFn: (v: ValeursParametres) => saveSettings(v),
    onSuccess: (d: Parametres) => {
      queryClient.setQueryData(['parametres'], d)
      void queryClient.invalidateQueries({ queryKey: ['health'] })
      setMessage({ ok: true, texte: 'Paramètres enregistrés.' })
      setTimeout(() => setMessage(null), 3000)
    },
    onError: (e) => setMessage({ ok: false, texte: (e as Error).message }),
  })
  const ouvrir = useMutation({ mutationFn: openProjectsFolder })

  if (error) return <p className="text-destructive">{(error as Error).message}</p>
  if (!data || !valeurs) return <PageHeader title="Paramètres" subtitle="Chargement…" />

  const set = <K extends keyof ValeursParametres>(cle: K, v: ValeursParametres[K]) => setValeurs({ ...valeurs, [cle]: v })
  const modifie = JSON.stringify(valeurs) !== JSON.stringify(data.valeurs)
  const { materiel, stockage, choix } = data
  const gpu = materiel.device === 'cuda'

  return (
    <>
      <PageHeader
        title={
          <>
            <Gradient>Paramètres</Gradient>
          </>
        }
        subtitle="Personnalisez VoxLingo selon vos besoins."
      />

      <motion.div variants={staggerVariants} initial="hidden" animate="visible" className="grid gap-6 *:min-w-0 xl:grid-cols-2">
        <Carte icone={<Cpu className="size-7" strokeWidth={1.5} />} titre="Matériel">
          <p className={cn('mb-4 flex items-center gap-2.5 text-[16px]', gpu ? 'text-success' : 'text-warning')}>
            {gpu ? <CircleCheck className="size-5 fill-success text-[#0e1119]" /> : <TriangleAlert className="size-5" />}
            {gpu ? `GPU NVIDIA détecté : ${materiel.gpu}` : 'Aucune carte NVIDIA : traitement sur le processeur'}
          </p>
          {gpu && (
            <div className="mb-4">
              <p className="mb-2 flex items-center gap-2 text-[14px] text-white/85">
                Mémoire vidéo (VRAM) <Aide texte="Mémoire de la carte graphique utilisée par les modèles d'IA." />
              </p>
              <div className="flex flex-wrap items-center gap-4">
                <Jauge valeur={materiel.vram_utilisee_go} max={materiel.vram_totale_go} />
                <span className="text-[14px] text-white tabular-nums">
                  {materiel.vram_utilisee_go.toFixed(1)} / {materiel.vram_totale_go.toFixed(1)} Go
                </span>
              </div>
            </div>
          )}
          <p className="mb-2 flex items-center gap-2 text-[14px] text-white/85">
            Mémoire vive (RAM) <Aide texte="La voix clonée (XTTS) a besoin d'environ 3 Go libres : fermez le navigateur et les autres applications pendant la génération." />
          </p>
          <div className="mb-4 flex items-center gap-4">
            <Jauge valeur={materiel.ram_utilisee_go} max={materiel.ram_totale_go} couleur={materiel.ram_libre_go < 3 ? 'bg-warning' : 'bg-brand-gradient'} />
            <span className="text-[14px] text-white tabular-nums">
              {materiel.ram_utilisee_go.toFixed(1)} / {materiel.ram_totale_go.toFixed(1)} Go
            </span>
          </div>
          <div className="mb-4 flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3 text-[14px] text-white/80">
            <Info className="size-4 shrink-0" />
            {gpu
              ? 'Votre GPU est utilisé pour le traitement IA.'
              : `Processeur : ${materiel.processeur}. Comptez environ 10 à 15 minutes de calcul par minute de vidéo pour un doublage complet.`}
          </div>
          <Choix
            libelle="Matériel utilisé"
            aide="« Automatique » utilise la carte NVIDIA si elle est présente."
            valeur={valeurs.force_device}
            options={[
              { valeur: 'auto', libelle: 'Automatique' },
              { valeur: 'cpu', libelle: 'Processeur (CPU)' },
              { valeur: 'cuda', libelle: 'Carte NVIDIA (GPU)' },
            ]}
            onChange={(v) => set('force_device', v as ValeursParametres['force_device'])}
          />
        </Carte>

        <Carte icone={<Box className="size-7" strokeWidth={1.5} />} titre="Modèles IA">
          <Choix
            libelle="Modèle de transcription"
            aide="Whisper : plus le modèle est grand, plus la transcription est précise, mais lente. « Automatique » utilise Large V3 Turbo, bien plus fidèle ; sur un simple processeur, « Small » transcrit environ 1,7 fois plus vite, au prix de contresens."
            valeur={valeurs.whisper_model}
            options={choix.transcription}
            onChange={(v) => set('whisper_model', v)}
          />
          <Choix
            libelle="Modèle de traduction"
            aide="Qwen 2.5 7B traduit nettement mieux l'argot et les expressions que 3B, mais demande ~5 Go de RAM et est ~3 fois plus lent sur processeur. Qwen3 14B, le plus fin, est conseillé sur GPU (Kaggle). Un modèle absent est téléchargé au premier usage."
            valeur={valeurs.ollama_model}
            options={choix.traduction}
            onChange={(v) => set('ollama_model', v)}
          />
          <Choix
            libelle="Modèle de voix (TTS)"
            aide="Chatterbox (Resemble AI, licence MIT) est plus expressif : son intensité suit le ton et l'émotion de chaque phrase (23 langues). XTTS v2 couvre toutes les langues (licence CPML : usage personnel). Les deux clonent la voix à partir de 6 à 15 s d'enregistrement ; si Chatterbox échoue, XTTS prend le relais."
            valeur={valeurs.moteur_voix ?? 'auto'}
            options={choix.voix ?? [{ valeur: 'auto', libelle: 'Automatique' }]}
            onChange={(v) => set('moteur_voix', v as ValeursParametres['moteur_voix'])}
          />
          <a
            href={href({ page: 'voix' })}
            className="mt-3 flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3 text-[14px] text-white transition-colors hover:border-violet-400/50"
          >
            <span className="flex items-center gap-2">
              <Library className="size-4 text-violet-300" /> Bibliothèque de voix : 51 voix (tous âges, variantes de style) + vos voix
            </span>
            <ArrowRight className="size-4" />
          </a>
        </Carte>

        <Carte icone={<Database className="size-7" strokeWidth={1.5} />} titre="Stockage">
          <p className="mb-2 flex items-center gap-2 text-[14px] text-white/85">
            Dossier de projets <Aide texte="Vidéos, voix générées et vidéos doublées de chaque projet." />
          </p>
          <div className="mb-4 flex gap-2">
            <code className="flex h-11 flex-1 items-center truncate rounded-xl border border-white/12 bg-[#0b0d16] px-4 text-[14px] text-white">
              {stockage.dossier}
            </code>
            <button
              type="button"
              onClick={() => ouvrir.mutate()}
              aria-label="Ouvrir le dossier dans l'Explorateur"
              title="Ouvrir le dossier dans l'Explorateur"
              className="flex size-11 items-center justify-center rounded-xl border border-white/12 text-white hover:border-white/30"
            >
              <FolderOpen className="size-5" />
            </button>
          </div>
          <p className="mb-2 flex items-center justify-between text-[14px] text-white/85">
            <span className="flex items-center gap-2">
              Espace utilisé par VoxLingo <Aide texte="Projets, modèles d'IA téléchargés compris." />
            </span>
            <span className="text-white tabular-nums">
              {formatGo(stockage.projets_octets + stockage.modeles_octets)} / {formatGo(stockage.disque_total_octets)}
            </span>
          </p>
          <Jauge valeur={stockage.projets_octets + stockage.modeles_octets} max={stockage.disque_total_octets} />
          <div className="mt-4 flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3 text-[14px] text-white/80">
            <Info className="size-4 shrink-0" />
            {formatGo(stockage.disque_libre_octets)} disponibles · projets {formatGo(stockage.projets_octets)} · modèles{' '}
            {formatGo(stockage.modeles_octets)}
          </div>
        </Carte>

        <Carte icone={<Palette className="size-7" strokeWidth={1.5} />} titre="Apparence">
          <div className="mb-5 flex items-center gap-6">
            <span className="flex-1 text-[15px] text-white">Thème</span>
            <button type="button" aria-pressed="true" className="flex h-24 w-32 flex-col items-center justify-center gap-2 rounded-2xl border-gradient text-[14px] text-white shadow-[0_0_18px_rgba(124,92,255,0.35)] [--gradient-fill:#141532]">
              <span className="flex size-7 items-center justify-center rounded-full border-2 border-violet-400">
                <span className="size-3 rounded-full bg-violet-400" />
              </span>
              Sombre
            </button>
            <button
              type="button"
              disabled
              title="Le thème clair arrivera dans une prochaine version"
              className="flex h-24 w-32 cursor-not-allowed flex-col items-center justify-center gap-2 rounded-2xl bg-white/90 text-[14px] text-black/70 opacity-60"
            >
              <span className="size-7 rounded-full border-2 border-black/30" />
              Clair (bientôt)
            </button>
          </div>
          <div className="flex items-center gap-3">
            <span className="flex-1 text-[15px] text-white">Langue de l'interface</span>
            <span className="flex h-10 w-56 items-center gap-2 rounded-xl border border-white/12 px-3 text-[14px] text-white">
              <Flag langue="fr" /> Français
            </span>
          </div>
        </Carte>

        <div className="xl:col-span-2">
          <Carte icone={<SlidersHorizontal className="size-7" strokeWidth={1.5} />} titre="Doublage par défaut">
            <div className="grid gap-x-10 gap-y-4 *:min-w-0 lg:grid-cols-2">
              <label className="flex flex-wrap items-center gap-4">
                <span className="flex flex-1 items-center gap-2 text-[15px] text-white">
                  Reformuler plutôt qu'accélérer <Aide texte="Une phrase trop longue est reformulée plus brièvement par l'IA, pour garder une voix naturelle." />
                </span>
                <Switch
                  checked={valeurs.strategie_duree === 'reformuler'}
                  onChange={(v) => set('strategie_duree', v ? 'reformuler' : 'accelerer')}
                  label="Reformuler plutôt qu'accélérer"
                />
              </label>
              <label className="flex flex-wrap items-center gap-4">
                <span className="flex flex-1 items-center gap-2 text-[15px] text-white">
                  Générer les sous-titres <Aide texte="Fichier .srt dans la langue cible." />
                </span>
                <Switch checked={valeurs.sous_titres} onChange={(v) => set('sous_titres', v)} label="Générer les sous-titres" />
              </label>
              <label className="flex flex-wrap items-center gap-4">
                <span className="flex flex-1 items-center gap-2 text-[15px] text-white">
                  Conserver le fond sonore <Aide texte="Musique et ambiance d'origine sous la nouvelle voix." />
                </span>
                <Switch checked={valeurs.conserver_fond} onChange={(v) => set('conserver_fond', v)} label="Conserver le fond sonore" />
              </label>
              <div className="flex flex-wrap items-center gap-4">
                <span className="text-[15px] text-white">Volume du fond</span>
                <Slider value={valeurs.volume_fond} onChange={(v) => set('volume_fond', v)} label="Volume du fond par défaut" />
                <span className="w-12 text-right text-[15px] text-white tabular-nums">{valeurs.volume_fond} %</span>
              </div>
            </div>
          </Carte>
        </div>
      </motion.div>

      <div className="mt-6 flex flex-wrap items-center gap-4">
        <OutlineButton onClick={() => setValeurs(data.defauts)} className="h-12 px-6">
          <RotateCcw className="size-4" /> Réinitialiser les paramètres
        </OutlineButton>
        <AnimatePresence>
          {message && (
            <motion.p
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={cn('text-[14px]', message.ok ? 'text-success' : 'text-destructive')}
            >
              {message.texte}
            </motion.p>
          )}
        </AnimatePresence>
        <span className="flex-1" />
        <PrimaryButton size="lg" disabled={!modifie || enregistrer.isPending} onClick={() => enregistrer.mutate(valeurs)} className="w-full sm:w-auto sm:min-w-72">
          <Check className="size-5" /> Enregistrer les paramètres
        </PrimaryButton>
      </div>
    </>
  )
}
