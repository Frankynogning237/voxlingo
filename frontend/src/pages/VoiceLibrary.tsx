// Bibliothèque de voix : 51 voix (XTTS v2, tous âges, variantes de style) et voix personnelles.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import { Loader2, Mic, Plus, Sparkles, Trash2, Upload, X } from 'lucide-react'
import { useRef, useState } from 'react'

import { Panel } from '@/components/Controls'
import { Gradient, PageHeader } from '@/components/Layout'
import { MiniPlayer } from '@/components/MiniPlayer'
import { OutlineButton, PrimaryButton } from '@/components/PrimaryButton'
import {
  LIBELLES_AGE,
  addVoice,
  deleteVoice,
  generatePreviews,
  listVoices,
  updateVoice,
  voicePreviewUrl,
  type AgeVoix,
  type Sexe,
  type Voix,
} from '@/lib/api'
import { cn } from '@/lib/utils'
import { cardVariants, staggerVariants, voxTransition } from '@/motion/variants'

type FiltreSexe = 'tous' | Sexe
type FiltreAge = 'tous' | AgeVoix
const COULEUR: Record<Sexe, string> = { femme: '#ec4899', homme: '#3b82f6' }

function Puce<T extends string>({ valeur, actif, onClick, children }: { valeur: T; actif: boolean; onClick: (v: T) => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={() => onClick(valeur)}
      className={cn(
        'h-10 rounded-full px-4 text-[14px] transition-colors duration-200',
        actif ? 'bg-brand-gradient text-white shadow-[0_0_14px_rgba(124,92,255,0.4)]' : 'border border-white/12 text-white/85 hover:border-white/30',
      )}
    >
      {children}
    </button>
  )
}

function CarteVoix({ voix }: { voix: Voix }) {
  const queryClient = useQueryClient()
  const actualiser = () => queryClient.invalidateQueries({ queryKey: ['voix'] })
  const modifier = useMutation({ mutationFn: (m: { age?: AgeVoix; sexe?: Sexe }) => updateVoice(voix.id, m), onSuccess: actualiser })
  const supprimer = useMutation({ mutationFn: () => deleteVoice(voix.id), onSuccess: actualiser })
  return (
    <motion.div variants={cardVariants} layout exit={{ opacity: 0, scale: 0.97 }}>
      <Panel className="flex h-full flex-col gap-3 px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="flex size-11 items-center justify-center rounded-full" style={{ background: `${COULEUR[voix.sexe]}33`, color: COULEUR[voix.sexe] }}>
            <Mic className="size-5" />
          </span>
          <div className="flex-1">
            <p className="text-[17px] font-medium text-white">{voix.nom}</p>
            <p className="text-[13px] text-white/70">
              {voix.sexe === 'femme' ? 'Femme' : 'Homme'}
              {voix.registre ? ` · voix ${voix.registre}` : ''}
              {voix.f0_hz ? ` · ${Math.round(voix.f0_hz)} Hz` : ''}
              {voix.source === 'perso' ? ' · voix personnelle' : ''}
            </p>
            {voix.style && (
              <span className="mt-1 inline-block rounded-full border border-violet-400/40 bg-violet-500/10 px-2.5 py-0.5 text-[12px] text-violet-200">
                Style {voix.style.toLowerCase()}
              </span>
            )}
          </div>
          {voix.source === 'perso' && (
            <button type="button" onClick={() => supprimer.mutate()} aria-label="Supprimer cette voix" className="text-white/60 hover:text-destructive">
              <Trash2 className="size-5" />
            </button>
          )}
        </div>
        <MiniPlayer src={voix.apercu ? voicePreviewUrl(voix.id) : null} couleur={COULEUR[voix.sexe]} libelle="Aperçu" vide="Aperçu à générer" />
        <div className="flex items-center gap-2">
          <span className="text-[13px] text-white/70" title="L'âge ne se mesure pas : ajustez-le après écoute.">
            Âge apparent :
          </span>
          {(Object.keys(LIBELLES_AGE) as AgeVoix[]).map((age) => (
            <button
              key={age}
              type="button"
              onClick={() => modifier.mutate({ age })}
              className={cn(
                'rounded-full px-2.5 py-0.5 text-[12px] transition-colors',
                voix.age === age ? 'bg-violet-500/80 text-white' : 'border border-white/12 text-white/70 hover:text-white',
              )}
            >
              {LIBELLES_AGE[age]}
            </button>
          ))}
        </div>
      </Panel>
    </motion.div>
  )
}

function AjoutVoix({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const fichier = useRef<HTMLInputElement>(null)
  const [choisi, setChoisi] = useState<File | null>(null)
  const [nom, setNom] = useState('')
  const [sexe, setSexe] = useState<Sexe>('femme')
  const [age, setAge] = useState<AgeVoix>('adulte')
  const ajouter = useMutation({
    mutationFn: () => addVoice(choisi!, nom, sexe, age),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['voix'] })
      onClose()
    },
  })
  return (
    <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={voxTransition(0.3)}>
      <Panel className="mb-6 flex flex-wrap items-end gap-4 px-6 py-5">
        <div className="w-full">
          <p className="flex items-center justify-between text-[18px] font-semibold text-white">
            Ajouter une voix personnelle
            <button type="button" onClick={onClose} aria-label="Fermer" className="text-white/70 hover:text-white">
              <X className="size-5" />
            </button>
          </p>
          <p className="text-[14px] text-white/70">
            Un enregistrement clair de 6 à 15 secondes, sans musique (WAV, MP3, M4A). Ne clonez que des voix dont vous avez l'accord.
          </p>
        </div>
        <button
          type="button"
          onClick={() => fichier.current?.click()}
          className="flex h-11 items-center gap-2 rounded-xl border border-dashed border-violet-400/60 px-4 text-[14px] text-white hover:bg-white/5"
        >
          <Upload className="size-4" /> {choisi ? choisi.name : 'Choisir un enregistrement'}
        </button>
        <input ref={fichier} type="file" accept="audio/*" className="hidden" onChange={(e) => setChoisi(e.target.files?.[0] ?? null)} />
        <label className="flex flex-col gap-1 text-[13px] text-white/70">
          Nom
          <input value={nom} onChange={(e) => setNom(e.target.value)} placeholder="Ex. : Mamie Rose" className="h-11 w-48 rounded-xl border border-white/12 bg-[#0b0d16] px-3 text-[14px] text-white outline-none focus:border-violet-400/70" />
        </label>
        <label className="flex flex-col gap-1 text-[13px] text-white/70">
          Sexe
          <select value={sexe} onChange={(e) => setSexe(e.target.value as Sexe)} className="h-11 rounded-xl border border-white/12 bg-[#0b0d16] px-3 text-[14px] text-white">
            <option value="femme">Femme</option>
            <option value="homme">Homme</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-[13px] text-white/70">
          Âge
          <select value={age} onChange={(e) => setAge(e.target.value as AgeVoix)} className="h-11 rounded-xl border border-white/12 bg-[#0b0d16] px-3 text-[14px] text-white">
            {(Object.keys(LIBELLES_AGE) as AgeVoix[]).map((a) => (
              <option key={a} value={a}>
                {LIBELLES_AGE[a]}
              </option>
            ))}
          </select>
        </label>
        <PrimaryButton disabled={!choisi || !nom.trim() || ajouter.isPending} onClick={() => ajouter.mutate()}>
          {ajouter.isPending ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />} Ajouter
        </PrimaryButton>
        {ajouter.error && <p className="w-full text-[14px] text-destructive">{(ajouter.error as Error).message}</p>}
      </Panel>
    </motion.div>
  )
}

export function VoiceLibrary() {
  const queryClient = useQueryClient()
  const { data } = useQuery({
    queryKey: ['voix'],
    queryFn: listVoices,
    refetchInterval: (q) => (q.state.data?.apercus.en_cours ? 4000 : false),
  })
  const [sexe, setSexe] = useState<FiltreSexe>('tous')
  const [age, setAge] = useState<FiltreAge>('tous')
  const [ajout, setAjout] = useState(false)
  const apercus = useMutation({ mutationFn: generatePreviews, onSuccess: () => queryClient.invalidateQueries({ queryKey: ['voix'] }) })

  const voix = (data?.voix ?? []).filter((v) => (sexe === 'tous' || v.sexe === sexe) && (age === 'tous' || v.age === age))
  const sansApercu = (data?.voix ?? []).filter((v) => !v.apercu).length
  const enCours = data?.apercus.en_cours

  return (
    <>
      <PageHeader
        back={{ page: 'parametres' }}
        title={
          <>
            Bibliothèque de <Gradient>voix</Gradient>
          </>
        }
        subtitle="Choisissez une voix par personne de la vidéo, ou gardez sa voix d'origine clonée."
      />

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <Puce valeur="tous" actif={sexe === 'tous'} onClick={setSexe}>Toutes</Puce>
        <Puce valeur="femme" actif={sexe === 'femme'} onClick={setSexe}>Femmes</Puce>
        <Puce valeur="homme" actif={sexe === 'homme'} onClick={setSexe}>Hommes</Puce>
        <span className="mx-2 h-6 w-px bg-white/15" />
        <Puce valeur="tous" actif={age === 'tous'} onClick={setAge}>Tous âges</Puce>
        {(Object.keys(LIBELLES_AGE) as AgeVoix[]).map((a) => (
          <Puce key={a} valeur={a} actif={age === a} onClick={setAge}>
            {LIBELLES_AGE[a]}
          </Puce>
        ))}
        <span className="flex-1" />
        {sansApercu > 0 && (
          <OutlineButton onClick={() => apercus.mutate()} disabled={enCours} className="h-11" title="Chaque voix lit une phrase d'exemple ; son registre est ensuite mesuré. Fermez le navigateur pour libérer la mémoire.">
            {enCours ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
            {enCours ? 'Génération des aperçus…' : `Générer les aperçus (${sansApercu})`}
          </OutlineButton>
        )}
        <PrimaryButton onClick={() => setAjout(true)}>
          <Plus className="size-4" /> Ajouter ma voix
        </PrimaryButton>
      </div>
      {data?.apercus.erreur && (
        <p className="mb-4 rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-2 text-[14px] text-destructive">{data.apercus.erreur}</p>
      )}
      <AnimatePresence>{ajout && <AjoutVoix onClose={() => setAjout(false)} />}</AnimatePresence>

      <motion.div variants={staggerVariants} initial="hidden" animate="visible" className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        <AnimatePresence>
          {voix.map((v) => (
            <CarteVoix key={v.id} voix={v} />
          ))}
        </AnimatePresence>
      </motion.div>
      {data && voix.length === 0 && <p className="py-10 text-center text-white/70">Aucune voix ne correspond à ces filtres.</p>}
    </>
  )
}
