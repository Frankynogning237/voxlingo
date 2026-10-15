import { AnimatePresence, motion } from 'framer-motion'
import { PlugZap, X } from 'lucide-react'
import { useState } from 'react'

import { PrimaryButton } from '@/components/PrimaryButton'
import { API_URL, API_URL_DEFAUT, CLE, enregistrerConnexion } from '@/lib/api'
import { voxTransition } from '@/motion/variants'

/**
 * Fenêtre « Connexion au moteur » : adresse du moteur (ce PC, ou Kaggle/Colab via un tunnel)
 * et clé d'accès. Le carnet Kaggle/Colab affiche aussi un lien qui remplit ces champs tout seul.
 */
export function ConnexionMoteur({ ouvert, onFermer, erreur }: { ouvert: boolean; onFermer: () => void; erreur?: string }) {
  const [moteur, setMoteur] = useState(API_URL)
  const [cle, setCle] = useState(CLE)

  const connecter = (adresse: string, motDePasse: string) => {
    enregistrerConnexion(adresse, motDePasse)
    window.location.reload()
  }

  return (
    <AnimatePresence>
      {ouvert && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
          onClick={onFermer}
        >
          <motion.form
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8 }}
            transition={voxTransition(0.3)}
            onClick={(e) => e.stopPropagation()}
            onSubmit={(e) => {
              e.preventDefault()
              connecter(moteur, cle)
            }}
            className="relative w-full max-w-[520px] rounded-2xl border border-white/12 bg-[#11131c] p-7 shadow-2xl"
          >
            <button type="button" onClick={onFermer} aria-label="Fermer" className="absolute top-5 right-5 text-white/60 hover:text-white">
              <X className="size-5" />
            </button>
            <h2 className="flex items-center gap-3 text-[22px] font-semibold text-white">
              <PlugZap className="size-6 text-violet-400" /> Connexion au moteur
            </h2>
            <p className="mt-2 text-[14px] leading-relaxed text-white/70">
              Le moteur fait le doublage : sur ce PC, ou sur Kaggle / Colab (GPU gratuit). Collez l'adresse
              et la clé affichées à la fin du carnet, ou ouvrez directement le lien qu'il vous donne.
            </p>
            {erreur && <p className="mt-4 rounded-lg bg-destructive/15 px-3 py-2 text-[14px] text-red-300">{erreur}</p>}

            <label className="mt-5 block text-[14px] text-white/80">
              Adresse du moteur
              <input
                value={moteur}
                onChange={(e) => setMoteur(e.target.value)}
                placeholder="https://exemple.trycloudflare.com"
                className="mt-1.5 h-11 w-full rounded-xl border border-white/12 bg-[#0b0d14] px-3.5 text-[15px] text-white outline-none focus:border-violet-500"
              />
            </label>
            <label className="mt-4 block text-[14px] text-white/80">
              Clé d'accès <span className="text-white/45">(vide pour le moteur de ce PC)</span>
              <input
                value={cle}
                onChange={(e) => setCle(e.target.value)}
                type="password"
                autoComplete="off"
                className="mt-1.5 h-11 w-full rounded-xl border border-white/12 bg-[#0b0d14] px-3.5 text-[15px] text-white outline-none focus:border-violet-500"
              />
            </label>

            <div className="mt-7 flex items-center justify-between gap-3">
              <button
                type="button"
                onClick={() => connecter(API_URL_DEFAUT, '')}
                className="text-[14px] text-white/60 underline-offset-4 hover:text-white hover:underline"
              >
                Moteur de ce PC
              </button>
              <PrimaryButton type="submit">Se connecter</PrimaryButton>
            </div>
          </motion.form>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
