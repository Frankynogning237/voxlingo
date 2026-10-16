// Variantes de voix et comparaison des réglages de timbre (page Locuteurs).
import type { Timbre } from '@/lib/api'

/** Variantes de voix proposées en un clic (combinaisons de réglages). */
export const VARIANTES: { nom: string; aide: string; timbre: Timbre }[] = [
  { nom: 'Naturelle', aide: 'Voix telle quelle', timbre: {} },
  { nom: 'Plus grave', aide: 'Plus basse et plus pleine', timbre: { ton: -2, formant: 0.95, graves: 2 } },
  { nom: 'Plus aiguë', aide: 'Plus haute et plus légère', timbre: { ton: 2, formant: 1.04 } },
  { nom: 'Chaleureuse', aide: 'Ronde et chaude', timbre: { graves: 4, aigus: -2, formant: 0.97 } },
  { nom: 'Claire', aide: 'Lumineuse et nette', timbre: { aigus: 4, graves: -1, formant: 1.04 } },
  { nom: 'Douce', aide: 'Feutrée, sans dureté', timbre: { aigus: -5, ton: -1 } },
  { nom: 'Radio', aide: 'Présente, comme une voix d’animateur', timbre: { graves: 3, aigus: 3 } },
  { nom: 'Robuste', aide: 'Grande carrure, voix profonde', timbre: { ton: -3, formant: 0.9, graves: 3 } },
  { nom: 'Menue', aide: 'Petite carrure, voix fine', timbre: { ton: 3, formant: 1.1, graves: -2 } },
  { nom: 'Enfantine', aide: 'Rapproche la voix de celle d’un enfant', timbre: { ton: 5, formant: 1.18, graves: -3 } },
  { nom: 'Âgée', aide: 'Légèrement tremblante et voilée', timbre: { ton: -1, vibrato: 0.12, aigus: -2 } },
  { nom: 'Dans une pièce', aide: 'Légère réverbération', timbre: { espace: 0.45 } },
  { nom: 'Téléphone', aide: 'Comme à travers un téléphone', timbre: { telephone: 1 } },
]

/** Réglages identiques (les valeurs neutres comptent comme absentes). */
export function memeTimbre(a: Timbre | undefined, b: Timbre | undefined) {
  const propre = (t: Timbre | undefined) =>
    JSON.stringify(
      Object.entries(t ?? {})
        .filter(([k, v]) => v !== undefined && v !== (k === 'formant' ? 1 : 0))
        .sort(),
    )
  return propre(a) === propre(b)
}
