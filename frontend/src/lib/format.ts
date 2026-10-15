// Libellés et mises en forme partagés par les pages.

/** Nom français des langues prises en charge (codes Whisper / XTTS). */
export const LANGUES: Record<string, string> = {
  fr: 'Français',
  en: 'Anglais',
  es: 'Espagnol',
  de: 'Allemand',
  it: 'Italien',
  pt: 'Portugais',
  pl: 'Polonais',
  tr: 'Turc',
  ru: 'Russe',
  nl: 'Néerlandais',
  cs: 'Tchèque',
  ar: 'Arabe',
  'zh-cn': 'Chinois',
  zh: 'Chinois',
  ja: 'Japonais',
  hu: 'Hongrois',
  ko: 'Coréen',
  hi: 'Hindi',
}

export const nomLangue = (code: string | null) =>
  code ? (LANGUES[code] ?? code.toUpperCase()) : 'Inconnue'

/** 75.4 → "1:15" ; 3725 → "1:02:05" */
export function formatDuree(secondes: number | null): string {
  if (secondes == null) return '—'
  const total = Math.round(secondes)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = String(total % 60).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}

export function formatTaille(octets: number): string {
  if (octets >= 1024 ** 3) return `${(octets / 1024 ** 3).toFixed(1)} Go`
  return `${(octets / 1024 ** 2).toFixed(1)} Mo`
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString('fr-FR', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** « À l'instant », « Il y a 5 min », « Il y a 2 h », « Hier », « 12 oct. » */
export function formatIlYa(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000)
  if (minutes < 1) return "À l'instant"
  if (minutes < 60) return `Il y a ${minutes} min`
  const heures = Math.round(minutes / 60)
  if (heures < 24) return `Il y a ${heures} h`
  if (heures < 48) return 'Hier'
  return new Date(iso).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' })
}

/** Nom du fichier sans extension, pour le titre des cartes. */
export const titreProjet = (nomFichier: string) => nomFichier.replace(/\.[^.]+$/, '')
