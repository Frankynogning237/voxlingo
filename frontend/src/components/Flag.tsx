// Drapeau associé à une langue (paquet flag-icons, images SVG locales).
import { nomLangue } from '@/lib/format'
import { cn } from '@/lib/utils'

const PAYS: Record<string, string> = {
  fr: 'fr',
  en: 'gb',
  es: 'es',
  de: 'de',
  it: 'it',
  pt: 'pt',
  pl: 'pl',
  tr: 'tr',
  ru: 'ru',
  nl: 'nl',
  cs: 'cz',
  ar: 'sa',
  'zh-cn': 'cn',
  zh: 'cn',
  ja: 'jp',
  hu: 'hu',
  ko: 'kr',
  hi: 'in',
}

export function Flag({ langue, className }: { langue: string | null; className?: string }) {
  const pays = langue ? PAYS[langue] : undefined
  if (!pays) {
    return (
      <span className={cn('inline-flex h-[0.95em] w-[1.33em] items-center justify-center rounded-[3px] bg-muted text-[8px] font-bold uppercase', className)}>
        {langue?.slice(0, 2) ?? '?'}
      </span>
    )
  }
  return <span className={cn('fi rounded-[3px]', `fi-${pays}`, className)} title={nomLangue(langue)} />
}

/** Drapeau + nom de la langue, tel qu'affiché dans les cartes des maquettes. */
export function LangueTag({ langue, className }: { langue: string | null; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-2', className)}>
      <Flag langue={langue} />
      {nomLangue(langue)}
    </span>
  )
}
