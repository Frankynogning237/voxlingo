// Navigation minimale par l'URL (#/...), suffisante pour une application locale.

import { useSyncExternalStore } from 'react'

export type Route =
  | { page: 'accueil' }
  | { page: 'nouveau' }
  | { page: 'projets' }
  | { page: 'parametres' }
  | { page: 'projet'; id: string }
  | { page: 'edition'; id: string }

function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean)
  switch (parts[0]) {
    case 'nouveau':
      return { page: 'nouveau' }
    case 'projets':
      if (parts[1] && parts[2] === 'edition') return { page: 'edition', id: parts[1] }
      return parts[1] ? { page: 'projet', id: parts[1] } : { page: 'projets' }
    case 'parametres':
      return { page: 'parametres' }
    default:
      return { page: 'accueil' }
  }
}

function subscribe(callback: () => void) {
  window.addEventListener('hashchange', callback)
  return () => window.removeEventListener('hashchange', callback)
}

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash)
  return parse(hash)
}

export function href(route: Route): string {
  switch (route.page) {
    case 'accueil':
      return '#/'
    case 'projet':
      return `#/projets/${route.id}`
    case 'edition':
      return `#/projets/${route.id}/edition`
    default:
      return `#/${route.page}`
  }
}

export function navigate(route: Route) {
  window.location.hash = href(route)
}
