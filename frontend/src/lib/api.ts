// Accès à l'API locale FastAPI (backend/main.py).

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export type Health = {
  status: string
  device: 'cuda' | 'cpu'
  mode: 'gpu_local' | 'gpu_limite' | 'cpu'
  vram_totale_go: number
  vram_libre_go: number
  modeles: {
    transcription: string
    traduction: string
    synchro_labiale_locale: boolean
  }
  langues_cibles: string[]
  nom_utilisateur: string
  outils: Record<string, string | null>
}

export type JobState = 'en_attente' | 'en_cours' | 'a_valider' | 'termine' | 'erreur' | 'annule'
export type StepState = 'attente' | 'en_cours' | 'termine' | 'erreur' | 'a_venir'

export type Step = {
  cle: string
  libelle: string
  etat: StepState
  duree_s: number | null
}

export type BilanAjustement = {
  silence: number
  raccourcies: number
  decoupees: number
  a_revoir: number
}

export type JobOptions = {
  conserver_fond: boolean
  synchro_labiale: boolean
  sous_titres: boolean
  volume_fond: number
  ajuster_trop_longs: boolean
}

export type Job = {
  id: string
  nom_fichier: string
  fichier: string
  langue_cible: string
  options: JobOptions
  langue_source: string | null
  probabilite_langue: number | null
  duree_s: number | null
  etat: JobState
  etape: string | null
  progression: number
  progression_etape: number
  erreur: string | null
  annulation_demandee: boolean
  moteur_traduction: string | null
  traduction_validee: boolean
  dernier_ajustement: BilanAjustement | null
  debut_traitement: string | null
  fin_traitement: string | null
  cree_le: string
  maj_le: string
  etapes: Step[]
}

export type Segment = {
  id: number
  debut: number
  fin: number
  texte: string
  traduction?: string
  traduction_ia?: string
  trop_long?: boolean
  fin_max?: number
  ajustement?: 'raccourcie' | 'decoupee' | 'a_revoir'
}

export type Transcription = {
  langue: string
  probabilite: number
  duree_s: number
  segments: Segment[]
  langue_cible?: string
  moteur_traduction?: string
}

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Erreur serveur (${res.status})`)
  }
  return res.json()
}

export const getHealth = () => request<Health>('/health')
export const listJobs = () => request<Job[]>('/jobs')
export const getJob = (id: string) => request<Job>(`/jobs/${id}`)
export const getSegments = (id: string) => request<Transcription>(`/jobs/${id}/segments`)
export const videoUrl = (id: string) => `${API_URL}/jobs/${id}/video`
export const thumbnailUrl = (id: string) => `${API_URL}/jobs/${id}/thumbnail`
export const filmstripUrl = (id: string) => `${API_URL}/jobs/${id}/filmstrip`
export const getWaveform = (id: string) => request<number[]>(`/jobs/${id}/waveform`)

export async function cancelJob(id: string): Promise<Job> {
  const res = await fetch(`${API_URL}/jobs/${id}/annuler`, { method: 'POST' })
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `Erreur serveur (${res.status})`)
  return res.json()
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const data = await res.json().catch(() => null)
    throw new Error(typeof data?.detail === 'string' ? data.detail : `Erreur serveur (${res.status})`)
  }
  return res.json()
}

export type TraductionModifiee = { id: number; traduction: string }

export const saveTranslations = (id: string, segments: TraductionModifiee[]) =>
  send<Transcription>(`/jobs/${id}/segments`, 'PUT', { segments })
export const validateTranslations = (id: string, segments: TraductionModifiee[]) =>
  send<Job>(`/jobs/${id}/valider`, 'POST', { segments })
export const resetTranslations = (id: string) => send<Transcription>(`/jobs/${id}/reinitialiser`, 'POST')
export const retranslate = (id: string) => send<Job>(`/jobs/${id}/traduire`, 'POST')
/** Ajuste les phrases trop longues (toutes, ou seulement `ids`). */
export const adjustTranslations = (id: string, ids?: number[]) =>
  send<Job>(`/jobs/${id}/ajuster`, 'POST', { ids: ids ?? null })

/**
 * Envoie la vidéo au serveur. XMLHttpRequest plutôt que fetch pour suivre
 * la progression de l'envoi (onProgress reçoit une valeur entre 0 et 1).
 */
export function createJob(
  video: File,
  langueCible: string,
  options: JobOptions,
  onProgress: (fraction: number) => void,
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('video', video)
    form.append('langue_cible', langueCible)
    for (const [cle, valeur] of Object.entries(options)) form.append(cle, String(valeur))

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_URL}/jobs`)
    xhr.responseType = 'json'
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as Job)
      else reject(new Error(xhr.response?.detail ?? `Erreur serveur (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Le serveur local ne répond pas.'))
    xhr.send(form)
  })
}
