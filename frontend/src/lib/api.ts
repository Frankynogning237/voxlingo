// Accès à l'API FastAPI (backend/main.py) : sur ce PC, ou moteur distant (Kaggle/Colab) via un tunnel.

const CLE_MOTEUR = 'voxlingo.moteur'
const CLE_ACCES = 'voxlingo.cle'

function lire(cle: string): string | null {
  try {
    return localStorage.getItem(cle)
  } catch {
    return null
  }
}

/** Lien envoyé par le carnet Kaggle/Colab : ?moteur=…&cle=… est enregistré puis retiré de l'adresse. */
function lireLienConnexion() {
  const params = new URLSearchParams(window.location.search)
  const moteur = params.get('moteur')
  if (!moteur) return
  enregistrerConnexion(moteur, params.get('cle') ?? '')
  window.history.replaceState(null, '', window.location.pathname + window.location.hash)
}

/** Enregistre l'adresse du moteur et sa clé d'accès (rechargez la page pour les appliquer). */
export function enregistrerConnexion(moteur: string, cle: string) {
  try {
    localStorage.setItem(CLE_MOTEUR, moteur.trim().replace(/\/+$/, ''))
    localStorage.setItem(CLE_ACCES, cle.trim())
  } catch {
    // stockage indisponible (navigation privée) : la connexion ne sera pas mémorisée
  }
}

lireLienConnexion()

export const API_URL_DEFAUT = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
export const API_URL = lire(CLE_MOTEUR) || API_URL_DEFAUT
export const CLE = lire(CLE_ACCES) ?? ''

/** En-têtes ajoutés à chaque appel : la clé d'accès du moteur distant. */
const entetes = (): Record<string, string> => (CLE ? { 'X-Cle-Acces': CLE } : {})

/** Adresse d'un fichier chargé directement par le navigateur (vidéo, image, téléchargement) :
 *  la clé passe dans l'adresse, car ces requêtes ne peuvent pas porter d'en-tête. */
const fichier = (path: string) => `${API_URL}${path}${CLE ? `?cle=${encodeURIComponent(CLE)}` : ''}`

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
export type StepState = 'attente' | 'en_cours' | 'termine' | 'erreur' | 'a_venir' | 'ignore' | 'kaggle'

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

export type StrategieDuree = 'reformuler' | 'accelerer'

export type JobOptions = {
  conserver_fond: boolean
  synchro_labiale: boolean
  sous_titres: boolean
  volume_fond: number
  ajuster_trop_longs: boolean
  /** Phrase trop longue : la reformuler (voix naturelle) ou accélérer la voix. */
  strategie_duree?: StrategieDuree
  /** Attribuer automatiquement une voix de la bibliothèque à chaque locuteur. */
  voix_bibliotheque?: boolean
  /** Glossaire (« terme = traduction », une ligne par terme) et consignes de style de la traduction */
  glossaire?: string
  consignes?: string
  /** Raccourcir la vidéo : passage gardé (en secondes) */
  debut_s?: number
  fin_s?: number
  decoupe?: { debut_s: number; fin_s: number }
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
  fichier_sortie: string | null
  segments_trop_longs?: number
  voix_generees?: number
  nb_locuteurs?: number
  synchro_importee?: boolean
  debut_traitement: string | null
  fin_traitement: string | null
  cree_le: string
  maj_le: string
  etapes: Step[]
}

export type Ton = 'energique' | 'calme' | 'intense' | 'doux' | 'neutre'

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
  duree_doublage?: number
  facteur_voix?: number
  locuteur?: string
  /** Ton de la phrase d'origine (phase 7), corrigeable */
  ton?: Ton
  ton_corrige?: boolean
  /** Voix doublée de cette phrase déjà générée */
  voix_disponible?: boolean
  /** Transcription corrigée à la main */
  texte_corrige?: boolean
  /** Texte d'origine corrigé depuis la dernière traduction : à retraduire */
  a_retraduire?: boolean
}

export type Transcription = {
  langue: string
  probabilite: number
  duree_s: number
  segments: Segment[]
  langue_cible?: string
  moteur_traduction?: string
  locuteurs?: Record<string, { nom: string; couleur: string; voix?: string }>
}

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { headers: entetes() })
  if (res.status === 401) throw new Error("Clé d'accès au moteur manquante ou incorrecte.")
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
export const videoUrl = (id: string) => fichier(`/jobs/${id}/video`)
export const thumbnailUrl = (id: string) => fichier(`/jobs/${id}/thumbnail`)
export const dubbedUrl = (id: string) => fichier(`/jobs/${id}/dubbed`)
export const downloadUrl = (id: string, quoi: 'download' | 'subtitles' | 'audio') => fichier(`/jobs/${id}/${quoi}`)
export const subtitlesVttUrl = (id: string) => fichier(`/jobs/${id}/subtitles.vtt`)
export const filmstripUrl = (id: string) => fichier(`/jobs/${id}/filmstrip`)
export const getWaveform = (id: string) => request<number[]>(`/jobs/${id}/waveform`)
/** Voix doublée d'une seule phrase, et nouvelle prise de cette phrase (puis réassemblage de la vidéo). */
export const segmentVoiceUrl = (id: string, segId: number) => fichier(`/jobs/${id}/segments/${segId}/voix`)
export const redoSegmentVoice = (id: string, segId: number) => send<Job>(`/jobs/${id}/segments/${segId}/refaire`, 'POST')

export async function cancelJob(id: string): Promise<Job> {
  const res = await fetch(`${API_URL}/jobs/${id}/annuler`, { method: 'POST', headers: entetes() })
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `Erreur serveur (${res.status})`)
  return res.json()
}

async function send<T>(path: string, method: string, body?: unknown, form?: FormData): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: body ? { ...entetes(), 'Content-Type': 'application/json' } : entetes(),
    body: form ?? (body ? JSON.stringify(body) : undefined),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => null)
    throw new Error(typeof data?.detail === 'string' ? data.detail : `Erreur serveur (${res.status})`)
  }
  return res.json()
}

/** Traduction modifiée ; `texte` corrige aussi la transcription d'origine. */
export type TraductionModifiee = { id: number; traduction: string; texte?: string }

export const saveTranslations = (id: string, segments: TraductionModifiee[]) =>
  send<Transcription>(`/jobs/${id}/segments`, 'PUT', { segments })
export const validateTranslations = (id: string, segments: TraductionModifiee[]) =>
  send<Job>(`/jobs/${id}/valider`, 'POST', { segments })
export const resetTranslations = (id: string) => send<Transcription>(`/jobs/${id}/reinitialiser`, 'POST')
/** Relance la traduction par l'IA : toutes les phrases, ou seulement `ids` (phrases corrigées). */
export const retranslate = (id: string, ids?: number[]) =>
  send<Job>(`/jobs/${id}/traduire`, 'POST', { ids: ids ?? null })
export type ValeursParametres = {
  force_device: 'auto' | 'cpu' | 'cuda'
  whisper_model: string
  ollama_model: string
  strategie_duree: StrategieDuree
  sous_titres: boolean
  conserver_fond: boolean
  volume_fond: number
}

export type Parametres = {
  valeurs: ValeursParametres
  defauts: ValeursParametres
  choix: { transcription: { valeur: string; libelle: string }[]; traduction: { valeur: string; libelle: string }[] }
  materiel: {
    device: 'cuda' | 'cpu'
    gpu: string | null
    processeur: string
    vram_totale_go: number
    vram_utilisee_go: number
    ram_totale_go: number
    ram_utilisee_go: number
    ram_libre_go: number
  }
  stockage: {
    dossier: string
    projets_octets: number
    modeles_octets: number
    disque_total_octets: number
    disque_libre_octets: number
  }
}

export const getSettings = () => request<Parametres>('/parametres')
export const saveSettings = (valeurs: ValeursParametres) => send<Parametres>('/parametres', 'PUT', valeurs)
export const openProjectsFolder = () => send<{ ok: boolean }>('/parametres/ouvrir-dossier', 'POST')

export const deleteJob = (id: string) => send<{ supprime: string }>(`/jobs/${id}`, 'DELETE')
export const duplicateJob = (id: string) => send<Job>(`/jobs/${id}/dupliquer`, 'POST')

export const setStrategy = (id: string, strategie_duree: StrategieDuree) =>
  send<Job>(`/jobs/${id}/options`, 'PUT', { strategie_duree })
/** Enregistre le glossaire et les consignes de traduction du projet. */
export const setTranslationRules = (id: string, regles: { glossaire: string; consignes: string }) =>
  send<Job>(`/jobs/${id}/options`, 'PUT', regles)

/** Ajuste les phrases trop longues (toutes, ou seulement `ids`). */
export const adjustTranslations = (id: string, ids?: number[]) =>
  send<Job>(`/jobs/${id}/ajuster`, 'POST', { ids: ids ?? null })

/** Taille des morceaux d'envoi : les tunnels gratuits (Cloudflare) refusent les requêtes de plus de 100 Mo. */
const MORCEAU = 50 * 1024 ** 2

/**
 * Envoie une requête avec XMLHttpRequest (plutôt que fetch) pour suivre la progression de l'envoi.
 */
function envoyer<T>(methode: string, path: string, corps: XMLHttpRequestBodyInit, onProgress: (e: ProgressEvent) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open(methode, `${API_URL}${path}`)
    for (const [nom, valeur] of Object.entries(entetes())) xhr.setRequestHeader(nom, valeur)
    xhr.responseType = 'json'
    xhr.upload.onprogress = onProgress
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as T)
      else reject(new Error(xhr.response?.detail ?? `Erreur serveur (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Le moteur ne répond pas.'))
    xhr.send(corps)
  })
}

/**
 * Envoie la vidéo au serveur et crée le projet (onProgress reçoit une valeur entre 0 et 1).
 * Une grosse vidéo part en morceaux de 50 Mo, chacun retenté en cas de coupure.
 */
export async function createJob(
  video: File,
  langueCible: string,
  options: JobOptions,
  onProgress: (fraction: number) => void,
): Promise<Job> {
  const form = new FormData()
  form.append('langue_cible', langueCible)
  for (const [cle, valeur] of Object.entries(options)) {
    if (valeur !== undefined && valeur !== null && typeof valeur !== 'object') form.append(cle, String(valeur))
  }

  if (video.size <= MORCEAU) {
    form.append('video', video)
    return envoyer<Job>('POST', '/jobs', form, (e) => e.lengthComputable && onProgress(e.loaded / e.total))
  }

  const { id } = await send<{ id: string }>('/televersements', 'POST')
  for (let debut = 0; debut < video.size; debut += MORCEAU) {
    const morceau = video.slice(debut, debut + MORCEAU)
    for (let essai = 1; ; essai++) {
      try {
        await envoyer('PUT', `/televersements/${id}?debut=${debut}`, morceau, (e) => onProgress((debut + e.loaded) / video.size))
        break
      } catch (erreur) {
        if (essai >= 4) throw erreur
        await new Promise((r) => setTimeout(r, 2000 * essai))
      }
    }
  }
  form.append('televersement', id)
  form.append('nom_fichier', video.name)
  return send<Job>('/jobs', 'POST', undefined, form)
}

// ───────────────────────── Locuteurs et bibliothèque de voix (phase 6) ─────────────────────────

export type Sexe = 'femme' | 'homme'
export type AgeVoix = 'enfant' | 'jeune' | 'adulte' | 'senior'

/** Réglages de timbre d'un locuteur (absents = neutres). */
export type Timbre = {
  ton?: number
  formant?: number
  graves?: number
  aigus?: number
  espace?: number
  telephone?: number
  vibrato?: number
}

export type Locuteur = {
  code: string
  nom: string
  couleur: string
  /** « clone » (voix d'origine clonée) ou identifiant d'une voix de la bibliothèque */
  voix: string
  /** Voix utilisée lors de la dernière génération (absent si jamais générée) */
  voix_generee?: string
  /** Réglages de timbre enregistrés, et ceux de la dernière génération */
  timbre?: Timbre
  timbre_genere?: Timbre
  f0_hz: number | null
  sexe_estime: Sexe | null
  registre: string | null
  nb_phrases: number
  duree_s: number
  part: number
}

export type Voix = {
  id: string
  nom: string
  source: 'xtts' | 'perso'
  sexe: Sexe
  age: AgeVoix
  registre: string | null
  f0_hz: number | null
  apercu: boolean
  /** Variante de style (Grave, Claire, Radio…) */
  style?: string
}

export const getSpeakers = (id: string) => request<Locuteur[]>(`/jobs/${id}/locuteurs`)
/** Écoute immédiate d'un réglage de timbre sur la voix choisie du locuteur. */
export function speakerTimbreUrl(id: string, code: string, timbre: Timbre): string {
  const base = fichier(`/jobs/${id}/locuteurs/${code}/timbre`)
  const params = new URLSearchParams(Object.entries(timbre).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)]))
  return params.size ? `${base}${base.includes('?') ? '&' : '?'}${params}` : base
}
export const updateSpeaker = (id: string, code: string, modif: { nom?: string; voix?: string; timbre?: Timbre }) =>
  send<Locuteur[]>(`/jobs/${id}/locuteurs/${code}`, 'PUT', modif)
export const mergeSpeakers = (id: string, source: string, cible: string) =>
  send<Locuteur[]>(`/jobs/${id}/locuteurs/fusion`, 'POST', { source, cible })
export const setSegmentSpeaker = (id: string, segId: number, locuteur: string) =>
  send<Transcription>(`/jobs/${id}/segments/${segId}/locuteur`, 'PUT', { locuteur })
export const setSegmentTone = (id: string, segId: number, ton: Ton) =>
  send<Transcription>(`/jobs/${id}/segments/${segId}/ton`, 'PUT', { ton })
export const detectSpeakers = (id: string) => send<Job>(`/jobs/${id}/locuteurs/detecter`, 'POST')
export const autoAssignVoices = (id: string) => send<Locuteur[]>(`/jobs/${id}/locuteurs/attribuer`, 'POST')
export const speakerAudioUrl = (id: string, code: string, quoi: 'original' | 'doublee') => fichier(`/jobs/${id}/locuteurs/${code}/${quoi}`)
export const speakerAvatarUrl = (id: string, code: string) => fichier(`/jobs/${id}/locuteurs/${code}/avatar`)

export const listVoices = () => request<{ voix: Voix[]; apercus: { en_cours: boolean; erreur: string | null } }>('/voix')
export const generatePreviews = () => send<{ en_cours: boolean }>('/voix/apercus', 'POST')
export const voicePreviewUrl = (voixId: string) => fichier(`/voix/${voixId}/apercu`)
export const updateVoice = (voixId: string, modif: { nom?: string; sexe?: Sexe; age?: AgeVoix }) =>
  send<Voix>(`/voix/${voixId}`, 'PUT', modif)
export const deleteVoice = (voixId: string) => send<{ supprime: string }>(`/voix/${voixId}`, 'DELETE')
export async function addVoice(fichier: File, nom: string, sexe: Sexe, age: AgeVoix): Promise<Voix> {
  const form = new FormData()
  form.append('echantillon', fichier)
  form.append('nom', nom)
  form.append('sexe', sexe)
  form.append('age', age)
  const res = await fetch(`${API_URL}/voix`, { method: 'POST', body: form, headers: entetes() })
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `Erreur serveur (${res.status})`)
  return res.json()
}

/** Régénère les voix du projet : seules les phrases dont le texte ou la voix a changé sont refaites. */
export async function regenerateVoices(id: string): Promise<Job> {
  const data = await getSegments(id)
  return validateTranslations(id, data.segments.map((s) => ({ id: s.id, traduction: s.traduction ?? '' })))
}

// ───────────────────────── Synchronisation labiale (phase 8) ─────────────────────────

export const kaggleUrl = (id: string) => fichier(`/jobs/${id}/kaggle`)
export async function importLipsync(id: string, video: File): Promise<Job> {
  const form = new FormData()
  form.append('video', video)
  const res = await fetch(`${API_URL}/jobs/${id}/synchro`, { method: 'POST', body: form, headers: entetes() })
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `Erreur serveur (${res.status})`)
  return res.json()
}
export const removeLipsync = (id: string) => send<Job>(`/jobs/${id}/synchro`, 'DELETE')

export const LIBELLES_AGE: Record<AgeVoix, string> = { enfant: 'Enfant', jeune: 'Jeune', adulte: 'Adulte', senior: 'Senior' }
