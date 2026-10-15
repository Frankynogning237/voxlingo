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
  outils: Record<string, string | null>
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API_URL}/health`)
  if (!res.ok) throw new Error(`Erreur serveur (${res.status})`)
  return res.json()
}
