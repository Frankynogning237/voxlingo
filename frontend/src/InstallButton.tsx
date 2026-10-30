import { useEffect, useState } from 'react'

interface InstallPrompt extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

// Écouté dès le chargement du module : le navigateur peut émettre l'événement avant l'affichage de l'interface.
let captured: InstallPrompt | null = null
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault()
  captured = e as InstallPrompt
  window.dispatchEvent(new Event('voxlingo:installable'))
})

/** Bouton « Installer » : n'apparaît que si le navigateur propose l'installation (Chrome, Edge…). */
export default function InstallButton() {
  const [prompt, setPrompt] = useState<InstallPrompt | null>(captured)

  useEffect(() => {
    const ready = () => setPrompt(captured)
    const installed = () => {
      captured = null
      setPrompt(null)
    }
    window.addEventListener('voxlingo:installable', ready)
    window.addEventListener('appinstalled', installed)
    return () => {
      window.removeEventListener('voxlingo:installable', ready)
      window.removeEventListener('appinstalled', installed)
    }
  }, [])

  if (!prompt) return null

  const install = async () => {
    await prompt.prompt()
    const { outcome } = await prompt.userChoice
    if (outcome === 'accepted') {
      captured = null
      setPrompt(null)
    }
  }

  return (
    <button
      type="button"
      onClick={install}
      title="Installer VoxLingo comme une application, à épingler à la barre des tâches"
      style={{
        position: 'fixed', right: 16, bottom: 16, zIndex: 1000, padding: '10px 16px', borderRadius: 12, border: 'none',
        background: 'linear-gradient(120deg, #4f46e5, #db2777)', color: '#fff', font: '600 14px system-ui, sans-serif',
        cursor: 'pointer', boxShadow: '0 8px 24px rgba(0,0,0,.35)',
      }}
    >
      Installer l'application
    </button>
  )
}
