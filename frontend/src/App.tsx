import { useQuery } from '@tanstack/react-query'
import {
  AudioLines,
  Cpu,
  FolderOpen,
  Home,
  Plus,
  RefreshCw,
  Settings,
} from 'lucide-react'
import type { ReactNode } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { getHealth, type Health } from '@/lib/api'
import { cn } from '@/lib/utils'

const NAV = [
  { label: 'Accueil', icon: Home, active: true },
  { label: 'Nouveau doublage', icon: Plus },
  { label: 'Mes projets', icon: FolderOpen },
  { label: 'Paramètres', icon: Settings },
]

const MODE_LABEL: Record<Health['mode'], string> = {
  gpu_local: 'GPU local',
  gpu_limite: 'GPU limité',
  cpu: 'CPU uniquement',
}

function Sidebar() {
  return (
    <aside className="flex w-60 shrink-0 flex-col gap-1 border-r border-sidebar-border bg-sidebar p-4">
      <div className="mb-6 flex items-center gap-2 px-2">
        <div className="flex size-8 items-center justify-center rounded-lg bg-linear-to-br from-brand-from to-brand-to">
          <AudioLines className="size-4 text-white" />
        </div>
        <span className="text-lg font-semibold">VoxLingo</span>
      </div>
      {NAV.map(({ label, icon: Icon, active }) => (
        <button
          key={label}
          type="button"
          disabled={!active}
          className={cn(
            'flex items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors duration-200',
            active
              ? 'bg-accent text-foreground'
              : 'text-muted-foreground hover:bg-accent/60 disabled:cursor-not-allowed disabled:opacity-60',
          )}
        >
          <Icon className="size-4" />
          {label}
        </button>
      ))}
    </aside>
  )
}

function SystemStatus({ health, error }: { health?: Health; error: boolean }) {
  if (error) {
    return (
      <Badge variant="destructive">Serveur local injoignable</Badge>
    )
  }
  if (!health) return <Badge variant="secondary">Vérification…</Badge>
  const gpu = health.device === 'cuda'
  return (
    <Badge
      variant="outline"
      className={cn(
        'gap-2',
        gpu ? 'border-success/40 text-success' : 'border-warning/40 text-warning',
      )}
    >
      <span className={cn('size-2 rounded-full', gpu ? 'bg-success' : 'bg-warning')} />
      {gpu ? `GPU NVIDIA prêt · ${health.vram_totale_go} Go` : 'Mode CPU'}
    </Badge>
  )
}

function InfoRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border py-2 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium">{value ?? '—'}</span>
    </div>
  )
}

export default function App() {
  const { data, isError, refetch, isFetching } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    retry: 1,
  })

  return (
    <div className="flex min-h-svh">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-8">
        <header className="mb-8 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">Bienvenue sur VoxLingo</h1>
            <p className="mt-1 text-muted-foreground">
              Doublez vos vidéos dans une autre langue, avec la voix de chaque personne.
            </p>
          </div>
          <SystemStatus health={data} error={isError} />
        </header>

        <Card className="mb-6">
          <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
            <div className="flex size-14 items-center justify-center rounded-2xl bg-linear-to-br from-brand-from to-brand-to">
              <Plus className="size-6 text-white" />
            </div>
            <div>
              <p className="text-lg font-semibold">Nouveau doublage</p>
              <p className="text-sm text-muted-foreground">
                Le téléversement de vidéo arrive à la phase 2.
              </p>
            </div>
            <Button disabled>Choisir une vidéo</Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="size-4" /> État du système
                </CardTitle>
                <CardDescription>Matériel détecté et modèles choisis.</CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetch()}
                disabled={isFetching}
              >
                <RefreshCw className={cn('size-4', isFetching && 'animate-spin')} />
                Actualiser
              </Button>
            </div>
          </CardHeader>
          <CardContent className="text-sm">
            {isError && (
              <p className="text-destructive">
                Le serveur local ne répond pas. Lancez <code>.\start.ps1</code> à la racine du projet.
              </p>
            )}
            {data && (
              <>
                <InfoRow
                  label="Matériel"
                  value={data.device === 'cuda' ? `GPU : ${data.vram_totale_go} Go de VRAM` : 'CPU'}
                />
                <InfoRow label="Mode" value={MODE_LABEL[data.mode]} />
                <InfoRow label="Transcription" value={data.modeles.transcription} />
                <InfoRow label="Traduction" value={data.modeles.traduction} />
                <InfoRow
                  label="Synchro labiale"
                  value={data.modeles.synchro_labiale_locale ? 'En local' : 'Via Kaggle'}
                />
                {Object.entries(data.outils).map(([nom, version]) => (
                  <InfoRow
                    key={nom}
                    label={nom}
                    value={version ?? <span className="text-warning">non trouvé</span>}
                  />
                ))}
              </>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
