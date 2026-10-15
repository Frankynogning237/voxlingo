import { Layout } from '@/components/Layout'
import { Home } from '@/pages/Home'
import { NewDubbing } from '@/pages/NewDubbing'
import { Project } from '@/pages/Project'
import { Projects } from '@/pages/Projects'
import { Settings } from '@/pages/Settings'
import { Speakers } from '@/pages/Speakers'
import { VoiceLibrary } from '@/pages/VoiceLibrary'
import { useRoute } from '@/lib/router'

export default function App() {
  const route = useRoute()

  return (
    <Layout current={route}>
      {route.page === 'nouveau' ? (
        <NewDubbing />
      ) : route.page === 'projets' ? (
        <Projects />
      ) : route.page === 'parametres' ? (
        <Settings />
      ) : route.page === 'voix' ? (
        <VoiceLibrary />
      ) : route.page === 'locuteurs' ? (
        <Speakers key={route.id} id={route.id} />
      ) : route.page === 'projet' || route.page === 'edition' ? (
        <Project key={route.id} id={route.id} edition={route.page === 'edition'} />
      ) : (
        <Home />
      )}
    </Layout>
  )
}
