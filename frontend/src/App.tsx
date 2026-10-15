import { Layout } from '@/components/Layout'
import { Home } from '@/pages/Home'
import { NewDubbing } from '@/pages/NewDubbing'
import { Project } from '@/pages/Project'
import { useRoute } from '@/lib/router'

export default function App() {
  const route = useRoute()

  return (
    <Layout current={route}>
      {route.page === 'nouveau' ? (
        <NewDubbing />
      ) : route.page === 'projet' ? (
        <Project key={route.id} id={route.id} />
      ) : (
        <Home />
      )}
    </Layout>
  )
}
