import { Routes, Route } from 'react-router-dom'
import Navbar      from './components/Navbar'
import Overview    from './pages/Overview'
import Upload      from './pages/Upload'
import Preprocess  from './pages/Preprocess'
import Train       from './pages/Train'
import Predict     from './pages/Predict'
import WhatIf      from './pages/WhatIf'
import History     from './pages/History'
import Comparison  from './pages/Comparison'

export default function App() {
  return (
    <div className="flex min-h-screen bg-gray-950">
      {/* Fixed left sidebar — 224px (w-56) wide */}
      <Navbar />

      {/* Main content — offset by sidebar width so nothing is hidden under it */}
      <main className="flex-1 ml-56 min-h-screen overflow-y-auto">
        <Routes>
          <Route path="/"           element={<Overview />} />
          <Route path="/overview"   element={<Overview />} />
          <Route path="/upload"     element={<Upload />} />
          <Route path="/preprocess" element={<Preprocess />} />
          <Route path="/train"      element={<Train />} />
          <Route path="/predict"    element={<Predict />} />
          <Route path="/whatif"     element={<WhatIf />} />
          <Route path="/history"    element={<History />} />
          <Route path="/comparison" element={<Comparison />} />
        </Routes>
      </main>
    </div>
  )
}
