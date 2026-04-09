import { NavLink } from 'react-router-dom'

const links = [
  {
    to: '/',
    label: 'Overview',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M3 13.5l5-5 4 4 5-5.5M21 12v6a1 1 0 01-1 1H4a1 1 0 01-1-1V6a1 1 0 011-1h6" />
      </svg>
    ),
  },
  {
    to: '/upload',
    label: 'Upload Data',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M16 10l-4-4m0 0L8 10m4-4v12" />
      </svg>
    ),
  },
  {
    to: '/preprocess',
    label: 'Preprocess',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M12 3c-1.5 0-3 .5-4 1.5L5 8H3a1 1 0 00-1 1v6a1 1 0 001 1h2l3 3.5c1 1 2.5 1.5 4 1.5s3-.5 4-1.5L19 16h2a1 1 0 001-1V9a1 1 0 00-1-1h-2l-3-3.5C15 3.5 13.5 3 12 3z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6M12 9v6" />
      </svg>
    ),
  },
  {
    to: '/train',
    label: 'Train Model',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15M14.25 3.104c.251.023.501.05.75.082M19.8 15a2.25 2.25 0 01-.659 1.591l-1.591 1.591a2.25 2.25 0 01-3.182 0l-1.591-1.591M19.8 15l-3.023 3.023M5 14.5a2.25 2.25 0 00-.659 1.591l-1.591 1.591a2.25 2.25 0 003.182 3.182L7.523 19.8M5 14.5l3.023 3.023" />
      </svg>
    ),
  },
  {
    to: '/predict',
    label: 'Predict',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M3 13.5l5-5 4 4 5-5.5M21 12v6a1 1 0 01-1 1H4a1 1 0 01-1-1V6a1 1 0 011-1h6" />
      </svg>
    ),
  },
  {
    to: '/whatif',
    label: 'What-If',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
      </svg>
    ),
  },
  {
    to: '/history',
    label: 'History',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  {
    to: '/comparison',
    label: 'Comparison',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M3 3v18h18" />
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M7 16V10M11 16V7M15 16V4M19 16V11" />
      </svg>
    ),
  },
]

export default function Sidebar() {
  return (
    <aside className="fixed top-0 left-0 h-screen w-56 bg-gray-900 border-r border-gray-800
                      flex flex-col z-50 select-none">

      {/* Brand */}
      <div className="px-5 py-5 border-b border-gray-800">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth={2.2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M4.5 12.75l6 6 9-13.5" />
            </svg>
          </div>
          <div>
            <p className="text-white font-bold text-sm leading-none">Multi X-Y</p>
            <p className="text-blue-400 text-xs mt-0.5 font-medium tracking-wide">ML Dashboard</p>
          </div>
        </div>
      </div>

      {/* Navigation links */}
      <nav className="flex-1 px-3 py-4 overflow-y-auto space-y-1">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
               transition-all duration-150 group ${
                isActive
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                  : 'text-gray-200 hover:text-white hover:bg-gray-800'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span className={`flex-shrink-0 transition-colors ${
                  isActive ? 'text-white' : 'text-gray-500 group-hover:text-gray-100'
                }`}>
                  {l.icon}
                </span>
                <span className="truncate">{l.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-gray-800">
        <p className="text-gray-600 text-xs">v1.0.0 · FastAPI + React</p>
      </div>
    </aside>
  )
}
