/**
 * Overview.jsx
 * ------------
 * Executive-level plant performance dashboard.
 *
 * Features:
 *  • Auto-selects the best completed model run (highest R²)
 *  • KPI card per Y variable: R² gauge, MAE, performance badge, mini sparkline
 *  • Editable display names — inline per card or via bulk-rename panel
 *  • Display names persisted in localStorage; survive page reloads
 *  • New Y variables added to the model appear automatically with their raw name
 *  • Model switcher to compare different runs
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { listRuns, predictTestData } from '../services/api'
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, ResponsiveContainer,
  Tooltip, CartesianGrid, Legend,
} from 'recharts'

// ── localStorage helpers ───────────────────────────────────────────────────────
const LS_KEY = 'dae_y_display_names'
const loadNames  = () => { try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}') } catch { return {} } }
const saveNames  = (obj) => localStorage.setItem(LS_KEY, JSON.stringify(obj))

// ── Performance tier ───────────────────────────────────────────────────────────
function perf(r2) {
  if (r2 == null) return { label: 'No Data',       ring: '#4b5563', bar: '#4b5563', text: 'text-gray-500',   badge: 'bg-gray-800 text-gray-200 border-gray-600' }
  if (r2 >= 0.90) return { label: 'Excellent',     ring: '#22c55e', bar: '#22c55e', text: 'text-green-400',  badge: 'bg-green-900/50 text-green-300 border-green-700' }
  if (r2 >= 0.80) return { label: 'Good',          ring: '#60a5fa', bar: '#60a5fa', text: 'text-blue-400',   badge: 'bg-blue-900/50 text-blue-300 border-blue-700' }
  if (r2 >= 0.65) return { label: 'Fair',          ring: '#fbbf24', bar: '#fbbf24', text: 'text-yellow-400', badge: 'bg-yellow-900/40 text-yellow-300 border-yellow-700' }
  return             { label: 'Needs Attention', ring: '#f87171', bar: '#f87171', text: 'text-red-400',    badge: 'bg-red-900/40 text-red-300 border-red-700' }
}

// ── R² circular gauge (SVG) ───────────────────────────────────────────────────
function R2Gauge({ value }) {
  const r   = 28
  const circ = 2 * Math.PI * r
  const pct  = value == null ? 0 : Math.max(0, Math.min(1, value))
  const p    = perf(value)
  const dash = pct * circ
  return (
    <svg width="80" height="80" viewBox="0 0 80 80">
      <circle cx="40" cy="40" r={r} fill="none" stroke="#374151" strokeWidth="8" />
      <circle
        cx="40" cy="40" r={r} fill="none"
        stroke={p.ring} strokeWidth="8"
        strokeDasharray={`${dash} ${circ - dash}`}
        strokeLinecap="round"
        transform="rotate(-90 40 40)"
        style={{ transition: 'stroke-dasharray 0.6s ease' }}
      />
      <text x="40" y="40" textAnchor="middle" dominantBaseline="central"
        fill={p.ring} fontSize="13" fontWeight="700">
        {value == null ? 'N/A' : `${(value * 100).toFixed(1)}%`}
      </text>
    </svg>
  )
}

// ── Mini sparkline ─────────────────────────────────────────────────────────────
function Sparkline({ actual, predicted, color }) {
  const data = (actual || []).map((a, i) => ({ i, actual: a, predicted: predicted?.[i] }))
  return (
    <ResponsiveContainer width="100%" height={70}>
      <AreaChart data={data} margin={{ top: 4, right: 2, left: 2, bottom: 0 }}>
        <defs>
          <linearGradient id={`ga-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#60a5fa" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
          </linearGradient>
          <linearGradient id={`gp-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="i" hide />
        <YAxis hide domain={['auto', 'auto']} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', fontSize: 11, borderRadius: 6 }}
          formatter={(v, n) => [v?.toFixed(4), n]}
          labelFormatter={() => ''}
        />
        <Area type="monotone" dataKey="actual"    stroke="#60a5fa" strokeWidth={1.5}
          fill={`url(#ga-${color.replace('#', '')})`} dot={false} name="Actual" />
        <Area type="monotone" dataKey="predicted" stroke={color}   strokeWidth={1.5}
          fill={`url(#gp-${color.replace('#', '')})`} dot={false} name="Predicted" />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── Inline editable name ───────────────────────────────────────────────────────
function EditableName({ rawKey, displayNames, onChange }) {
  const [editing, setEditing] = useState(false)
  const [draft,   setDraft]   = useState(displayNames[rawKey] || rawKey)
  const inputRef = useRef(null)

  useEffect(() => { if (editing) inputRef.current?.focus() }, [editing])
  useEffect(() => { setDraft(displayNames[rawKey] || rawKey) }, [displayNames, rawKey])

  const commit = () => {
    const trimmed = draft.trim() || rawKey
    setDraft(trimmed)
    onChange(rawKey, trimmed)
    setEditing(false)
  }

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={e => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') { setDraft(displayNames[rawKey] || rawKey); setEditing(false) } }}
        className="bg-gray-700 border border-blue-500 rounded px-2 py-0.5 text-white
                   text-sm font-semibold w-full focus:outline-none"
      />
    )
  }
  return (
    <button
      onClick={() => setEditing(true)}
      className="group flex items-center gap-1.5 text-left w-full"
      title="Click to rename"
    >
      <span className="text-white font-semibold text-sm leading-snug">
        {displayNames[rawKey] || rawKey}
      </span>
      <svg className="w-3.5 h-3.5 text-gray-600 group-hover:text-blue-400 transition-colors flex-shrink-0"
        fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 013.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
      </svg>
    </button>
  )
}

// ── KPI Card ──────────────────────────────────────────────────────────────────
const CARD_COLORS = ['#22c55e', '#60a5fa', '#f59e0b', '#f87171', '#a78bfa', '#fb923c', '#38bdf8']

function KpiCard({ rawKey, colData, idx, displayNames, onRename }) {
  const p     = perf(colData?.r2)
  const color = CARD_COLORS[idx % CARD_COLORS.length]

  return (
    <div className={`bg-gray-800 rounded-2xl p-5 border ${p.badge.includes('green') ? 'border-green-900/60'
                    : p.badge.includes('blue') ? 'border-blue-900/60'
                    : p.badge.includes('yellow') ? 'border-yellow-900/60'
                    : p.badge.includes('red') ? 'border-red-900/60'
                    : 'border-gray-700'} flex flex-col gap-3 hover:shadow-lg transition-shadow`}>

      {/* Top row: badge + raw key */}
      <div className="flex items-start justify-between gap-2">
        <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold flex-shrink-0 ${p.badge}`}>
          {p.label}
        </span>
        <span className="text-gray-600 text-xs text-right truncate" title={rawKey}>{rawKey}</span>
      </div>

      {/* Editable display name */}
      <EditableName rawKey={rawKey} displayNames={displayNames} onChange={onRename} />

      {/* Gauge + metrics */}
      <div className="flex items-center gap-4">
        <R2Gauge value={colData?.r2} />
        <div className="flex-1 space-y-2">
          <div>
            <p className="text-gray-500 text-xs">R² Score</p>
            <p className={`text-lg font-bold ${p.text}`}>
              {colData?.r2 != null ? colData.r2.toFixed(4) : '—'}
            </p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">MAE</p>
            <p className="text-base font-semibold text-white">
              {colData?.mae != null ? colData.mae.toFixed(4) : '—'}
            </p>
          </div>
        </div>
      </div>

      {/* R² progress bar */}
      <div>
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Model Accuracy</span>
          <span>{colData?.r2 != null ? `${(colData.r2 * 100).toFixed(1)}%` : '—'}</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="h-2 rounded-full transition-all duration-700"
            style={{
              width: `${colData?.r2 != null ? Math.max(0, Math.min(100, colData.r2 * 100)) : 0}%`,
              background: `linear-gradient(90deg, #ef4444 0%, #fbbf24 50%, ${p.bar} 100%)`,
            }}
          />
        </div>
      </div>

      {/* Mini sparkline */}
      {colData?.actual?.length > 0 && (
        <div>
          <p className="text-gray-600 text-xs mb-1">
            <span className="text-blue-400">━</span> Actual &nbsp;
            <span style={{ color }}>━</span> Predicted
          </p>
          <Sparkline actual={colData.actual} predicted={colData.predicted} color={color} />
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Overview() {
  const [runs, setRuns]               = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const [result, setResult]           = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  // Display names (localStorage)
  const [displayNames, setDisplayNamesState] = useState(loadNames)

  // Bulk rename panel
  const [showRename, setShowRename]   = useState(false)
  const [draftNames, setDraftNames]   = useState({})

  // ── Load runs ──────────────────────────────────────────────────────────────
  useEffect(() => {
    listRuns()
      .then(r => {
        const done = r.data.filter(run => ['done', 'stopped'].includes(run.status))
        setRuns(done)
        if (done.length > 0) {
          // Auto-select best R² run
          const best = done.reduce((b, c) =>
            (c.metrics?.r2_score ?? 0) > (b.metrics?.r2_score ?? 0) ? c : b
          )
          selectRun(best)
        }
      })
      .catch(() => setError('Failed to load model runs.'))
  }, [])

  // ── Select a run & load predictions ───────────────────────────────────────
  const selectRun = useCallback(async (run) => {
    setSelectedRun(run)
    setResult(null)
    setError('')
    setLoading(true)
    try {
      const r = await predictTestData(run.id)
      setResult(r.data)
      setLastUpdated(new Date())
    } catch (e) {
      setError('Failed to load predictions: ' + (e.response?.data?.detail || e.message))
    }
    setLoading(false)
  }, [])

  // ── Persist display names ──────────────────────────────────────────────────
  const handleRename = useCallback((rawKey, newName) => {
    setDisplayNamesState(prev => {
      const next = { ...prev, [rawKey]: newName }
      saveNames(next)
      return next
    })
  }, [])

  // ── Open bulk rename panel ─────────────────────────────────────────────────
  const openRenamePanel = () => {
    const yCols = result ? Object.keys(result.columns) : []
    const draft = {}
    for (const k of yCols) draft[k] = displayNames[k] || k
    setDraftNames(draft)
    setShowRename(true)
  }

  const saveBulkNames = () => {
    const next = { ...displayNames, ...draftNames }
    saveNames(next)
    setDisplayNamesState(next)
    setShowRename(false)
  }

  const resetAllNames = () => {
    const yCols = result ? Object.keys(result.columns) : []
    const next  = { ...displayNames }
    for (const k of yCols) delete next[k]
    saveNames(next)
    setDisplayNamesState(next)
    setShowRename(false)
  }

  // ── Derived values ─────────────────────────────────────────────────────────
  const yCols       = result ? Object.keys(result.columns) : []
  const r2Values    = yCols.map(c => result.columns[c]?.r2).filter(v => v != null)
  const overallR2   = selectedRun?.metrics?.r2_score
  const bestR2      = r2Values.length ? Math.max(...r2Values) : null
  const worstR2     = r2Values.length ? Math.min(...r2Values) : null
  const aboveTarget = r2Values.filter(v => v >= 0.85).length

  const perfCounts  = yCols.reduce((acc, c) => {
    const p = perf(result?.columns[c]?.r2).label
    acc[p] = (acc[p] || 0) + 1
    return acc
  }, {})

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="p-8 max-w-screen-xl mx-auto">

      {/* ── PAGE HEADER ── */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M3 13.5l5-5 4 4 5-5.5M21 12v6a1 1 0 01-1 1H4a1 1 0 01-1-1V6a1 1 0 011-1h6" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-white">Plant Performance Overview</h1>
          </div>
          <p className="text-gray-200 text-sm">
            Executive dashboard — model accuracy per output variable.
            {lastUpdated && (
              <span className="ml-2 text-gray-600">
                Last refreshed {lastUpdated.toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {result && (
            <button
              onClick={openRenamePanel}
              className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700
                         text-gray-100 hover:text-white px-4 py-2 rounded-lg text-sm transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 013.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
              Rename Y Variables
            </button>
          )}
          {selectedRun && (
            <button
              onClick={() => selectRun(selectedRun)}
              disabled={loading}
              className="flex items-center gap-2 bg-blue-700 hover:bg-blue-600 disabled:opacity-50
                         text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </button>
          )}
        </div>
      </div>

      {/* ── MODEL SELECTOR ── */}
      <div className="bg-gray-800 rounded-xl p-5 mb-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-white font-semibold text-sm">Active Model Run</h2>
          <span className="text-gray-500 text-xs">
            ★ = auto-selected (best R²)
          </span>
        </div>
        {runs.length === 0 ? (
          <p className="text-gray-500 text-sm">No trained models. Go to Train Model first.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {runs.map(run => {
              const r2  = run.metrics?.r2_score
              const p   = perf(r2)
              const best = runs.reduce((b, c) =>
                (c.metrics?.r2_score ?? 0) > (b.metrics?.r2_score ?? 0) ? c : b
              )
              return (
                <button
                  key={run.id}
                  onClick={() => selectRun(run)}
                  disabled={loading}
                  className={`text-left px-4 py-2.5 rounded-lg text-sm border transition-colors
                               disabled:opacity-50 ${
                    selectedRun?.id === run.id
                      ? 'bg-blue-600 border-blue-500 text-white'
                      : 'bg-gray-700 border-gray-600 text-gray-100 hover:border-blue-400'
                  }`}
                >
                  <div className="font-semibold flex items-center gap-1.5">
                    {best.id === run.id && <span title="Best R²">★</span>}
                    Run #{run.id}
                    {run.status === 'stopped' && <span className="text-orange-300 text-xs">partial</span>}
                  </div>
                  <div className={`text-xs mt-0.5 ${selectedRun?.id === run.id ? 'text-blue-200' : p.text}`}>
                    R² {r2 != null ? r2.toFixed(4) : 'N/A'} · {run.epochs} ep ·
                    {run.x_columns?.length}X → {run.y_columns?.length}Y
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* ── LOSS CURVE ── shown as soon as a run is selected, no extra API call */}
      {selectedRun?.metrics?.train_loss_history?.length > 0 && (() => {
        const trainLoss = selectedRun.metrics.train_loss_history
        const valLoss   = selectedRun.metrics.val_loss_history || []
        const lossData  = trainLoss.map((v, i) => ({
          epoch: i + 1,
          'Train Loss': +v.toFixed(6),
          ...(valLoss[i] != null ? { 'Val Loss': +valLoss[i].toFixed(6) } : {}),
        }))
        return (
          <div className="bg-gray-800 rounded-xl p-5 mb-6">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-white font-semibold text-sm">Training Loss Curve</h2>
                <p className="text-gray-500 text-xs mt-0.5">
                  Run #{selectedRun.id} · {trainLoss.length} epochs · Final train loss {trainLoss[trainLoss.length - 1]?.toFixed(6)}
                  {valLoss.length > 0 && ` · val ${valLoss[valLoss.length - 1]?.toFixed(6)}`}
                </p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={lossData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="epoch" tick={{ fill: '#6b7280', fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottomRight', offset: -8, fill: '#6b7280', fontSize: 11 }} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} domain={['auto', 'auto']} width={70} tickFormatter={v => v.toFixed(4)} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', fontSize: 12, borderRadius: 8 }}
                  formatter={(v, n) => [v.toFixed(6), n]}
                  labelFormatter={l => `Epoch ${l}`}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                <Line type="monotone" dataKey="Train Loss" stroke="#60a5fa" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                {valLoss.length > 0 && (
                  <Line type="monotone" dataKey="Val Loss" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )
      })()}

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl px-4 py-3 mb-6">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {loading && (
        <div className="bg-gray-800 rounded-xl p-12 text-center mb-6">
          <div className="text-blue-400 text-lg animate-pulse mb-2">Loading performance data…</div>
          <p className="text-gray-500 text-sm">Running predictions on test set</p>
        </div>
      )}

      {result && !loading && (<>

        {/* ── SUMMARY BANNER ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
          {[
            { label: 'Overall R²',   value: overallR2 != null ? `${(overallR2*100).toFixed(2)}%` : '—', sub: 'Model-wide', color: perf(overallR2).text },
            { label: 'Best Output',  value: bestR2  != null ? `${(bestR2*100).toFixed(2)}%`  : '—', sub: 'Highest R²',  color: perf(bestR2).text  },
            { label: 'Weakest Output', value: worstR2 != null ? `${(worstR2*100).toFixed(2)}%` : '—', sub: 'Lowest R²', color: perf(worstR2).text },
            { label: 'Above 85%',    value: `${aboveTarget} / ${yCols.length}`, sub: 'Y variables', color: aboveTarget === yCols.length ? 'text-green-400' : 'text-yellow-400' },
            { label: 'Test Rows',    value: result.n_test_rows?.toLocaleString() ?? '—', sub: 'Held-out rows', color: 'text-white' },
            { label: 'Y Variables',  value: yCols.length, sub: 'Tracked outputs', color: 'text-blue-400' },
          ].map(s => (
            <div key={s.label} className="bg-gray-800 rounded-xl px-4 py-3 border border-gray-700/60">
              <p className="text-gray-500 text-xs mb-0.5">{s.label}</p>
              <p className={`text-xl font-bold ${s.color}`}>{s.value}</p>
              <p className="text-gray-600 text-xs mt-0.5">{s.sub}</p>
            </div>
          ))}
        </div>

        {/* ── PERFORMANCE BREAKDOWN BAR ── */}
        <div className="bg-gray-800 rounded-xl px-5 py-4 mb-6 flex items-center gap-4">
          <span className="text-gray-200 text-sm font-medium flex-shrink-0">Performance Split</span>
          <div className="flex-1 flex rounded-full overflow-hidden h-4 bg-gray-700">
            {[
              { key: 'Excellent',     color: '#22c55e' },
              { key: 'Good',          color: '#60a5fa' },
              { key: 'Fair',          color: '#fbbf24' },
              { key: 'Needs Attention', color: '#f87171' },
              { key: 'No Data',       color: '#4b5563' },
            ].map(({ key, color }) => {
              const n = perfCounts[key] || 0
              if (!n) return null
              return (
                <div
                  key={key}
                  title={`${key}: ${n}`}
                  className="h-4 transition-all duration-700"
                  style={{ width: `${(n / yCols.length) * 100}%`, backgroundColor: color }}
                />
              )
            })}
          </div>
          <div className="flex gap-4 text-xs text-gray-200 flex-shrink-0">
            {[
              { label: 'Excellent', color: '#22c55e' },
              { label: 'Good',      color: '#60a5fa' },
              { label: 'Fair',      color: '#fbbf24' },
              { label: 'Needs Attention', color: '#f87171' },
            ].map(l => (
              <span key={l.label} className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: l.color }} />
                {l.label} ({perfCounts[l.label] || 0})
              </span>
            ))}
          </div>
        </div>

        {/* ── KPI CARDS GRID ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {yCols.map((col, i) => (
            <KpiCard
              key={col}
              rawKey={col}
              colData={result.columns[col]}
              idx={i}
              displayNames={displayNames}
              onRename={handleRename}
            />
          ))}
        </div>

        <p className="text-gray-700 text-xs mt-6 text-center">
          Click any variable name to rename it · Names saved automatically to browser storage
        </p>
      </>)}

      {/* ── BULK RENAME PANEL (modal overlay) ── */}
      {showRename && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-6">
          <div className="bg-gray-900 rounded-2xl border border-gray-700 w-full max-w-lg shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
              <div>
                <h2 className="text-white font-bold text-lg">Rename Y Variables</h2>
                <p className="text-gray-500 text-xs mt-0.5">
                  Set display names shown on the dashboard. Raw variable names are unchanged.
                </p>
              </div>
              <button onClick={() => setShowRename(false)}
                className="text-gray-500 hover:text-white transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Fields */}
            <div className="px-6 py-4 space-y-3 max-h-96 overflow-y-auto">
              {Object.keys(draftNames).map(rawKey => (
                <div key={rawKey}>
                  <label className="block text-gray-500 text-xs mb-1 font-mono">{rawKey}</label>
                  <input
                    type="text"
                    value={draftNames[rawKey]}
                    onChange={e => setDraftNames(p => ({ ...p, [rawKey]: e.target.value }))}
                    placeholder={rawKey}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                               text-white text-sm focus:border-blue-500 focus:outline-none"
                  />
                </div>
              ))}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-gray-800">
              <button
                onClick={resetAllNames}
                className="text-gray-500 hover:text-red-400 text-sm transition-colors"
              >
                ↺ Reset to Raw Names
              </button>
              <div className="flex gap-3">
                <button onClick={() => setShowRename(false)}
                  className="text-gray-200 hover:text-white text-sm px-4 py-2 transition-colors">
                  Cancel
                </button>
                <button
                  onClick={saveBulkNames}
                  className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg
                             text-sm font-semibold transition-colors"
                >
                  Save Names
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
