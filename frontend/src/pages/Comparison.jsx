/**
 * Comparison.jsx
 * --------------
 * Model Comparison Dashboard
 * Shows all completed runs side-by-side, highlights the best model overall
 * and the best model per Y output.
 */

import { useState, useEffect } from 'react'
import { getModelComparison } from '../services/api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from 'recharts'

// ── Model metadata ────────────────────────────────────────────────────────────
const MODEL_META = {
  dae:     { label: 'DAE',     fullName: 'Denoising AutoEncoder', color: '#60a5fa', bg: 'bg-blue-900/40',   border: 'border-blue-600',   icon: '🧠' },
  lstm:    { label: 'LSTM',    fullName: 'Long Short-Term Memory', color: '#34d399', bg: 'bg-emerald-900/40', border: 'border-emerald-600', icon: '🔄' },
  xgboost: { label: 'XGBoost', fullName: 'XGBoost',              color: '#f59e0b', bg: 'bg-amber-900/40',   border: 'border-amber-600',   icon: '🌲' },
  gpr:     { label: 'GPR',     fullName: 'Gaussian Process',      color: '#a78bfa', bg: 'bg-violet-900/40',  border: 'border-violet-600',  icon: '📈' },
  ssm:     { label: 'SSM',     fullName: 'State Space Model',     color: '#fb923c', bg: 'bg-orange-900/40',  border: 'border-orange-600',  icon: '📡' },
}

function getMeta(type) {
  return MODEL_META[type] ?? MODEL_META['dae']
}

function fmt(v, dp = 4) {
  if (v == null || isNaN(v)) return '—'
  return Number(v).toFixed(dp)
}

function R2Badge({ value }) {
  if (value == null) return <span className="text-gray-600 text-xs">—</span>
  const pct = Math.round(value * 100)
  const cls = value >= 0.9 ? 'text-green-400' : value >= 0.7 ? 'text-yellow-400' : 'text-red-400'
  return <span className={`font-bold text-sm ${cls}`}>{pct}%</span>
}

// ── component ─────────────────────────────────────────────────────────────────
export default function Comparison() {
  const [data, setData]           = useState(null)
  const [loading, setLoading]     = useState(true)
  const [filterType, setFilterType] = useState('all')
  const [activeY, setActiveY]     = useState(null)   // Y column for per-Y chart
  const [sortBy, setSortBy]       = useState('r2')   // 'r2' | 'created'

  useEffect(() => {
    setLoading(true)
    getModelComparison()
      .then(r => {
        setData(r.data)
        // Auto-select first Y column
        const allY = [...new Set(r.data.runs.flatMap(run => run.y_columns || []))]
        if (allY.length > 0) setActiveY(allY[0])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-96">
        <span className="text-gray-400">Loading comparison data…</span>
      </div>
    )
  }

  if (!data || data.runs.length === 0) {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-2">📊 Model Comparison</h1>
        <div className="bg-gray-800 rounded-xl p-12 text-center">
          <span className="text-5xl mb-4 block">🏁</span>
          <p className="text-gray-300 font-semibold mb-2">No completed model runs yet</p>
          <p className="text-gray-500 text-sm">Train at least one model to see comparisons here.</p>
        </div>
      </div>
    )
  }

  const { runs, best_overall, best_per_y } = data

  // Filter + sort
  const filtered = runs
    .filter(r => filterType === 'all' || r.model_type === filterType)
    .sort((a, b) => {
      if (sortBy === 'r2') return (b.r2_score ?? -1) - (a.r2_score ?? -1)
      return new Date(b.created_at) - new Date(a.created_at)
    })

  // All Y columns across all runs
  const allYCols = [...new Set(runs.flatMap(r => r.y_columns || []))]

  // Chart data: overall R² per run
  const barData = filtered.map(r => ({
    name:  `#${r.id} ${getMeta(r.model_type).label}`,
    r2:    r.r2_score != null ? Math.max(0, r.r2_score) : 0,
    color: getMeta(r.model_type).color,
    id:    r.id,
  }))

  // Per-Y R² chart for selected Y
  const perYBar = activeY
    ? filtered
        .filter(r => r.per_y_r2?.[activeY] != null)
        .map(r => ({
          name:  `#${r.id} ${getMeta(r.model_type).label}`,
          r2:    Math.max(0, r.per_y_r2[activeY]),
          color: getMeta(r.model_type).color,
        }))
    : []

  const availableTypes = [...new Set(runs.map(r => r.model_type))]

  return (
    <div className="p-8 max-w-screen-xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-1">📊 Model Comparison</h1>
      <p className="text-gray-400 text-sm mb-6">
        Compare all trained models. Best model recommended for prediction and What-If analysis.
      </p>

      {/* ── Best Model Banner ── */}
      {best_overall && (
        <div className={`mb-6 rounded-xl p-5 border ${getMeta(best_overall.model_type).border} ${getMeta(best_overall.model_type).bg}`}>
          <div className="flex items-center gap-4">
            <span className="text-4xl">{getMeta(best_overall.model_type).icon}</span>
            <div>
              <p className="text-white font-bold text-lg">
                ★ Best Overall Model — Run #{best_overall.run_id}
              </p>
              <p className="text-gray-300 text-sm mt-0.5">
                <span className="font-semibold" style={{ color: getMeta(best_overall.model_type).color }}>
                  {getMeta(best_overall.model_type).fullName}
                </span>
                <span className="ml-3 text-white font-bold text-base">
                  R² {(best_overall.r2_score * 100).toFixed(1)}%
                </span>
                <span className="ml-2 text-gray-400">overall validation</span>
              </p>
            </div>
            <div className="ml-auto text-right">
              <p className="text-gray-400 text-xs">Recommended for</p>
              <div className="flex gap-2 mt-1">
                <span className="text-xs bg-blue-700 text-blue-200 px-2 py-1 rounded">Prediction</span>
                <span className="text-xs bg-purple-700 text-purple-200 px-2 py-1 rounded">What-If</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Controls ── */}
      <div className="flex items-center gap-4 mb-5 flex-wrap">
        {/* Model type filter */}
        <div className="flex bg-gray-800 rounded-lg p-1 gap-1">
          <button
            onClick={() => setFilterType('all')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              filterType === 'all' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            All ({runs.length})
          </button>
          {availableTypes.map(t => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                filterType === t ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              {getMeta(t).icon} {getMeta(t).label}
            </button>
          ))}
        </div>

        {/* Sort */}
        <div className="flex items-center gap-2 ml-auto text-xs text-gray-400">
          <span>Sort:</span>
          {[['r2', 'Best R²'], ['created', 'Newest']].map(([k, lbl]) => (
            <button
              key={k}
              onClick={() => setSortBy(k)}
              className={`px-2.5 py-1 rounded border transition-colors ${
                sortBy === k ? 'bg-gray-600 border-gray-500 text-white' : 'border-gray-700 hover:text-white'
              }`}
            >
              {lbl}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-5 gap-5">

        {/* ── LEFT: charts ── */}
        <div className="col-span-3 space-y-5">

          {/* Overall R² bar chart */}
          <div className="bg-gray-800 rounded-xl p-5">
            <h2 className="text-white font-semibold mb-4">Overall R² by Run</h2>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="name"
                  stroke="#6b7280"
                  tick={{ fontSize: 9 }}
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis
                  stroke="#6b7280"
                  tick={{ fontSize: 10 }}
                  domain={[0, 1]}
                  tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
                  formatter={v => [`${(v * 100).toFixed(2)}%`, 'R²']}
                />
                <Bar dataKey="r2" radius={[4, 4, 0, 0]}>
                  {barData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Per-Y R² chart */}
          {allYCols.length > 0 && (
            <div className="bg-gray-800 rounded-xl p-5">
              <div className="flex items-center gap-3 mb-4 flex-wrap">
                <h2 className="text-white font-semibold">Per-Y R² for</h2>
                <div className="flex flex-wrap gap-1.5">
                  {allYCols.map(yc => (
                    <button
                      key={yc}
                      onClick={() => setActiveY(yc)}
                      className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                        activeY === yc
                          ? 'bg-blue-600 border-blue-500 text-white'
                          : 'border-gray-600 text-gray-400 hover:text-white'
                      }`}
                    >
                      {yc}
                    </button>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={perYBar} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#6b7280" tick={{ fontSize: 9 }} angle={-35} textAnchor="end" interval={0} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
                    formatter={v => [`${(v * 100).toFixed(2)}%`, `R² (${activeY})`]}
                  />
                  <Bar dataKey="r2" radius={[4, 4, 0, 0]}>
                    {perYBar.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {/* Best per Y */}
              {best_per_y?.[activeY] && (
                <div className="mt-3 px-3 py-2 rounded-lg bg-gray-700/60 flex items-center gap-2 text-xs">
                  <span className="text-yellow-400">★</span>
                  <span className="text-gray-300">Best for <span className="text-white font-medium">{activeY}</span>:</span>
                  <span style={{ color: getMeta(best_per_y[activeY].model_type).color }} className="font-semibold">
                    {getMeta(best_per_y[activeY].model_type).fullName}
                  </span>
                  <span className="text-gray-400">Run #{best_per_y[activeY].run_id}</span>
                  <span className="text-white font-bold ml-auto">R² {(best_per_y[activeY].r2 * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT: run cards ── */}
        <div className="col-span-2 space-y-3 max-h-[700px] overflow-y-auto pr-1">
          {filtered.map(run => {
            const meta      = getMeta(run.model_type)
            const isBest    = best_overall?.run_id === run.id
            return (
              <div
                key={run.id}
                className={`rounded-xl p-4 border transition-colors ${
                  isBest ? `${meta.bg} ${meta.border}` : 'bg-gray-800 border-gray-700'
                }`}
              >
                {/* Card header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">{meta.icon}</span>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-white font-semibold text-sm">Run #{run.id}</span>
                        {isBest && <span className="text-yellow-400 text-xs">★ Best</span>}
                        {run.status === 'stopped' && (
                          <span className="text-orange-400 text-xs">partial</span>
                        )}
                      </div>
                      <span className="text-xs font-medium" style={{ color: meta.color }}>
                        {meta.fullName}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <R2Badge value={run.r2_score} />
                    <p className="text-gray-500 text-xs mt-0.5">overall R²</p>
                  </div>
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-3 gap-2 mb-3">
                  <div className="bg-gray-700/60 rounded-lg px-2 py-1.5 text-center">
                    <p className="text-blue-300 text-xs font-semibold">{run.x_columns?.length ?? 0}X</p>
                    <p className="text-gray-500 text-xs">inputs</p>
                  </div>
                  <div className="bg-gray-700/60 rounded-lg px-2 py-1.5 text-center">
                    <p className="text-green-300 text-xs font-semibold">{run.y_columns?.length ?? 0}Y</p>
                    <p className="text-gray-500 text-xs">outputs</p>
                  </div>
                  <div className="bg-gray-700/60 rounded-lg px-2 py-1.5 text-center">
                    <p className="text-gray-200 text-xs font-semibold">{run.n_test_rows?.toLocaleString() ?? '?'}</p>
                    <p className="text-gray-500 text-xs">test rows</p>
                  </div>
                </div>

                {/* Model params */}
                {run.model_params && Object.keys(run.model_params).length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {Object.entries(run.model_params).map(([k, v]) => (
                      <span key={k} className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded">
                        {k}: {v}
                      </span>
                    ))}
                  </div>
                )}

                {/* Per-Y R² mini table */}
                {run.per_y_r2 && Object.keys(run.per_y_r2).length > 0 && (
                  <div className="space-y-1">
                    {Object.entries(run.per_y_r2).map(([yc, r2]) => {
                      const isYBest = best_per_y?.[yc]?.run_id === run.id
                      const pct     = r2 != null ? Math.max(0, r2) * 100 : 0
                      const barCol  = r2 >= 0.9 ? '#4ade80' : r2 >= 0.7 ? '#fbbf24' : '#f87171'
                      return (
                        <div key={yc} className="flex items-center gap-2">
                          <span className="text-gray-400 text-xs w-28 truncate" title={yc}>
                            {isYBest && <span className="text-yellow-400 mr-1">★</span>}
                            {yc}
                          </span>
                          <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                            <div
                              className="h-1.5 rounded-full transition-all"
                              style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: barCol }}
                            />
                          </div>
                          <span className="text-xs font-bold w-10 text-right" style={{ color: barCol }}>
                            {r2 != null ? `${pct.toFixed(0)}%` : '—'}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Full per-Y comparison table ── */}
      {allYCols.length > 0 && (
        <div className="mt-6 bg-gray-800 rounded-xl p-5">
          <h2 className="text-white font-semibold mb-4">Full Per-Y R² Comparison Table</h2>
          <div className="overflow-auto rounded-lg border border-gray-700">
            <table className="w-full text-xs border-collapse">
              <thead className="sticky top-0 bg-gray-700">
                <tr>
                  <th className="px-3 py-2.5 text-left text-gray-300 border-r border-gray-600 whitespace-nowrap">Run</th>
                  <th className="px-3 py-2.5 text-left text-gray-300 border-r border-gray-600 whitespace-nowrap">Model</th>
                  <th className="px-3 py-2.5 text-center text-gray-300 border-r border-gray-600 whitespace-nowrap">Overall R²</th>
                  {allYCols.map(yc => (
                    <th
                      key={yc}
                      className="px-3 py-2.5 text-center whitespace-nowrap min-w-20"
                      style={{ color: best_per_y?.[yc] ? '#fbbf24' : '#9ca3af' }}
                    >
                      {yc}
                      {best_per_y?.[yc] && <span className="ml-1 text-yellow-400">★</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((run, ri) => {
                  const meta   = getMeta(run.model_type)
                  const isBest = best_overall?.run_id === run.id
                  return (
                    <tr
                      key={run.id}
                      className={`border-t border-gray-700 ${
                        isBest ? 'bg-blue-950/40' : ri % 2 === 0 ? 'bg-gray-800' : ''
                      }`}
                    >
                      <td className="px-3 py-2 text-gray-200 border-r border-gray-700 whitespace-nowrap">
                        #{run.id}
                        {isBest && <span className="ml-1 text-yellow-400">★</span>}
                      </td>
                      <td className="px-3 py-2 border-r border-gray-700 whitespace-nowrap">
                        <span className="font-semibold text-xs" style={{ color: meta.color }}>
                          {meta.icon} {meta.label}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center border-r border-gray-700">
                        <R2Badge value={run.r2_score} />
                      </td>
                      {allYCols.map(yc => {
                        const r2      = run.per_y_r2?.[yc]
                        const isYBest = best_per_y?.[yc]?.run_id === run.id
                        return (
                          <td key={yc} className={`px-3 py-2 text-center ${isYBest ? 'bg-yellow-900/20' : ''}`}>
                            {r2 != null ? (
                              <span className={`font-semibold ${
                                r2 >= 0.9 ? 'text-green-400' : r2 >= 0.7 ? 'text-yellow-400' : 'text-red-400'
                              }`}>
                                {(r2 * 100).toFixed(1)}%
                              </span>
                            ) : (
                              <span className="text-gray-600">—</span>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="text-gray-600 text-xs mt-2">
            ★ = best model for that Y output &nbsp;·&nbsp; R² on held-out test set &nbsp;·&nbsp; Higher is better (100% = perfect)
          </p>
        </div>
      )}
    </div>
  )
}
