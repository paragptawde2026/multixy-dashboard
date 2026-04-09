import { useEffect, useState, useCallback } from 'react'
import { listRuns, deleteRun, bulkDeleteRuns } from '../services/api'

// ── Status styles ──────────────────────────────────────────────────────────────
const STATUS_COLOR = {
  pending:  'bg-yellow-900/60 text-yellow-300 border-yellow-700',
  training: 'bg-blue-900/60  text-blue-300  border-blue-700',
  done:     'bg-green-900/60 text-green-300 border-green-700',
  stopped:  'bg-orange-900/60 text-orange-300 border-orange-700',
  error:    'bg-red-900/60   text-red-300   border-red-700',
}

const STATUS_LABEL = { pending: 'Pending', training: 'Training', done: 'Done', stopped: 'Partial', error: 'Error' }

function r2Color(v) {
  if (v == null) return 'text-gray-500'
  if (v >= 0.9)  return 'text-green-400'
  if (v >= 0.8)  return 'text-blue-400'
  if (v >= 0.6)  return 'text-yellow-400'
  return 'text-red-400'
}

// ── Component ──────────────────────────────────────────────────────────────────
export default function History() {
  const [runs, setRuns]               = useState([])
  const [loading, setLoading]         = useState(true)

  // Per-run delete
  const [confirmId, setConfirmId]     = useState(null)
  const [deleting, setDeleting]       = useState(false)

  // Bulk / cleanup
  const [keepLast, setKeepLast]       = useState(10)
  const [bulkConfirm, setBulkConfirm] = useState(null)   // 'cleanup' | 'errors' | 'stopped'
  const [bulking, setBulking]         = useState(false)

  // Filter
  const [filterStatus, setFilterStatus] = useState('all')

  const [error, setError]             = useState('')

  const refresh = useCallback(() => {
    setLoading(true)
    listRuns()
      .then(r => setRuns(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // ── Single delete ──────────────────────────────────────────────────────────
  const handleDelete = async (id) => {
    setDeleting(true)
    setError('')
    try {
      await deleteRun(id)
      setConfirmId(null)
      setRuns(prev => prev.filter(r => r.id !== id))
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setConfirmId(null)
    }
    setDeleting(false)
  }

  // ── Bulk delete ────────────────────────────────────────────────────────────
  const handleBulk = async (type) => {
    setBulking(true)
    setError('')
    try {
      if (type === 'cleanup') {
        await bulkDeleteRuns(keepLast, null)
      } else {
        await bulkDeleteRuns(0, type)   // 'error' | 'stopped'
      }
      setBulkConfirm(null)
      refresh()
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setBulkConfirm(null)
    }
    setBulking(false)
  }

  // ── Derived data ───────────────────────────────────────────────────────────
  const counts = runs.reduce((acc, r) => {
    acc[r.status] = (acc[r.status] || 0) + 1
    return acc
  }, {})

  const doneRuns    = runs.filter(r => r.status === 'done')
  const bestR2      = doneRuns.reduce((best, r) => {
    const v = r.metrics?.r2_score ?? 0
    return v > best ? v : best
  }, 0)

  const visible = filterStatus === 'all'
    ? runs
    : runs.filter(r => r.status === filterStatus)

  const wouldDeleteCount = runs.filter(r => {
    if (r.status === 'done') {
      const sorted = [...doneRuns].sort((a, b) => b.id - a.id)
      return sorted.indexOf(r) >= keepLast
    }
    return r.status !== 'pending' && r.status !== 'training'
  }).length

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="p-8 max-w-5xl mx-auto">

      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Training History</h1>
          <p className="text-gray-200 text-sm">Manage all past model training runs.</p>
        </div>
        {/* Summary badges */}
        <div className="flex gap-2 flex-wrap justify-end">
          {Object.entries(counts).map(([s, n]) => (
            <span key={s}
              className={`text-xs px-2.5 py-1 rounded-full border font-medium ${STATUS_COLOR[s] ?? 'bg-gray-800 text-gray-100 border-gray-700'}`}>
              {n} {STATUS_LABEL[s] ?? s}
            </span>
          ))}
          {doneRuns.length > 0 && (
            <span className="text-xs px-2.5 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-100">
              Best R² <span className={`font-bold ${r2Color(bestR2)}`}>{bestR2.toFixed(4)}</span>
            </span>
          )}
        </div>
      </div>

      {/* ── Cleanup Panel ── */}
      <div className="bg-gray-800 rounded-xl p-5 mb-6">
        <h2 className="text-white font-semibold mb-4 flex items-center gap-2">
          <span>🗂</span> Run Retention Settings
        </h2>

        <div className="flex flex-wrap items-end gap-4">
          {/* Keep last N */}
          <div>
            <label className="block text-gray-200 text-xs mb-1">
              Keep last <span className="text-white font-semibold">{keepLast}</span> successful runs
            </label>
            <div className="flex items-center gap-2">
              <input
                type="range" min="1" max="30" step="1"
                value={keepLast}
                onChange={e => setKeepLast(Number(e.target.value))}
                className="w-36 accent-blue-500"
              />
              <input
                type="number" min="1" max="100"
                value={keepLast}
                onChange={e => setKeepLast(Math.max(1, Number(e.target.value)))}
                className="w-16 bg-gray-700 border border-gray-600 rounded px-2 py-1
                           text-white text-sm text-center focus:border-blue-500 focus:outline-none"
              />
            </div>
            <p className="text-gray-600 text-xs mt-1">
              {doneRuns.length} successful runs · keeps top {Math.min(keepLast, doneRuns.length)},
              {' '}removes {Math.max(0, doneRuns.length - keepLast)} older ones
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2 flex-wrap">
            {bulkConfirm === 'cleanup' ? (
              <div className="flex items-center gap-2 bg-red-900/40 border border-red-700 rounded-lg px-3 py-2">
                <span className="text-red-300 text-xs">
                  Delete {Math.max(0, doneRuns.length - keepLast)} old run(s) + their files?
                </span>
                <button
                  onClick={() => handleBulk('cleanup')}
                  disabled={bulking}
                  className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-3 py-1
                             rounded text-xs font-bold"
                >
                  {bulking ? '…' : 'Confirm'}
                </button>
                <button onClick={() => setBulkConfirm(null)}
                  className="text-gray-200 hover:text-white text-xs">Cancel</button>
              </div>
            ) : (
              <button
                onClick={() => setBulkConfirm('cleanup')}
                disabled={doneRuns.length <= keepLast}
                className="bg-blue-700 hover:bg-blue-600 disabled:opacity-40
                           disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg
                           text-sm font-medium transition-colors"
              >
                🧹 Keep Last {keepLast} — Remove {Math.max(0, doneRuns.length - keepLast)} older
              </button>
            )}

            {bulkConfirm === 'error' ? (
              <div className="flex items-center gap-2 bg-red-900/40 border border-red-700 rounded-lg px-3 py-2">
                <span className="text-red-300 text-xs">Delete all {counts.error ?? 0} error run(s)?</span>
                <button onClick={() => handleBulk('error')} disabled={bulking}
                  className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-3 py-1 rounded text-xs font-bold">
                  {bulking ? '…' : 'Confirm'}
                </button>
                <button onClick={() => setBulkConfirm(null)} className="text-gray-200 hover:text-white text-xs">Cancel</button>
              </div>
            ) : (
              <button
                onClick={() => setBulkConfirm('error')}
                disabled={!counts.error}
                className="bg-gray-700 hover:bg-red-800 disabled:opacity-40 disabled:cursor-not-allowed
                           text-red-400 border border-red-800 hover:border-red-600 px-4 py-2
                           rounded-lg text-sm font-medium transition-colors"
              >
                🗑 Delete All Errors ({counts.error ?? 0})
              </button>
            )}

            {bulkConfirm === 'stopped' ? (
              <div className="flex items-center gap-2 bg-red-900/40 border border-red-700 rounded-lg px-3 py-2">
                <span className="text-red-300 text-xs">Delete all {counts.stopped ?? 0} stopped run(s)?</span>
                <button onClick={() => handleBulk('stopped')} disabled={bulking}
                  className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-3 py-1 rounded text-xs font-bold">
                  {bulking ? '…' : 'Confirm'}
                </button>
                <button onClick={() => setBulkConfirm(null)} className="text-gray-200 hover:text-white text-xs">Cancel</button>
              </div>
            ) : (
              <button
                onClick={() => setBulkConfirm('stopped')}
                disabled={!counts.stopped}
                className="bg-gray-700 hover:bg-orange-900 disabled:opacity-40 disabled:cursor-not-allowed
                           text-orange-400 border border-orange-800 hover:border-orange-600 px-4 py-2
                           rounded-lg text-sm font-medium transition-colors"
              >
                🗑 Delete Partial Runs ({counts.stopped ?? 0})
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-gray-500 text-sm">Filter:</span>
        {['all', 'done', 'stopped', 'error', 'pending', 'training'].map(s => (
          <button
            key={s}
            onClick={() => setFilterStatus(s)}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors capitalize ${
              filterStatus === s
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-200 hover:text-white border border-gray-700'
            }`}
          >
            {s === 'all' ? `All (${runs.length})` : `${STATUS_LABEL[s] ?? s} (${counts[s] ?? 0})`}
          </button>
        ))}
        <button onClick={refresh}
          className="ml-auto text-gray-500 hover:text-white text-xs border border-gray-700
                     hover:border-gray-500 px-3 py-1 rounded-lg transition-colors">
          ↺ Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 mb-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Run list */}
      {loading ? (
        <p className="text-gray-500 text-sm">Loading…</p>
      ) : visible.length === 0 ? (
        <div className="bg-gray-800 rounded-xl p-10 text-center">
          <p className="text-gray-500">No runs match this filter.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {visible.map(run => {
            const isDone    = run.status === 'done'
            const isActive  = run.status === 'pending' || run.status === 'training'
            const r2        = run.metrics?.r2_score
            const isBest    = isDone && r2 != null && r2 === bestR2 && doneRuns.length > 1

            return (
              <div key={run.id}
                className="bg-gray-800 rounded-xl p-5 border border-transparent
                           hover:border-gray-700 transition-colors">

                {/* Run header */}
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-white font-semibold">Run #{run.id}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full border font-medium
                                     ${STATUS_COLOR[run.status] ?? 'bg-gray-700 text-gray-100 border-gray-600'}`}>
                      {STATUS_LABEL[run.status] ?? run.status}
                    </span>
                    {isBest && (
                      <span className="text-xs bg-yellow-900/60 border border-yellow-700
                                       text-yellow-300 px-2 py-0.5 rounded-full">
                        ★ Best Model
                      </span>
                    )}
                    {r2 != null && (
                      <span className={`text-sm font-bold ${r2Color(r2)}`}>
                        R² {r2.toFixed(4)}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-3 flex-shrink-0">
                    <span className="text-gray-500 text-xs">
                      {new Date(run.created_at).toLocaleString()}
                    </span>
                    {/* Delete button */}
                    {!isActive && (
                      confirmId === run.id ? (
                        <div className="flex items-center gap-2">
                          <span className="text-red-300 text-xs">Delete Run #{run.id}?</span>
                          <button
                            onClick={() => handleDelete(run.id)}
                            disabled={deleting}
                            className="bg-red-600 hover:bg-red-500 disabled:opacity-50
                                       text-white px-3 py-1 rounded text-xs font-bold"
                          >{deleting ? '…' : 'Yes'}</button>
                          <button onClick={() => setConfirmId(null)}
                            className="text-gray-200 hover:text-white text-xs">Cancel</button>
                        </div>
                      ) : (
                        <button
                          onClick={() => { setConfirmId(run.id); setError('') }}
                          className="text-gray-600 hover:text-red-400 transition-colors"
                          title="Delete this run"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round"
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      )
                    )}
                  </div>
                </div>

                {/* Run params grid */}
                <div className="grid grid-cols-3 gap-x-6 gap-y-1 mt-3 text-sm">
                  <div><span className="text-gray-500">Epochs </span>
                    <span className="text-gray-200">{run.metrics?.epochs_completed ?? run.epochs}</span></div>
                  <div><span className="text-gray-500">Hidden Dim </span>
                    <span className="text-gray-200">{run.hidden_dim}</span></div>
                  <div><span className="text-gray-500">Noise </span>
                    <span className="text-gray-200">{run.noise_factor}</span></div>
                  <div className="col-span-3">
                    <span className="text-gray-500">X ({run.x_columns?.length}) </span>
                    <span className="text-blue-300 text-xs">{run.x_columns?.join(', ')}</span>
                  </div>
                  <div className="col-span-3">
                    <span className="text-gray-500">Y ({run.y_columns?.length}) </span>
                    <span className="text-green-300 text-xs">{run.y_columns?.join(', ')}</span>
                  </div>
                </div>

                {/* Metrics */}
                {isDone && run.metrics && (
                  <div className="mt-3 pt-3 border-t border-gray-700 flex flex-wrap gap-6 text-sm">
                    <span><span className="text-gray-500">Train Loss </span>
                      <span className="font-mono text-white">{run.metrics.train_loss_final?.toFixed(6) ?? '—'}</span></span>
                    <span><span className="text-gray-500">Val Loss </span>
                      <span className="font-mono text-white">{run.metrics.val_loss_final?.toFixed(6) ?? '—'}</span></span>
                    <span><span className="text-gray-500">Train Rows </span>
                      <span className="text-white">{run.metrics.n_train_rows?.toLocaleString() ?? '—'}</span></span>
                    <span><span className="text-gray-500">Test Rows </span>
                      <span className="text-white">{run.metrics.n_test_rows?.toLocaleString() ?? '—'}</span></span>
                  </div>
                )}
                {run.status === 'stopped' && (
                  <p className="text-orange-400 text-xs mt-2 border-t border-gray-700 pt-2">
                    Stopped early — partial model saved ({run.metrics?.epochs_completed ?? '?'} epochs)
                  </p>
                )}
                {run.status === 'error' && (
                  <p className="text-red-400 text-xs mt-2 border-t border-gray-700 pt-2">
                    {run.metrics?.error}
                  </p>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
