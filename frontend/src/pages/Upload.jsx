import { useState, useRef } from 'react'
import { uploadDataset, listDatasets, previewDataset } from '../services/api'

export default function Upload() {
  const [file, setFile]         = useState(null)
  const [uploading, setUploading] = useState(false)
  const [datasets, setDatasets]   = useState([])
  const [preview, setPreview]     = useState(null)
  const [message, setMessage]     = useState('')
  const fileRef = useRef()

  const loadDatasets = () =>
    listDatasets().then(r => setDatasets(r.data)).catch(() => {})

  useState(() => { loadDatasets() }, [])

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setMessage('')
    try {
      const fd = new FormData()
      fd.append('file', file)
      await uploadDataset(fd)
      setMessage('File uploaded successfully!')
      setFile(null)
      fileRef.current.value = ''
      loadDatasets()
    } catch (e) {
      setMessage('Upload failed: ' + (e.response?.data?.detail || e.message))
    }
    setUploading(false)
  }

  const handlePreview = async (id) => {
    const r = await previewDataset(id, 8)
    setPreview(r.data)
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-1">Upload Dataset</h1>
      <p className="text-gray-200 text-sm mb-6">
        Upload a CSV or Excel file. Only numeric columns will be used for modeling.
      </p>

      {/* Upload card */}
      <div className="bg-gray-800 rounded-xl p-6 mb-6">
        <label className="block text-sm text-gray-100 mb-2">Select file (CSV or .xlsx)</label>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={e => setFile(e.target.files[0])}
          className="block w-full text-sm text-gray-200 file:mr-4 file:py-2 file:px-4
                     file:rounded-lg file:border-0 file:bg-blue-600 file:text-white
                     file:cursor-pointer hover:file:bg-blue-500 mb-4"
        />
        {file && (
          <p className="text-gray-200 text-xs mb-3">
            Selected: <span className="text-white">{file.name}</span> ({(file.size / 1024).toFixed(1)} KB)
          </p>
        )}
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed
                     text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          {uploading ? 'Uploading...' : 'Upload File'}
        </button>
        {message && (
          <p className={`mt-3 text-sm ${message.includes('success') ? 'text-green-400' : 'text-red-400'}`}>
            {message}
          </p>
        )}
      </div>

      {/* Dataset list */}
      {datasets.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6 mb-6">
          <h2 className="text-white font-semibold mb-4">Uploaded Datasets</h2>
          <div className="space-y-3">
            {datasets.map(d => (
              <div key={d.id} className="flex items-center justify-between bg-gray-700 rounded-lg px-4 py-3">
                <div>
                  <p className="text-white text-sm font-medium">{d.original_name}</p>
                  <p className="text-gray-200 text-xs">
                    {d.row_count} rows · {d.columns.length} numeric columns · ID: {d.id}
                  </p>
                  <p className="text-gray-500 text-xs mt-0.5">
                    Columns: {d.columns.join(', ')}
                  </p>
                </div>
                <button
                  onClick={() => handlePreview(d.id)}
                  className="text-blue-400 hover:text-blue-300 text-xs border border-blue-700 rounded px-3 py-1"
                >
                  Preview
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Preview table */}
      {preview && (
        <div className="bg-gray-800 rounded-xl p-6 overflow-x-auto">
          <h2 className="text-white font-semibold mb-3">Data Preview (first 8 rows)</h2>
          <table className="w-full text-xs text-gray-100">
            <thead>
              <tr>
                {preview.columns.map(c => (
                  <th key={c} className="text-left text-gray-200 pb-2 pr-4 border-b border-gray-600">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, i) => (
                <tr key={i} className="border-b border-gray-700">
                  {preview.columns.map(c => (
                    <td key={c} className="py-1.5 pr-4">{row[c]?.toFixed?.(4) ?? row[c]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
