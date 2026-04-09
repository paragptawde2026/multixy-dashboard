import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listDatasets, listRuns } from '../services/api'

export default function Home() {
  const [datasets, setDatasets] = useState([])
  const [runs, setRuns]         = useState([])

  useEffect(() => {
    listDatasets().then(r => setDatasets(r.data)).catch(() => {})
    listRuns().then(r => setRuns(r.data)).catch(() => {})
  }, [])

  const doneRuns = runs.filter(r => r.status === 'done').length

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-white mb-2">
        Multi X-Y Denoising AutoEncoder
      </h1>
      <p className="text-gray-200 mb-4 text-sm">
        Train a neural model that learns mappings from multiple input variables (X) to
        multiple output variables (Y), using a noise-robust autoencoder architecture.
      </p>
      <div className="bg-gray-800 border border-gray-700 rounded-xl px-5 py-3 mb-6 text-sm">
        <span className="text-gray-200">Dataset: </span>
        <span className="text-white font-medium">Data_DAE.xlsx</span>
        <span className="text-gray-500 mx-3">|</span>
        <span className="text-blue-400 font-medium">32 X inputs</span>
        <span className="text-gray-500 mx-2">→</span>
        <span className="text-green-400 font-medium">7 Y outputs</span>
        <span className="text-gray-500 mx-3">|</span>
        <span className="text-gray-200">44,788 rows</span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-10">
        {[
          { label: 'Datasets uploaded',  value: datasets.length },
          { label: 'Training runs',      value: runs.length },
          { label: 'Completed models',   value: doneRuns },
        ].map(s => (
          <div key={s.label} className="bg-gray-800 rounded-xl p-5 text-center">
            <p className="text-3xl font-bold text-blue-400">{s.value}</p>
            <p className="text-gray-200 text-sm mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Steps */}
      <h2 className="text-lg font-semibold text-gray-200 mb-4">How to use this dashboard</h2>
      <div className="grid grid-cols-2 gap-4">
        {[
          { step: '1', title: 'Upload Data',    desc: 'Upload a CSV or Excel file with your numeric columns.', to: '/upload' },
          { step: '2', title: 'Train Model',    desc: 'Choose X columns (inputs) and Y columns (outputs), set parameters, start training.', to: '/train' },
          { step: '3', title: 'Run Predictions',desc: 'Enter new X values and get predicted Y values from your trained model.', to: '/predict' },
          { step: '4', title: 'View History',   desc: 'Review all past training runs and their accuracy metrics.', to: '/history' },
        ].map(s => (
          <Link
            key={s.step}
            to={s.to}
            className="bg-gray-800 hover:bg-gray-700 transition-colors rounded-xl p-5 block"
          >
            <span className="text-blue-400 font-bold text-lg">Step {s.step}</span>
            <h3 className="text-white font-semibold mt-1">{s.title}</h3>
            <p className="text-gray-200 text-sm mt-1">{s.desc}</p>
          </Link>
        ))}
      </div>

      {/* What is DAE */}
      <div className="mt-10 bg-gray-800 rounded-xl p-6">
        <h2 className="text-white font-semibold mb-2">What is a Denoising AutoEncoder?</h2>
        <p className="text-gray-200 text-sm leading-relaxed">
          A Denoising AutoEncoder (DAE) is a neural network trained to predict clean outputs even
          when the inputs are slightly noisy or imperfect. It works by:<br /><br />
          <span className="text-gray-100">1.</span> Adding random noise to your input data (X)<br />
          <span className="text-gray-100">2.</span> Passing through an Encoder that compresses the data<br />
          <span className="text-gray-100">3.</span> Passing through a Decoder that reconstructs the clean output (Y)<br />
          <span className="text-gray-100">4.</span> Learning to minimize the error between predicted and actual Y<br /><br />
          This makes it ideal for real-world data that may have measurement errors or missing values.
        </p>
      </div>
    </div>
  )
}
