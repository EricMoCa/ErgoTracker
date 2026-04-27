import axios from 'axios'

const api = axios.create({ baseURL: '' })

export async function checkHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function startAnalysis(formData) {
  const { data } = await api.post('/analyze/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function pollJob(jobId) {
  const { data } = await api.get(`/analyze/${jobId}`)
  return data
}

export async function extractRules(formData) {
  const { data } = await api.post('/rules/extract', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function pollRulesJob(jobId) {
  const { data } = await api.get(`/rules/${jobId}`)
  return data
}

export function getReportUrl(reportId) {
  return `/reports/${reportId}/download`
}

export function getAnnotatedVideoUrl(reportId) {
  return `/reports/${reportId}/video`
}

export async function getSetupStatus() {
  const { data } = await api.get('/setup/status')
  return data
}

export async function startModelDownload(modelName) {
  const { data } = await api.post(`/setup/download/${modelName}`)
  return data
}

export async function pollDownloadJob(jobId) {
  const { data } = await api.get(`/setup/download/${jobId}`)
  return data
}
