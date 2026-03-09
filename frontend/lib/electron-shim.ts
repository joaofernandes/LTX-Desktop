/**
 * Web-mode shim for window.electronAPI.
 *
 * Injected as the first import in main.tsx when building for the web.
 * Replaces Electron IPC calls with web-compatible stubs so the React
 * frontend can run against the FastAPI backend without Electron.
 */

const _backendUrl: string = (() => {
  // Allow override via build-time env var (e.g. VITE_BACKEND_URL=http://192.168.1.99:8001)
  const envUrl = (import.meta as unknown as Record<string, Record<string, string>>).env?.VITE_BACKEND_URL
  if (envUrl) return envUrl
  return `http://${window.location.hostname}:8001`
})()

const _noop = () => Promise.resolve(undefined)
const _noopVoid = () => undefined

window.electronAPI = {
  // Backend URL — points to the FastAPI server
  getBackendUrl: () => Promise.resolve(_backendUrl),
  getModelsPath: () => Promise.resolve(''),

  // First-run / setup — skip entirely in web mode
  checkFirstRun: () => Promise.resolve({ needsSetup: false, needsLicense: false }),
  acceptLicense: () => Promise.resolve(true),
  completeSetup: () => Promise.resolve(true),
  fetchLicenseText: () => Promise.resolve(''),
  getNoticesText: () => Promise.resolve(''),

  // Python / backend process — already running, always healthy
  checkPythonReady: () => Promise.resolve({ ready: true }),
  startPythonSetup: _noop,
  startPythonBackend: _noop,
  getBackendHealthStatus: () => Promise.resolve({ status: 'alive' }),
  onBackendHealthStatus: (cb) => {
    // Fire immediately so the app boots
    setTimeout(() => cb({ status: 'alive' }), 0)
    return _noopVoid
  },
  onPythonSetupProgress: _noopVoid,
  removePythonSetupProgress: _noopVoid,

  // GPU / app info
  checkGpu: () => Promise.resolve({ available: true }),
  getAppInfo: () =>
    Promise.resolve({ version: '1.0.0', isPackaged: false, modelsPath: '', userDataPath: '' }),

  // File system — not available in browser; return null / empty
  readLocalFile: () => Promise.resolve({ data: '', mimeType: '' }),
  showSaveDialog: () => Promise.resolve(null),
  showOpenFileDialog: () => Promise.resolve(null),
  showOpenDirectoryDialog: () => Promise.resolve(null),
  saveFile: () => Promise.resolve({ success: false, error: 'Not available in web mode' }),
  saveBinaryFile: () => Promise.resolve({ success: false, error: 'Not available in web mode' }),
  copyFile: () => Promise.resolve({ success: false, error: 'Not available in web mode' }),
  checkFilesExist: (paths) =>
    Promise.resolve(Object.fromEntries(paths.map((p) => [p, false]))),
  searchDirectoryForFiles: (_dir, filenames) =>
    Promise.resolve(Object.fromEntries(filenames.map((f) => [f, null]))),
  ensureDirectory: () => Promise.resolve({ success: false }),
  openParentFolderOfFile: _noop,
  showItemInFolder: _noop,

  // Logs
  getLogs: () => Promise.resolve({ logPath: '', lines: [] }),
  getLogPath: () => Promise.resolve({ logPath: '', logDir: '' }),
  openLogFolder: () => Promise.resolve(false),
  writeLog: _noop,

  // URLs / links
  openLtxApiKeyPage: () => Promise.resolve(false),
  openFalApiKeyPage: () => Promise.resolve(false),
  getResourcePath: () => Promise.resolve(null),
  getDownloadsPath: () => Promise.resolve(''),

  // Export (native ffmpeg) — not available in web mode
  exportNative: () =>
    Promise.resolve({ error: 'Native export not available in web mode' }),
  exportCancel: () => Promise.resolve({ ok: false }),

  // Video frame extraction — not available
  extractVideoFrame: () =>
    Promise.resolve({ path: '', url: '' }),

  // Analytics — silently disabled
  getAnalyticsState: () =>
    Promise.resolve({ analyticsEnabled: false, installationId: '' }),
  setAnalyticsEnabled: _noop,
  sendAnalyticsEvent: _noop,

  platform: 'web',
}
