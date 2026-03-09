import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

// Rename index.web.html → index.html in the output directory
function renameHtmlOutput(): Plugin {
  return {
    name: 'rename-html-output',
    closeBundle() {
      const src = path.resolve(__dirname, 'dist/index.web.html')
      const dst = path.resolve(__dirname, 'dist/index.html')
      if (fs.existsSync(src)) {
        fs.renameSync(src, dst)
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), renameHtmlOutput()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend'),
    },
  },
  base: '/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(__dirname, 'index.web.html'),
    },
  },
  root: '.',
})
