import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The local FastAPI backend (run_api.py) listens on 127.0.0.1:8000.
// The frontend calls "/api/*" and Vite proxies it to the backend, so the
// browser never needs CORS and the app stays fully local.
const apiProxy = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: apiProxy },
  preview: { proxy: apiProxy },
})
