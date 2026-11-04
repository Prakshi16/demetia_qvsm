import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Every component calls the API with a relative path (`/api/v1/...`), which
    // would otherwise hit the Vite dev server on :5173 and 404. Proxying keeps
    // the same relative URLs working in dev and in a real deployment, and avoids
    // needing CORS or an environment-specific base URL in the components.
    proxy: {
      // DEV_API_TARGET overrides the backend address when :8000 is taken by
      // something else (e.g. DEV_API_TARGET=http://localhost:8001 npm run dev).
      '/api': {
        target: process.env.DEV_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
