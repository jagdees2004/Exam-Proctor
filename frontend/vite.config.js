import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/exam': {
        target: 'http://localhost:8000',
        ws: true,
      },
      '/health': 'http://localhost:8000',
    },
  },
})
