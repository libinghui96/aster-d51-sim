import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: {
    host: true,
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8765",
        ws: true,
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: true,
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8765",
        ws: true,
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 15000,
  },
});
