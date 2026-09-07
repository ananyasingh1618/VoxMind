# Development-mode frontend image: runs Vite's dev server (with HMR) inside
# the container. A production image (static build served by nginx) belongs
# to the Phase 10 deployment work, not Phase 1 - documented as a known gap,
# not silently assumed to exist.
FROM node:20-slim AS base

WORKDIR /app

COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm install

COPY apps/web/ ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
