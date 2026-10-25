FROM node:22-alpine

WORKDIR /app

COPY packages/app/web/web/package*.json ./packages/app/web/web/
RUN cd packages/app/web/web && npm ci

COPY packages/app/web/web ./packages/app/web/web

WORKDIR /app/packages/app/web/web

EXPOSE 3000

CMD ["npm", "run", "dev"]
