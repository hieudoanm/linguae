/**
 * Linguae Service Worker
 *
 * Provides offline-first behavior for the Linguae web app:
 * - Caches static assets (app shell, icons, fonts) on install
 * - Stale-while-revalidate for cached assets
 * - Network-first for API/detection requests
 * - Navigation fallback to cached index.html
 */

const CACHE_NAME = 'linguae-v1';

const STATIC_ASSETS = [
  '/',
  '/favicon.svg',
  '/logo.svg',
  '/logo-light.svg',
  '/og-image.svg',
  '/og-image-light.svg',
  '/manifest.json',
  '/site.webmanifest',
  '/robots.txt',
  '/sitemap.xml',
  '/humans.txt',
];

const CACHE_FIRST_PATTERNS = [
  /\.(png|jpg|jpeg|svg|gif|webp|ico)$/,
  /\.(woff|woff2|ttf|otf|eot)$/,
  /\/_next\/static\//,
];

const NETWORK_FIRST_PATTERNS = [
  /\/detect/,
  /\/detect\/form/,
  /\/detect\/batch/,
  /\/detect\/file/,
  /\/health/,
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Only handle same-origin GET requests
  if (request.method !== 'GET' || !self.location) {
    return;
  }

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    return;
  }

  // Network-first for API endpoints
  if (NETWORK_FIRST_PATTERNS.some((p) => p.test(url.pathname))) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Cache-first for static assets
  if (CACHE_FIRST_PATTERNS.some((p) => p.test(url.pathname))) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Stale-while-revalidate for navigation and other requests
  event.respondWith(staleWhileRevalidate(request));
});

function cacheFirst(request) {
  return caches.match(request).then((cached) => {
    if (cached) return cached;
    return fetch(request).then((response) => {
      if (response && response.status === 200) {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
      }
      return response;
    });
  });
}

function networkFirst(request) {
  return fetch(request)
    .then((response) => {
      if (response && response.status === 200) {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
      }
      return response;
    })
    .catch(() => caches.match(request));
}

function staleWhileRevalidate(request) {
  return caches.match(request).then((cached) => {
    const fetchPromise = fetch(request)
      .then((response) => {
        if (response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
        }
        return response;
      })
      .catch(() => null);

    return cached || fetchPromise;
  });
}