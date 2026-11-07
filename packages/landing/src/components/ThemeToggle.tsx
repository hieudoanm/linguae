'use client';

import { type FC, useSyncExternalStore } from 'react';

export const THEME_STORAGE_KEY = 'linguae-theme';
export const LIGHT_THEME = 'linguae';
export const DARK_THEME = 'linguae-dark';
export type Theme = typeof LIGHT_THEME | typeof DARK_THEME;

const isTheme = (value: string | null): value is Theme =>
  value === LIGHT_THEME || value === DARK_THEME;

export const getTheme = (): Theme => {
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (isTheme(stored)) {
    return stored;
  }
  const explicit = document.documentElement.getAttribute('data-theme');
  if (isTheme(explicit)) {
    return explicit;
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? DARK_THEME
    : LIGHT_THEME;
};

const listeners = new Set<() => void>();

const subscribe = (onStoreChange: () => void) => {
  listeners.add(onStoreChange);
  window.addEventListener('storage', onStoreChange);
  return () => {
    listeners.delete(onStoreChange);
    window.removeEventListener('storage', onStoreChange);
  };
};

const getSnapshot = (): Theme => getTheme();
const getServerSnapshot = (): Theme => LIGHT_THEME;

export const applyTheme = (theme: Theme) => {
  document.documentElement.setAttribute('data-theme', theme);
  window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  listeners.forEach((listener) => listener());
};

const MoonIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="size-5">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);

const SunIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="size-5">
    <circle cx="12" cy="12" r="5" />
    <line x1="12" y1="1" x2="12" y2="3" />
    <line x1="12" y1="21" x2="12" y2="23" />
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
    <line x1="1" y1="12" x2="3" y2="12" />
    <line x1="21" y1="12" x2="23" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
  </svg>
);

const ThemeToggle: FC = () => {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const toggle = () => {
    applyTheme(theme === DARK_THEME ? LIGHT_THEME : DARK_THEME);
  };

  return (
    <button
      type="button"
      className="btn btn-ghost btn-circle"
      onClick={toggle}
      aria-label="Toggle color theme">
      {theme === DARK_THEME ? <SunIcon /> : <MoonIcon />}
    </button>
  );
};

export default ThemeToggle;
