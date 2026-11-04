'use client';

import {
  createContext,
  useContext,
  useEffect,
  ReactNode,
  useSyncExternalStore,
} from 'react';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const getSnapshot = (): Theme => {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function')
    return 'light';
  const stored = localStorage.getItem('theme') as Theme | null;
  if (stored) return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
};

const subscribe = (callback: () => void) => {
  if (typeof window === 'undefined') return () => {};
  const handler = () => callback();
  window.addEventListener('storage', handler);
  let mediaQuery: MediaQueryList | null = null;
  if (typeof window.matchMedia === 'function') {
    mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', callback);
  }
  return () => {
    window.removeEventListener('storage', handler);
    if (mediaQuery) mediaQuery.removeEventListener('change', callback);
  };
};

const getServerSnapshot = (): Theme => {
  return 'light';
};

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  // Use useSyncExternalStore as the single source of truth for theme
  const theme = useSyncExternalStore<Theme>(
    subscribe,
    getSnapshot,
    getServerSnapshot
  );

  // Apply theme to document - this is a side effect, not state sync
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', newTheme);
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
    // Trigger a re-render by dispatching a storage event
    window.dispatchEvent(
      new StorageEvent('storage', { key: 'theme', newValue: newTheme })
    );
  };

  const setThemeDirect = (newTheme: Theme) => {
    localStorage.setItem('theme', newTheme);
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
    window.dispatchEvent(
      new StorageEvent('storage', { key: 'theme', newValue: newTheme })
    );
  };

  return (
    <ThemeContext.Provider
      value={{ theme, toggleTheme, setTheme: setThemeDirect }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
