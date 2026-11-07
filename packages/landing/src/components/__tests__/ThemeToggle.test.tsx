import { fireEvent, render, screen } from '@testing-library/react';
import ThemeToggle, {
  DARK_THEME,
  LIGHT_THEME,
  THEME_STORAGE_KEY,
  applyTheme,
  getTheme,
} from '../ThemeToggle';

describe('ThemeToggle', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  it('renders a toggle button', () => {
    render(<ThemeToggle />);
    expect(
      screen.getByRole('button', { name: /toggle color theme/i })
    ).toBeInTheDocument();
  });

  it('switches to the dark theme on click and persists it', () => {
    render(<ThemeToggle />);
    fireEvent.click(screen.getByRole('button', { name: /toggle/i }));
    expect(document.documentElement.getAttribute('data-theme')).toBe(
      DARK_THEME
    );
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe(DARK_THEME);
  });

  it('switches back to the light theme on second click', () => {
    render(<ThemeToggle />);
    const button = screen.getByRole('button', { name: /toggle/i });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(document.documentElement.getAttribute('data-theme')).toBe(
      LIGHT_THEME
    );
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe(LIGHT_THEME);
  });
});

describe('getTheme', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  it('returns the stored theme when present', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, DARK_THEME);
    expect(getTheme()).toBe(DARK_THEME);
  });

  it('falls back to the data-theme attribute', () => {
    document.documentElement.setAttribute('data-theme', DARK_THEME);
    expect(getTheme()).toBe(DARK_THEME);
  });

  it('defaults to the light theme', () => {
    expect(getTheme()).toBe(LIGHT_THEME);
  });

  it('follows prefers-color-scheme when nothing is stored', () => {
    window.matchMedia = (query: string) =>
      ({
        matches: query === '(prefers-color-scheme: dark)',
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList;
    expect(getTheme()).toBe(DARK_THEME);
  });
});

describe('applyTheme', () => {
  it('sets the data-theme attribute and localStorage', () => {
    applyTheme(DARK_THEME);
    expect(document.documentElement.getAttribute('data-theme')).toBe(
      DARK_THEME
    );
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe(DARK_THEME);
  });
});
