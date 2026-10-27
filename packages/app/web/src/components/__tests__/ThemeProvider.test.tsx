import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../ThemeProvider';

const UseTheme = () => {
  const { theme, toggleTheme, setTheme } = useTheme();
  return (
    <div>
      <span data-testid="theme">{theme}</span>
      <button onClick={toggleTheme} data-testid="toggle">
        toggle
      </button>
      <button onClick={() => setTheme('dark')} data-testid="set-dark">
        set dark
      </button>
      <button onClick={() => setTheme('light')} data-testid="set-light">
        set light
      </button>
    </div>
  );
};

const renderWithTheme = (ui: React.ReactElement) => {
  return render(<ThemeProvider>{ui}</ThemeProvider>);
};

describe('ThemeProvider', () => {
  beforeEach(() => {
    localStorage.clear();
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(() => ({
        matches: false,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      })),
    });
  });

  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('defaults to light theme', () => {
    renderWithTheme(<UseTheme />);
    expect(screen.getByTestId('theme')).toHaveTextContent('light');
  });

  it('toggles theme and persists to localStorage', () => {
    renderWithTheme(<UseTheme />);
    fireEvent.click(screen.getByTestId('toggle'));
    expect(screen.getByTestId('theme')).toHaveTextContent('dark');
    expect(localStorage.getItem('theme')).toBe('dark');
    fireEvent.click(screen.getByTestId('toggle'));
    expect(screen.getByTestId('theme')).toHaveTextContent('light');
  });

  it('setTheme writes theme to localStorage', () => {
    renderWithTheme(<UseTheme />);
    fireEvent.click(screen.getByTestId('set-dark'));
    expect(screen.getByTestId('theme')).toHaveTextContent('dark');
    expect(localStorage.getItem('theme')).toBe('dark');
    fireEvent.click(screen.getByTestId('set-light'));
    expect(screen.getByTestId('theme')).toHaveTextContent('light');
  });

  it('uses stored theme from localStorage', () => {
    localStorage.setItem('theme', 'dark');
    renderWithTheme(<UseTheme />);
    expect(screen.getByTestId('theme')).toHaveTextContent('dark');
  });

  it('uses dark when prefers-color-scheme is dark', () => {
    (window.matchMedia as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      matches: true,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }));
    localStorage.clear();
    renderWithTheme(<UseTheme />);
    expect(screen.getByTestId('theme')).toHaveTextContent('dark');
  });

  it('throws when useTheme is used outside a ThemeProvider', () => {
    const Thrower = () => {
      useTheme();
      return null;
    };
    expect(() => render(<Thrower />)).toThrow('useTheme must be used within a ThemeProvider');
  });
});
