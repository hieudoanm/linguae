import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from '@/components/ThemeProvider';
import About from '../page';

const renderWithProviders = (ui: React.ReactElement) =>
  render(<ThemeProvider>{ui}</ThemeProvider>);

describe('About page', () => {
  it('renders marketing content', () => {
    renderWithProviders(<About />);
    fireEvent.click(screen.getByText('Linguae'));
    expect(screen.getByText('Linguae')).toBeInTheDocument();
  });

  it('scrolls when element exists', () => {
    const el = { scrollIntoView: vi.fn() };
    vi.spyOn(document, 'getElementById').mockReturnValue(
      el as unknown as HTMLElement
    );
    renderWithProviders(<About />);
    fireEvent.click(screen.getByText('Linguae'));
    expect(el.scrollIntoView).toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});
