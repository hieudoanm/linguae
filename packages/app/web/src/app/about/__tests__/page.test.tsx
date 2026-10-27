import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '@/components/ThemeProvider';
import About from '../page';

const renderWithProviders = (ui: React.ReactElement) => {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>{ui}</ThemeProvider>
    </QueryClientProvider>
  );
};

describe('About page', () => {
  it('renders marketing content', () => {
    renderWithProviders(<About />);
    fireEvent.click(screen.getByText('Linguae'));
    expect(screen.getByText('Linguae')).toBeInTheDocument();
  });

  it('scrolls when element exists', () => {
    const el = { scrollIntoView: vi.fn() };
    vi.spyOn(document, 'getElementById').mockReturnValue(el as any);
    const qc = new QueryClient();
    render(<QueryClientProvider client={qc}><ThemeProvider><About /></ThemeProvider></QueryClientProvider>);
    fireEvent.click(screen.getByText('Linguae'));
    expect(el.scrollIntoView).toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});
