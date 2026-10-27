import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '../ThemeProvider';
import Navbar from '../Navbar';

const renderWithProviders = (ui: React.ReactElement) => {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>{ui}</ThemeProvider>
    </QueryClientProvider>
  );
};

describe('Navbar', () => {
  it('renders the logo and navigation links', () => {
    renderWithProviders(<Navbar scrollTo={() => {}} />);
    expect(screen.getByText('Linguae')).toBeInTheDocument();
    expect(screen.getByText('Features')).toBeInTheDocument();
    expect(screen.getByText('Detect')).toBeInTheDocument();
    expect(screen.getByText('About')).toBeInTheDocument();
  });

  it('calls scrollTo when a link is clicked', () => {
    const scrollTo = vi.fn();
    renderWithProviders(<Navbar scrollTo={scrollTo} />);
    fireEvent.click(screen.getByText('Features'));
    expect(scrollTo).toHaveBeenCalledWith('features');
  });

  it('calls scrollTo with #hero when the logo is clicked', () => {
    const scrollTo = vi.fn();
    renderWithProviders(<Navbar scrollTo={scrollTo} />);
    fireEvent.click(screen.getByText('Linguae'));
    expect(scrollTo).toHaveBeenCalledWith('#hero');
  });
});
