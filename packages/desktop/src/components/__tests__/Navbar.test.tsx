import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from '../ThemeProvider';
import Navbar from '../Navbar';

const renderWithProviders = (ui: React.ReactElement) =>
  render(<ThemeProvider>{ui}</ThemeProvider>);

describe('Navbar', () => {
  it('renders the logo and navigation links', () => {
    renderWithProviders(<Navbar scrollTo={() => {}} />);
    expect(screen.getByText('Linguae')).toBeInTheDocument();
    expect(screen.getByText('About')).toBeInTheDocument();
  });

  it('calls scrollTo with #hero when the logo is clicked', () => {
    const scrollTo = vi.fn();
    renderWithProviders(<Navbar scrollTo={scrollTo} />);
    fireEvent.click(screen.getByText('Linguae'));
    expect(scrollTo).toHaveBeenCalledWith('#hero');
  });
});
