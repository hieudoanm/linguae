import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Footer from '../Footer';

describe('Footer', () => {
  it('renders the copyright text with current year', () => {
    render(<Footer />);
    const year = new Date().getFullYear().toString();
    expect(screen.getByText(`© ${year} Linguae`)).toBeInTheDocument();
  });
});
