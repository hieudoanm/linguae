import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Navbar from '../Navbar';

describe('Navbar', () => {
  it('renders the logo and navigation links', () => {
    render(<Navbar scrollTo={() => {}} />);
    expect(screen.getByText('Linguae')).toBeInTheDocument();
    expect(screen.getByText('Features')).toBeInTheDocument();
    expect(screen.getByText('Detect')).toBeInTheDocument();
    expect(screen.getByText('About')).toBeInTheDocument();
  });

  it('calls scrollTo when a link is clicked', () => {
    const scrollTo = vi.fn();
    render(<Navbar scrollTo={scrollTo} />);
    fireEvent.click(screen.getByText('Features'));
    expect(scrollTo).toHaveBeenCalledWith('features');
  });
});
