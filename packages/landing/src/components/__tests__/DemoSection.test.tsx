import { render, screen } from '@testing-library/react';
import DemoSection from '../DemoSection';

describe('DemoSection', () => {
  it('renders the section heading', () => {
    render(<DemoSection />);
    expect(
      screen.getByRole('heading', { name: 'See it in action' })
    ).toBeInTheDocument();
  });

  it('renders example commands and outputs', () => {
    render(<DemoSection />);
    expect(
      screen.getByText(/linguae "Bonjour tout le monde/)
    ).toBeInTheDocument();
    expect(screen.getByText('fra')).toBeInTheDocument();
    expect(screen.getByText(/eng/)).toBeInTheDocument();
    expect(screen.getByText(/"language":"rus"/)).toBeInTheDocument();
  });

  it('renders example window titles', () => {
    render(<DemoSection />);
    expect(screen.getByText('French sentence')).toBeInTheDocument();
    expect(screen.getByText('Confidence score')).toBeInTheDocument();
    expect(screen.getByText('JSON output')).toBeInTheDocument();
  });
});
