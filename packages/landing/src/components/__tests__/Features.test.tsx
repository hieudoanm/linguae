import { render, screen } from '@testing-library/react';
import Features from '../Features';

describe('Features', () => {
  it('renders the section heading', () => {
    render(<Features />);
    expect(
      screen.getByRole('heading', { name: 'Features' })
    ).toBeInTheDocument();
  });

  it('renders all feature cards', () => {
    render(<Features />);
    const titles = [
      'Local-first & private',
      'One command',
      'Confidence & top-k',
      'Batch mode',
      'Device selection',
      'Reproducible',
    ];
    for (const title of titles) {
      expect(screen.getByRole('heading', { name: title })).toBeInTheDocument();
    }
  });
});
