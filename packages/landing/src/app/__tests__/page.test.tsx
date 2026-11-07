import { render, screen } from '@testing-library/react';
import Home from '../page';

describe('Home page', () => {
  it('renders the navbar', () => {
    render(<Home />);
    expect(
      screen.getAllByRole('link', { name: /linguae/i }).length
    ).toBeGreaterThan(0);
  });

  it('renders all page sections', () => {
    render(<Home />);
    expect(
      screen.getByRole('heading', {
        name: /detect the language of any text/i,
      })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'Features' })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'See it in action' })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'How it works' })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'Get started' })
    ).toBeInTheDocument();
  });
});
