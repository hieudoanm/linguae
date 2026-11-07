import { render, screen } from '@testing-library/react';
import Hero from '../Hero';

describe('Hero', () => {
  it('renders the headline', () => {
    render(<Hero />);
    expect(
      screen.getByRole('heading', {
        name: /detect the language of any text/i,
      })
    ).toBeInTheDocument();
  });

  it('renders call-to-action links', () => {
    render(<Hero />);
    expect(screen.getByRole('link', { name: /get started/i })).toHaveAttribute(
      'href',
      '#install'
    );
    expect(
      screen.getByRole('link', { name: /see it in action/i })
    ).toHaveAttribute('href', '#demo');
  });

  it('renders stats', () => {
    render(<Hero />);
    expect(screen.getByText('235k+')).toBeInTheDocument();
    expect(screen.getByText('network calls')).toBeInTheDocument();
  });
});
