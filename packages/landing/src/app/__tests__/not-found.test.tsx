import { render, screen } from '@testing-library/react';
import NotFound from '../not-found';

describe('not-found.tsx', () => {
  it('renders the 404 message', () => {
    render(<NotFound />);
    expect(screen.getByText('404')).toBeInTheDocument();
    expect(screen.getByText('Page not found')).toBeInTheDocument();
  });

  it('links back home', () => {
    render(<NotFound />);
    expect(screen.getByRole('link', { name: /go home/i })).toHaveAttribute(
      'href',
      '/'
    );
  });
});
