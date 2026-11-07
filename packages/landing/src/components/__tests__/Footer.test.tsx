import { render, screen } from '@testing-library/react';
import Footer from '../Footer';

describe('Footer', () => {
  it('renders footer columns', () => {
    render(<Footer />);
    for (const title of ['Product', 'Resources', 'Legal']) {
      expect(screen.getByText(title)).toBeInTheDocument();
    }
  });

  it('links to the MIT license', () => {
    render(<Footer />);
    expect(screen.getByRole('link', { name: 'MIT License' })).toHaveAttribute(
      'href',
      'https://github.com/hieudoanm/linguae/blob/master/LICENSE'
    );
  });

  it('renders copyright notice', () => {
    render(<Footer />);
    expect(screen.getByText(/© \d{4} Linguae/)).toBeInTheDocument();
  });
});
