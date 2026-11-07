import { render, screen } from '@testing-library/react';
import Navbar from '../Navbar';

describe('Navbar', () => {
  it('renders the brand linking home', () => {
    render(<Navbar />);
    const brand = screen.getAllByRole('link', { name: /linguae/i })[0];
    expect(brand).toHaveAttribute('href', '/');
  });

  it('renders navigation links', () => {
    render(<Navbar />);
    for (const label of ['Features', 'Demo', 'How it works', 'Install']) {
      expect(
        screen.getAllByRole('link', { name: label }).length
      ).toBeGreaterThan(0);
    }
  });

  it('links to the GitHub repository', () => {
    render(<Navbar />);
    const links = screen.getAllByRole('link', { name: 'GitHub' });
    expect(links.length).toBeGreaterThan(0);
    expect(links[0]).toHaveAttribute(
      'href',
      'https://github.com/hieudoanm/linguae'
    );
  });

  it('renders the theme toggle', () => {
    render(<Navbar />);
    expect(
      screen.getByRole('button', { name: /toggle color theme/i })
    ).toBeInTheDocument();
  });
});
