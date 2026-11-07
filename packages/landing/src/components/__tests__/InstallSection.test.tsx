import { render, screen } from '@testing-library/react';
import InstallSection from '../InstallSection';

describe('InstallSection', () => {
  it('renders the section heading', () => {
    render(<InstallSection />);
    expect(
      screen.getByRole('heading', { name: 'Get started' })
    ).toBeInTheDocument();
  });

  it('renders prerequisites', () => {
    render(<InstallSection />);
    expect(screen.getByText('Rust 1.85+')).toBeInTheDocument();
    expect(
      screen.queryByText('Python 3.14.8 + uv (server only)')
    ).not.toBeInTheDocument();
  });

  it('renders install commands', () => {
    render(<InstallSection />);
    expect(
      screen.getByText('cd crates && cargo build --release -p linguae')
    ).toBeInTheDocument();
    expect(screen.getByText('fra')).toBeInTheDocument();
  });

  it('renders action links', () => {
    render(<InstallSection />);
    expect(
      screen.getByRole('link', { name: /view on github/i })
    ).toHaveAttribute('href', 'https://github.com/hieudoanm/linguae');
    expect(
      screen.getByRole('link', { name: /documentation/i })
    ).toHaveAttribute(
      'href',
      'https://github.com/hieudoanm/linguae/blob/master/AGENTS.md'
    );
  });
});
