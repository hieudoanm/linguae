import { render, screen } from '@testing-library/react';
import Pipeline from '../Pipeline';

describe('Pipeline', () => {
  it('renders the section heading', () => {
    render(<Pipeline />);
    expect(
      screen.getByRole('heading', { name: 'How it works' })
    ).toBeInTheDocument();
  });

  it('renders all three steps', () => {
    render(<Pipeline />);
    for (const title of ['Prepare data', 'Train & export', 'Run anywhere']) {
      expect(screen.getByRole('heading', { name: title })).toBeInTheDocument();
    }
  });

  it('renders step commands', () => {
    render(<Pipeline />);
    expect(
      screen.getByText('cargo run -p linguae-data -- all')
    ).toBeInTheDocument();
    expect(
      screen.getByText('cargo run -p linguae-train -- train')
    ).toBeInTheDocument();
    expect(
      screen.getByText('linguae "Bonjour tout le monde"')
    ).toBeInTheDocument();
  });
});
