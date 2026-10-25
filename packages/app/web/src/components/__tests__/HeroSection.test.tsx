import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import HeroSection from '../HeroSection';

describe('HeroSection', () => {
  it('renders the heading and description', () => {
    render(<HeroSection />);
    expect(screen.getByText('Language Detection')).toBeInTheDocument();
    expect(
      screen.getByText('Detect the language of any text with a local character-level model.')
    ).toBeInTheDocument();
  });
});
