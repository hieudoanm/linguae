import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import FeaturesSection from '../FeaturesSection';

describe('FeaturesSection', () => {
  it('renders all three feature cards', () => {
    render(<FeaturesSection />);
    expect(screen.getByText('Local-first')).toBeInTheDocument();
    expect(screen.getByText('Character-level CNN')).toBeInTheDocument();
    expect(screen.getByText('235 languages')).toBeInTheDocument();
  });

  it('renders feature descriptions', () => {
    render(<FeaturesSection />);
    expect(
      screen.getByText(
        'Runs entirely on your machine — no network calls or pretrained weights.'
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'A lightweight character-level convolutional neural network trained from scratch.'
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Supports detection across 235 languages with top-k alternatives.'
      )
    ).toBeInTheDocument();
  });
});
