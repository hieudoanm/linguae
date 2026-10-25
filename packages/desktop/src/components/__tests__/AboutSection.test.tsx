import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import AboutSection from '../AboutSection';

describe('AboutSection', () => {
  it('renders the about heading and description', () => {
    render(<AboutSection />);
    expect(screen.getByText('About Linguae')).toBeInTheDocument();
    expect(
      screen.getByText(
        /Linguae is a local-first language identification project/
      )
    ).toBeInTheDocument();
  });
});
