import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import Providers from '../providers';

describe('Providers', () => {
  it('renders children', () => {
    const { container } = render(<Providers><div>child</div></Providers>);
    expect(container.textContent).toContain('child');
  });
});
