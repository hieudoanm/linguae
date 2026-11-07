import { fireEvent, render, screen } from '@testing-library/react';
import ErrorPage from '../error';

describe('error.tsx', () => {
  it('renders the error message', () => {
    render(<ErrorPage error={new Error('boom')} reset={() => {}} />);
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('boom')).toBeInTheDocument();
  });

  it('calls reset when Try again is clicked', () => {
    const reset = vi.fn();
    render(<ErrorPage error={new Error('boom')} reset={reset} />);
    fireEvent.click(screen.getByRole('button', { name: /try again/i }));
    expect(reset).toHaveBeenCalledTimes(1);
  });
});
