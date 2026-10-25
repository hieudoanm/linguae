import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '../ThemeProvider';
import DetectionForm from '../DetectionForm';

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

const renderWithProviders = (ui: React.ReactElement)  => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>{ui}</ThemeProvider>
    </QueryClientProvider>
  );
}

describe('DetectionForm', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders the textarea and submit button', () => {
    renderWithProviders(<DetectionForm />);
    expect(screen.getByPlaceholderText('Enter text to detect language...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument();
  });

  it('shows Detecting... while the mutation is pending', async () => {
    let resolveFetch: (value: Response) => void = () => {};
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            resolveFetch = resolve;
          })
      )
    );
    renderWithProviders(<DetectionForm />);
    const textarea = screen.getByPlaceholderText('Enter text to detect language...');
    fireEvent.change(textarea, { target: { value: 'Bonjour' } });
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    });
    await vi.waitFor(() => {
      expect(screen.getByText('Detecting...')).toBeInTheDocument();
    });
    resolveFetch(new Response(JSON.stringify({ ok: true }), { status: 200, statusText: 'OK' }));
  });

  it('displays the detected language and confidence on success', async () => {
    const response = {
      ok: true,
      json: async () => ({
        text: 'Bonjour',
        language: 'fra',
        language_name: 'French',
        confidence: 0.95,
        top_k: [
          { language: 'fra', confidence: 0.95 },
          { language: 'eng', confidence: 0.03 },
        ],
      }),
    } as unknown as Response;
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(response))
    );
    renderWithProviders(<DetectionForm />);
    const textarea = screen.getByPlaceholderText('Enter text to detect language...');
    fireEvent.change(textarea, { target: { value: 'Bonjour' } });
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    });
    await vi.waitFor(() => {
      expect(screen.getByText('fra (French)')).toBeInTheDocument();
    });
    expect(screen.getByText('95.00%')).toBeInTheDocument();
  });

  it('shows an error message when the request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve({ ok: false } as Response))
    );
    renderWithProviders(<DetectionForm />);
    const textarea = screen.getByPlaceholderText('Enter text to detect language...');
    fireEvent.change(textarea, { target: { value: 'Hello' } });
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    });
    await vi.waitFor(() => {
      expect(
        screen.getByText('Error detecting language. Make sure the server is running.')
      ).toBeInTheDocument();
    });
  });
});
