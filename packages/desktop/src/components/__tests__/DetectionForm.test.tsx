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

const renderWithProviders = (ui: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>{ui}</ThemeProvider>
    </QueryClientProvider>
  );
};

describe('DetectionForm', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('renders the textarea and submit button', () => {
    renderWithProviders(<DetectionForm />);
    expect(
      screen.getByPlaceholderText('Enter text to detect language...')
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument();
  });

  it('shows Detecting... while the mutation is pending', async () => {
    let resolveInvoke: (value: unknown) => void = () => {};
    const invokeMock = vi.fn(
      () =>
        new Promise<unknown>((resolve) => {
          resolveInvoke = resolve;
        })
    );
    vi.doMock('@tauri-apps/api/core', () => ({ invoke: invokeMock }));
    const { default: DetectionFormComponent } = await import('../DetectionForm');

    renderWithProviders(<DetectionFormComponent />);
    const textarea = screen.getByPlaceholderText(
      'Enter text to detect language...'
    );
    fireEvent.change(textarea, { target: { value: 'Bonjour' } });
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    });
    await vi.waitFor(() => {
      expect(screen.getByText('Detecting...')).toBeInTheDocument();
    });
    resolveInvoke({
      text: 'Bonjour',
      language: 'fra',
      confidence: 0.95,
      top_k: [{ language: 'fra', score: 0.95 }],
    });
  });

  it('displays the detected language and confidence on success', async () => {
    const invokeMock = vi.fn(
      () =>
        Promise.resolve({
          text: 'Bonjour',
          language: 'fra',
          confidence: 0.95,
          top_k: [
            { language: 'fra', score: 0.95 },
            { language: 'eng', score: 0.03 },
          ],
        })
    );
    vi.doMock('@tauri-apps/api/core', () => ({ invoke: invokeMock }));
    const { default: DetectionFormComponent } = await import('../DetectionForm');

    renderWithProviders(<DetectionFormComponent />);
    const textarea = screen.getByPlaceholderText(
      'Enter text to detect language...'
    );
    fireEvent.change(textarea, { target: { value: 'Bonjour' } });
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    });
    await vi.waitFor(() => {
      expect(screen.getByText('fra')).toBeInTheDocument();
    });
    expect(screen.getByText('95.00%')).toBeInTheDocument();
  });

  it('shows an error message when the request fails', async () => {
    const invokeMock = vi.fn(() => Promise.reject(new Error('boom')));
    vi.doMock('@tauri-apps/api/core', () => ({ invoke: invokeMock }));
    const { default: DetectionFormComponent } = await import('../DetectionForm');

    renderWithProviders(<DetectionFormComponent />);
    const textarea = screen.getByPlaceholderText(
      'Enter text to detect language...'
    );
    fireEvent.change(textarea, { target: { value: 'Hello' } });
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    });
    await vi.waitFor(() => {
      expect(
        screen.getByText(
          'Error detecting language. Make sure the Tauri runtime is available.'
        )
      ).toBeInTheDocument();
    });
  });
});