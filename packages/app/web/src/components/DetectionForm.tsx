import { useMutation } from '@tanstack/react-query';
import { useState, type FC } from 'react';

import type { Prediction } from './types';

const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

const DetectionForm: FC = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState<Prediction | null>(null);

  const mutation = useMutation({
    mutationFn: async (inputText: string) => {
      if (isTauri) {
        const { invoke } = await import('@tauri-apps/api/core');
        return invoke<Prediction>('detect', { text: inputText, topK: 3 });
      }
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText, top_k: 3 }),
      });
      if (!response.ok) {
        throw new Error('Failed to detect language');
      }
      return response.json() as Promise<Prediction>;
    },
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim()) {
      mutation.mutate(text);
    }
  };

  return (
    <section
      id="detect"
      className="flex h-full items-center justify-center bg-white py-12 dark:bg-gray-900">
      <div className="mx-auto w-full max-w-4xl px-4">
        <div className="mb-8 text-center">
          <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-white">
            Try It
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Enter text to detect its language.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mb-6">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to detect language..."
            className="h-32 w-full rounded-lg border border-gray-300 p-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:outline-none dark:border-gray-600 dark:text-white"
            required
          />
          <button
            type="submit"
            disabled={mutation.isPending || !text.trim()}
            className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-400">
            {mutation.isPending ? 'Detecting...' : 'Submit'}
          </button>
        </form>

        {mutation.isError && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            Error detecting language. Make sure the server is running.
          </div>
        )}

        {result && (
          <div className="rounded-lg border border-gray-200 p-4 shadow-sm dark:border-gray-700">
            <h2 className="mb-3 text-xl font-semibold text-gray-900 dark:text-white">
              Result
            </h2>
            <p className="mb-2 text-gray-700">
              <span className="font-medium">Language:</span>{' '}
              {result.language_name
                ? `${result.language} (${result.language_name})`
                : result.language}
            </p>
            <p className="mb-3 text-gray-700">
              <span className="font-medium">Confidence:</span>{' '}
              {(result.confidence * 100).toFixed(2)}%
            </p>
            {result.top_k && result.top_k.length > 0 && (
              <div>
                <p className="mb-2 font-medium text-gray-700">
                  Top alternatives:
                </p>
                <ul className="list-inside list-disc space-y-1">
                  {result.top_k.map((item, idx) => (
                    <li key={idx} className="text-gray-600 dark:text-gray-300">
                      {item.language}: {(item.score * 100).toFixed(2)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default DetectionForm;
