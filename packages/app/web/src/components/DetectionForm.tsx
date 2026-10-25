import { useMutation } from '@tanstack/react-query';
import { useState } from 'react';

import type { Prediction } from './types';

export default function DetectionForm() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<Prediction | null>(null);

  const mutation = useMutation({
    mutationFn: async (inputText: string) => {
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
    <section id="detect" className="py-12 bg-white">
      <div className="max-w-2xl mx-auto px-4">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Try It</h2>
          <p className="text-gray-600">Enter text to detect its language.</p>
        </div>

        <form onSubmit={handleSubmit} className="mb-6">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to detect language..."
            className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
            required
          />
          <button
            type="submit"
            disabled={mutation.isPending || !text.trim()}
            className="mt-3 w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {mutation.isPending ? 'Detecting...' : 'Submit'}
          </button>
        </form>

        {mutation.isError && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 mb-4">
            Error detecting language. Make sure the server is running.
          </div>
        )}

        {result && (
          <div className="p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
            <h2 className="text-xl font-semibold mb-3 text-gray-900">Result</h2>
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
                <p className="font-medium mb-2 text-gray-700">Top alternatives:</p>
                <ul className="space-y-1">
                  {result.top_k.map((item, idx) => (
                    <li key={idx} className="text-gray-600">
                      {item.language}: {(item.confidence * 100).toFixed(2)}%
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
}
