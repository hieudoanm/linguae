import type { FC } from 'react';
import TerminalWindow, { type TerminalLine } from './TerminalWindow';

const EXAMPLES: { title: string; lines: TerminalLine[] }[] = [
  {
    title: 'French sentence',
    lines: [
      {
        prefix: '$',
        code: 'linguae "Bonjour tout le monde, comment allez-vous aujourd\'hui ?"',
      },
      { code: 'fra', output: true },
    ],
  },
  {
    title: 'Confidence score',
    lines: [
      {
        prefix: '$',
        code: 'linguae --confidence "Hello world, this is a longer English text"',
      },
      { code: 'eng\t0.6114', output: true },
    ],
  },
  {
    title: 'JSON output',
    lines: [
      {
        prefix: '$',
        code: 'linguae --json "Привет, мир, это русский текст"',
      },
      {
        code: '{"text":"...","language":"rus","confidence":0.87}',
        output: true,
      },
    ],
  },
];

const DemoSection: FC = () => {
  return (
    <section id="demo" className="bg-base-200 py-20">
      <div className="mx-auto max-w-6xl px-4">
        <div className="text-center">
          <h2 className="text-3xl font-bold">See it in action</h2>
          <p className="text-base-content/70 mt-4">
            One language label per string — short sentences, long paragraphs, or
            single words.
          </p>
        </div>
        <div className="mt-12 grid gap-6 lg:grid-cols-3">
          {EXAMPLES.map((example) => (
            <TerminalWindow
              key={example.title}
              title={example.title}
              lines={example.lines}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default DemoSection;
