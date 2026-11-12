import type { Metadata, NextPage } from 'next';
import type { FC } from 'react';
import CTA from '@/components/molecules/CTA';
import Features, { type Feature } from '@/components/molecules/Features';
import Hero from '@/components/molecules/Hero';
import SectionHead from '@/components/molecules/SectionHead';
import Terminal from '@/components/molecules/Terminal';
import {
  LuChartBar,
  LuBrainCircuit,
  LuFileInput,
  LuLock,
  LuZap,
} from 'react-icons/lu';

export const metadata: Metadata = {
  alternates: { canonical: '/' },
};

const GITHUB = 'https://github.com/hieudoanm/linguae';

const FEATURES: Feature[] = [
  {
    icon: LuLock,
    title: 'Runs fully offline',
    body: 'Inference happens in-process inside the Rust binary. No sockets, no telemetry, no accounts — safe for air-gapped machines.',
  },
  {
    icon: LuBrainCircuit,
    title: 'Trained from scratch',
    body: 'A character-level CNN trained with Candle on Tatoeba, WiLI-2018, and a curated word dataset. No pretrained weights to trust or audit.',
  },
  {
    icon: LuZap,
    title: 'Fast and small',
    body: 'Character-level encoding keeps the model compact. Detection of a sentence completes in milliseconds on CPU.',
  },
  {
    icon: LuFileInput,
    title: 'Flexible input',
    body: 'Pass text as an argument, a file with --file, or pipe it through stdin — it composes with standard Unix tools.',
  },
  {
    icon: LuChartBar,
    title: 'Confidence and JSON',
    body: '--confidence prints per-language scores and --format json emits structured output for scripts.',
  },
  {
    icon: LuFileInput,
    title: 'Batch processing',
    body: 'Detect languages across whole files or streams line by line — built for corpus-scale workloads.',
  },
];

const HERO_LINES = [
  '$ linguae "Bonjour tout le monde"',
  'fra',
  '$ linguae --format json "Это тест"',
  '{',
  '  "language": "rus",',
  '  "confidence": 0.98',
  '}',
];

const DEMO_TERMINALS: { caption: string; lines: string[]; wide?: boolean }[] = [
  {
    caption: 'French',
    lines: ['$ linguae "Bonjour tout le monde, comment allez-vous ?"', 'fra'],
  },
  {
    caption: 'JSON output',
    lines: [
      '$ linguae --format json "Это тестовый пример текста"',
      '{',
      '  "language": "rus",',
      '  "confidence": 0.98',
      '}',
    ],
  },
  {
    caption: 'Batch from file',
    lines: ['$ linguae --file texts.txt', 'fra', 'eng', 'rus', 'deu'],
  },
];

const STEPS = [
  {
    title: 'Normalize',
    body: 'Text is lowercased, stripped of diacritics variation, and mapped onto a fixed character vocabulary.',
    snippet: '"Bonjour" → char indices',
  },
  {
    title: 'Encode',
    body: 'Characters become indices fed into a CharCNN — small kernels capture local n-gram patterns, global pooling summarizes the text.',
    snippet: 'CharCNN · Candle',
  },
  {
    title: 'Classify',
    body: 'The model emits a probability per language; the CLI prints the top ISO 639-3 code, or scores with --confidence.',
    snippet: 'fra 0.98 → "fra"',
  },
];

const INSTALL_LINES = [
  '$ git clone https://github.com/hieudoanm/linguae',
  '$ cd linguae',
  '$ cargo build --release -p linguae',
  '$ ./target/release/linguae "Bonjour tout le monde"',
  'fra',
];

const StepCard: FC<{ step: (typeof STEPS)[number]; index: number }> = ({
  step,
  index,
}) => (
  <article className="card border-base-300 bg-base-100 rounded-2xl border p-6">
    <span className="text-accent mb-2 block font-mono text-sm font-semibold">
      {String(index + 1).padStart(2, '0')}
    </span>
    <h3 className="text-base-content m-0 mb-2 text-lg font-bold">
      {step.title}
    </h3>
    <p className="text-base-content/60 text-sm">{step.body}</p>
    <code className="border-base-300 bg-base-200 text-base-content/60 mt-4 block overflow-x-auto rounded-lg border px-3 py-2 font-mono text-xs">
      {step.snippet}
    </code>
  </article>
);

const HomePage: NextPage = () => (
  <main>
    <Hero
      title="Detect the language of any text, entirely offline."
      description="Linguae trains a character-level CNN from scratch with Candle and ships a standalone Rust binary. Your text never leaves your machine — no cloud APIs, no pretrained weights, no network calls."
      primary={{ label: 'Get started', href: '#install' }}
      secondary={{ label: 'See it in action', href: '#demo' }}
      stats={[
        { value: '235k+', label: 'training examples' },
        { value: '200+', label: 'languages' },
        { value: '0', label: 'network calls' },
      ]}>
      <Terminal caption="Detect a language" lines={HERO_LINES} />
    </Hero>

    <Features
      title="Built for privacy and simplicity"
      description="A single static binary that does one thing well — identify languages — without phoning home."
      features={FEATURES}
    />

    <section id="demo" className="bg-base-200 py-20">
      <div className="mx-auto w-full max-w-6xl px-5">
        <SectionHead
          title="See it in action"
          description="Real output from the CLI — no mockups."
        />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {DEMO_TERMINALS.map((t) => (
            <Terminal
              key={t.caption}
              caption={t.caption}
              lines={t.lines}
              className={t.wide ? 'lg:col-span-2' : ''}
            />
          ))}
        </div>
      </div>
    </section>

    <section id="how-it-works" className="py-20">
      <div className="mx-auto w-full max-w-6xl px-5">
        <SectionHead
          title="How it works"
          description="From raw text to a language code in three steps."
        />
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {STEPS.map((step, i) => (
            <StepCard key={step.title} step={step} index={i} />
          ))}
        </div>
      </div>
    </section>

    <section id="install" className="bg-base-200 py-20">
      <div className="mx-auto w-full max-w-3xl px-5">
        <SectionHead
          title="Install and run"
          description="Requires Rust 1.85+. No Python runtime needed for the CLI."
        />
        <Terminal caption="Build from source" lines={INSTALL_LINES} />
        <p className="text-base-content/60 mt-6 text-center text-sm">
          See the{' '}
          <a
            href={`${GITHUB}#readme`}
            target="_blank"
            rel="noreferrer"
            className="text-primary font-semibold">
            README
          </a>{' '}
          for dataset and server details.
        </p>
      </div>
    </section>

    <CTA
      title="Ready to identify languages?"
      description="One binary, fully offline — build linguae and start detecting."
      primary={{ label: 'View on GitHub', href: GITHUB, external: true }}
      secondary={{
        label: 'Read the README',
        href: `${GITHUB}#readme`,
        external: true,
      }}
    />
  </main>
);

export default HomePage;
