import type { FC } from 'react';

const STATS = [
  { value: '235k+', label: 'training examples' },
  { value: '1', label: 'language per string' },
  { value: '0', label: 'network calls' },
];

const Hero: FC = () => {
  return (
    <section className="hero bg-base-200 py-16 sm:py-24">
      <div className="hero-content flex-col items-center text-center">
        <div className="mb-4 flex flex-wrap justify-center gap-2">
          <span className="badge badge-primary">Local-first</span>
          <span className="badge badge-primary">No pretrained weights</span>
          <span className="badge badge-primary">MIT licensed</span>
        </div>
        <h1 className="max-w-3xl text-4xl font-bold sm:text-5xl">
          Detect the language of any text, entirely offline.
        </h1>
        <p className="text-base-content/70 mt-6 max-w-2xl text-lg">
          Linguae trains a character-level CNN from scratch with Candle and
          ships a standalone Rust binary. Your text never leaves your machine —
          no cloud APIs, no pretrained weights, no network calls.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-4">
          <a href="#install" className="btn btn-primary">
            Get started
          </a>
          <a href="#demo" className="btn btn-ghost">
            See it in action
          </a>
        </div>
        <div className="mt-12 grid grid-cols-3 gap-8">
          {STATS.map((stat) => (
            <div key={stat.label}>
              <div className="text-3xl font-bold">{stat.value}</div>
              <div className="text-base-content/60 text-sm">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Hero;
