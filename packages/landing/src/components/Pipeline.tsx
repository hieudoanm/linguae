import type { FC } from 'react';

const STEPS = [
  {
    step: '01',
    title: 'Prepare data',
    description:
      'linguae-data converts the Tatoeba sentences, WiLI-2018 paragraphs, and per-language word lists into CSVs with one shared label space.',
    command: 'cargo run -p linguae-data -- all',
  },
  {
    step: '02',
    title: 'Train & export',
    description:
      'linguae-train fits the character-level CNN with Candle and exports a portable artifact: config, vocabulary, labels, weights, and metrics.',
    command: 'cargo run -p linguae-train -- train',
  },
  {
    step: '03',
    title: 'Run anywhere',
    description:
      'The linguae binary or the FastAPI server loads the artifact and predicts one ISO 639-3 code per string — fully offline.',
    command: 'linguae "Bonjour tout le monde"',
  },
];

const Pipeline: FC = () => {
  return (
    <section id="how-it-works" className="py-20">
      <div className="mx-auto max-w-6xl px-4">
        <div className="text-center">
          <h2 className="text-3xl font-bold">How it works</h2>
          <p className="text-base-content/70 mt-4">
            From raw corpora to a standalone binary in three reproducible steps.
          </p>
        </div>
        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {STEPS.map((item) => (
            <div key={item.step} className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <div className="flex items-center gap-3">
                  <span className="badge badge-primary badge-lg">
                    {item.step}
                  </span>
                  <h3 className="card-title">{item.title}</h3>
                </div>
                <p className="text-base-content/70">{item.description}</p>
                <code className="rounded-box bg-base-300 mt-2 block px-3 py-2 font-mono text-sm">
                  {item.command}
                </code>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Pipeline;
