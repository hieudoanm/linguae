import { type FC } from 'react';

interface Feature {
  title: string;
  description: string;
}

const FEATURES: Feature[] = [
  {
    title: 'Local-first',
    description:
      'Runs entirely on your machine — no network calls or pretrained weights.',
  },
  {
    title: 'Character-level CNN',
    description:
      'A lightweight character-level convolutional neural network trained from scratch.',
  },
  {
    title: '235 languages',
    description:
      'Supports detection across 235 languages with top-k alternatives.',
  },
];

const FeaturesSection: FC = () => {
  return (
    <section id="features" className="py-12">
      <div className="mx-auto w-full max-w-4xl px-4">
        <h2 className="mb-8 text-center text-2xl font-bold text-gray-900 dark:text-white">
          Features
        </h2>
        <div className="grid gap-6 md:grid-cols-3">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-900">
              <h3 className="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
