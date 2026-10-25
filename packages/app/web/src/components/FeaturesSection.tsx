interface Feature {
  title: string;
  description: string;
}

const FEATURES: Feature[] = [
  {
    title: 'Local-first',
    description: 'Runs entirely on your machine — no network calls or pretrained weights.',
  },
  {
    title: 'Character-level CNN',
    description: 'A lightweight character-level convolutional neural network trained from scratch.',
  },
  {
    title: '235 languages',
    description: 'Supports detection across 235 languages with top-k alternatives.',
  },
];

const FeaturesSection = () => {
  return (
    <section id="features" className="py-12">
      <div className="max-w-2xl mx-auto px-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-8 text-center">Features</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {FEATURES.map((feature) => (
            <div key={feature.title} className="p-6 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default FeaturesSection;
