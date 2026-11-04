import { type FC } from 'react';

const HeroSection: FC = () => {
  return (
    <section id="hero" className="pt-32 pb-12 text-center">
      <h1 className="mb-4 text-4xl font-bold text-gray-900 dark:text-white">
        Language Detection
      </h1>
      <p className="mx-auto max-w-2xl text-lg text-gray-600 dark:text-gray-300">
        Detect the language of any text with a local character-level model.
      </p>
    </section>
  );
};

export default HeroSection;
