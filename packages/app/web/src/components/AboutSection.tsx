import { type FC } from 'react';

const AboutSection: FC = () => {
  return (
    <section id="about" className="py-12 text-center">
      <div className="mx-auto max-w-2xl px-4">
        <h2 className="mb-4 text-2xl font-bold text-gray-900 dark:text-white">
          About Linguae
        </h2>
        <p className="mx-auto max-w-lg text-gray-600 dark:text-gray-300">
          Linguae is a local-first language identification project that trains a
          character-level CNN from scratch and ships a standalone CLI binary and
          HTTP API for detecting the language of text.
        </p>
      </div>
    </section>
  );
};

export default AboutSection;
