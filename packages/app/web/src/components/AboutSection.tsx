const AboutSection = () => {
  return (
    <section id="about" className="py-12 text-center">
      <div className="max-w-2xl mx-auto px-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">About Linguae</h2>
        <p className="text-gray-600 dark:text-gray-300 max-w-lg mx-auto">
          Linguae is a local-first language identification project that trains a character-level CNN
          from scratch and ships a standalone CLI binary and HTTP API for detecting the language of
          text.
        </p>
      </div>
    </section>
  );
}

export default AboutSection;
