'use client';

import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import FeaturesSection from '@/components/FeaturesSection';
import DetectionForm from '@/components/DetectionForm';
import AboutSection from '@/components/AboutSection';
import Footer from '@/components/Footer';

const Home = () => {
  const scrollTo = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-950 py-8 px-4">
      <Navbar scrollTo={scrollTo} />
      <HeroSection />
      <FeaturesSection />
      <DetectionForm />
      <AboutSection />
      <Footer />
    </main>
  );
}

export default Home;
