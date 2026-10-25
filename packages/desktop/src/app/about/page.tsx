'use client';

import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import FeaturesSection from '@/components/FeaturesSection';
import AboutSection from '@/components/AboutSection';
import Footer from '@/components/Footer';
import { NextPage } from 'next';

const AboutPage: NextPage = () => {
  const scrollTo = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 px-4 py-8 dark:bg-gray-950">
      <Navbar scrollTo={scrollTo} />
      <HeroSection />
      <FeaturesSection />
      <AboutSection />
      <Footer />
    </main>
  );
};

export default AboutPage;
