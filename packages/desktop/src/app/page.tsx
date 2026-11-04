'use client';

import DetectionForm from '@/components/DetectionForm';
import Navbar from '@/components/Navbar';
import { NextPage } from 'next';

const Home: NextPage = () => {
  const scrollTo = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <main className="h-screen bg-white px-4 py-8 dark:bg-black">
      <Navbar scrollTo={scrollTo} />
      <DetectionForm />
    </main>
  );
};

export default Home;
