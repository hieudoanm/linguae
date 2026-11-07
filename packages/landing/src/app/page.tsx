import DemoSection from '@/components/DemoSection';
import Features from '@/components/Features';
import Footer from '@/components/Footer';
import Hero from '@/components/Hero';
import InstallSection from '@/components/InstallSection';
import Navbar from '@/components/Navbar';
import Pipeline from '@/components/Pipeline';

const Home = () => {
  return (
    <>
      <Navbar />
      <main className="flex-1">
        <Hero />
        <Features />
        <DemoSection />
        <Pipeline />
        <InstallSection />
      </main>
      <Footer />
    </>
  );
};

export default Home;
