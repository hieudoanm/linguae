'use client';

import { ThemeProvider } from '@/components/ThemeProvider';

const Providers = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
);

export default Providers;
