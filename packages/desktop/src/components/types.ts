export interface Prediction {
  text: string;
  language: string;
  confidence: number;
  top_k?: Array<{ language: string; score: number }>;
}

export interface NavLink {
  href: string;
  label: string;
}

export const NAV_LINKS: NavLink[] = [{ href: '/about', label: 'About' }];
