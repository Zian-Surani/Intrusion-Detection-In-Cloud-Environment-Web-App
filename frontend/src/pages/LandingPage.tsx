import React from 'react';
import { Hero } from '@/components/landing/Hero';
import { FeatureCards } from '@/components/landing/FeatureCards';
import { DemoSection } from '@/components/demo/DemoSection';
import { Header } from '@/components/layout/Header';

export function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-background">
      <Header title="Welcome to Sentinel X" />
      <Hero />
      <FeatureCards />
      <DemoSection />
      
      {/* Footer */}
      <footer className="border-t border-border/40 px-4 py-12">
        <div className="container mx-auto text-center">
          <p className="text-muted-foreground">
            Made with ❤️ for security students & cloud engineers
          </p>
        </div>
      </footer>
    </div>
  );
}