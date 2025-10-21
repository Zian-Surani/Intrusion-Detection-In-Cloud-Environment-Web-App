import React from 'react';
import { Header } from './Header';
import { BottomNav } from './BottomNav';

interface LayoutProps {
  children: React.ReactNode;
  title?: string;
}

export function Layout({ children, title }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-background">
      <Header title={title} />
      <main className="pb-20 md:pb-6">
        {children}
      </main>
      <BottomNav />
    </div>
  );
}