import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Shield, ArrowRight, Play } from 'lucide-react';

export function Hero() {
  const scrollToDemo = () => {
    const element = document.getElementById('demo');
    element?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="relative overflow-hidden px-4 py-24 text-center">
      {/* Animated Background */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-background" />
        <motion.div
          animate={{
            backgroundPosition: ['0% 0%', '100% 100%'],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            repeatType: 'reverse',
          }}
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `
              radial-gradient(circle at 20% 50%, hsl(158 64% 52% / 0.1) 0%, transparent 50%),
              radial-gradient(circle at 80% 20%, hsl(213 94% 68% / 0.1) 0%, transparent 50%),
              radial-gradient(circle at 40% 80%, hsl(258 90% 66% / 0.1) 0%, transparent 50%)
            `,
            backgroundSize: '100% 100%',
          }}
        />
      </div>

      <div className="container mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <div className="mb-6 inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 text-sm font-medium text-primary">
            <Shield className="h-4 w-4" />
            AI-Powered Security
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <h1 className="mb-6 text-5xl font-bold leading-tight tracking-tight text-foreground md:text-7xl">
            <span className="gradient-text">Sentinel X</span>
            <br />
            <span className="text-3xl font-medium text-muted-foreground md:text-4xl">
              Detect threats before they detect you
            </span>
          </h1>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <p className="mb-10 text-xl text-muted-foreground md:text-2xl">
            Real-time intrusion detection combining deterministic automata and neural intelligence.
            <br className="hidden md:block" />
            Cloud-ready, fast, and explainable.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center"
        >
          <Button
            variant="hero"
            size="lg"
            onClick={() => window.location.href = '/dashboard'}
            className="w-full sm:w-auto"
          >
            Dive into Sentinel
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
          <Button
            variant="glass"
            size="lg"
            onClick={scrollToDemo}
            className="w-full sm:w-auto"
          >
            <Play className="mr-2 h-5 w-5" />
            See Demo
          </Button>
        </motion.div>
      </div>
    </section>
  );
}