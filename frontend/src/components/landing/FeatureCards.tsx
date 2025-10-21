import React from 'react';
import { motion } from 'framer-motion';
import { Zap, Brain, Cloud, Eye } from 'lucide-react';

const features = [
  {
    icon: Zap,
    title: 'DFA Processing',
    description: 'Rule-based speed for instant red flags with deterministic finite automata.',
    gradient: 'from-yellow-400 to-orange-500',
  },
  {
    icon: Brain,
    title: 'ANN Intelligence',
    description: 'Trained neural models to detect subtle attacks and anomalies.',
    gradient: 'from-purple-400 to-pink-500',
  },
  {
    icon: Cloud,
    title: 'Cloud-Ready',
    description: 'Artifacts, model management, auto-scaling, and JSON exports.',
    gradient: 'from-blue-400 to-cyan-500',
  },
  {
    icon: Eye,
    title: 'Explainable AI',
    description: 'Per-line highlights and plain-language guidance for every detection.',
    gradient: 'from-green-400 to-teal-500',
  },
];

export function FeatureCards() {
  return (
    <section className="px-4 py-24">
      <div className="container mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-16 text-center"
        >
          <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
            Powered by Advanced AI
          </h2>
          <p className="text-lg text-muted-foreground">
            Multiple detection layers working together for comprehensive threat analysis
          </p>
        </motion.div>

        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ y: -5 }}
              className="ios-card group cursor-pointer"
            >
              <div className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-r ${feature.gradient}`}>
                <feature.icon className="h-6 w-6 text-white" />
              </div>
              <h3 className="mb-2 text-xl font-semibold text-foreground">
                {feature.title}
              </h3>
              <p className="text-muted-foreground">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}