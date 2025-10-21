import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Play, Upload, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const sampleLog = `192.168.1.100 - - [10/Oct/2023:13:55:36 +0000] "GET /admin/login.php HTTP/1.1" 200 2326
192.168.1.100 - - [10/Oct/2023:13:55:37 +0000] "POST /admin/login.php HTTP/1.1" 401 1234
192.168.1.100 - - [10/Oct/2023:13:55:38 +0000] "POST /admin/login.php HTTP/1.1" 401 1234`;

export function DemoSection() {
  const [logText, setLogText] = useState(sampleLog);
  const [useCrawl, setUseCrawl] = useState(false);
  const [useANN, setUseANN] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    // Simulate analysis
    setTimeout(() => {
      setResults({
        verdict: 'THREAT_DETECTED',
        confidence: 0.87,
        threats: [
          { line: 2, type: 'Brute Force', confidence: 0.92 },
          { line: 3, type: 'Failed Login', confidence: 0.85 },
        ]
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  return (
    <section id="demo" className="px-4 py-24">
      <div className="container mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-16 text-center"
        >
          <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
            Try It Live
          </h2>
          <p className="text-lg text-muted-foreground">
            Paste your log data and see Sentinel X in action
          </p>
        </motion.div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Input Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <Card className="ios-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Input Data
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Paste your log data here..."
                  value={logText}
                  onChange={(e) => setLogText(e.target.value)}
                  rows={8}
                  className="resize-none"
                />
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="crawl"
                      checked={useCrawl}
                      onCheckedChange={setUseCrawl}
                    />
                    <label htmlFor="crawl" className="text-sm font-medium">
                      Use Crawl
                    </label>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="ann"
                      checked={useANN}
                      onCheckedChange={setUseANN}
                    />
                    <label htmlFor="ann" className="text-sm font-medium">
                      Use ANN
                    </label>
                  </div>
                </div>

                <Button
                  onClick={runAnalysis}
                  disabled={isAnalyzing}
                  variant="hero"
                  className="w-full"
                >
                  {isAnalyzing ? (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Play className="mr-2 h-4 w-4" />
                    </motion.div>
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              </CardContent>
            </Card>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card className="ios-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Analysis Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                {results ? (
                  <div className="space-y-4">
                    {/* Verdict Banner */}
                    <div className={`rounded-lg p-4 ${
                      results.verdict === 'THREAT_DETECTED' 
                        ? 'bg-destructive/10 border border-destructive/20' 
                        : 'bg-success/10 border border-success/20'
                    }`}>
                      <div className="flex items-center gap-2">
                        {results.verdict === 'THREAT_DETECTED' ? (
                          <XCircle className="h-5 w-5 text-destructive" />
                        ) : (
                          <CheckCircle className="h-5 w-5 text-success" />
                        )}
                        <span className="font-semibold">
                          {results.verdict === 'THREAT_DETECTED' ? 'Threat Detected' : 'No Threats'}
                        </span>
                        <Badge variant="secondary">
                          {(results.confidence * 100).toFixed(0)}% confidence
                        </Badge>
                      </div>
                    </div>

                    {/* Threat List */}
                    <div className="space-y-2">
                      <h4 className="font-medium">Detected Threats:</h4>
                      {results.threats.map((threat: any, index: number) => (
                        <div key={index} className="flex items-center justify-between rounded-lg bg-muted/50 p-3">
                          <div>
                            <span className="font-medium">Line {threat.line}</span>
                            <span className="ml-2 text-sm text-muted-foreground">
                              {threat.type}
                            </span>
                          </div>
                          <Badge variant="outline">
                            {(threat.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex h-32 items-center justify-center text-muted-foreground">
                    Run analysis to see results
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </section>
  );
}