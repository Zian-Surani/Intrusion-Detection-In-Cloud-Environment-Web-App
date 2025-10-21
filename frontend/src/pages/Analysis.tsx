import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { 
  Upload, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  Download,
  Eye
} from 'lucide-react';

export function Analysis() {
  const [analysisMode, setAnalysisMode] = useState<'upload' | 'url'>('upload');

  return (
    <Layout title="Analysis">
      <div className="container mx-auto p-4 space-y-6">
        {/* Analysis Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="ios-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                New Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Mode Toggle */}
              <div className="flex gap-2">
                <Button
                  variant={analysisMode === 'upload' ? 'hero' : 'outline'}
                  size="sm"
                  onClick={() => setAnalysisMode('upload')}
                >
                  <FileText className="mr-2 h-4 w-4" />
                  Upload Logs
                </Button>
                <Button
                  variant={analysisMode === 'url' ? 'hero' : 'outline'}
                  size="sm"
                  onClick={() => setAnalysisMode('url')}
                >
                  <Eye className="mr-2 h-4 w-4" />
                  Enter URL
                </Button>
              </div>

              {analysisMode === 'upload' ? (
                <div className="border-2 border-dashed border-border rounded-xl p-8 text-center">
                  <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold mb-2">Drop log files here</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    or click to browse files
                  </p>
                  <Button variant="outline">Choose Files</Button>
                </div>
              ) : (
                <Textarea
                  placeholder="Enter URL to analyze..."
                  rows={3}
                />
              )}

              {/* Options */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Switch id="crawl" />
                  <label htmlFor="crawl" className="text-sm font-medium">
                    Enable Crawling
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch id="ann" defaultChecked />
                  <label htmlFor="ann" className="text-sm font-medium">
                    Use Neural Network
                  </label>
                </div>
              </div>

              <Button variant="hero" className="w-full">
                Start Analysis
              </Button>
            </CardContent>
          </Card>
        </motion.div>

        {/* Recent Results */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="ios-card">
            <CardHeader>
              <CardTitle>Recent Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    id: 'analysis-001',
                    timestamp: '2023-10-10 15:30:22',
                    source: 'security_logs.txt',
                    verdict: 'THREAT_DETECTED',
                    threats: 5,
                    confidence: 0.94
                  },
                  {
                    id: 'analysis-002',
                    timestamp: '2023-10-10 14:15:18',
                    source: 'apache_access.log',
                    verdict: 'CLEAN',
                    threats: 0,
                    confidence: 0.98
                  }
                ].map((result) => (
                  <div
                    key={result.id}
                    className="flex items-center justify-between rounded-lg bg-muted/30 p-4"
                  >
                    <div className="flex items-center gap-4">
                      <div className={`p-2 rounded-full ${
                        result.verdict === 'THREAT_DETECTED'
                          ? 'bg-destructive/10 text-destructive'
                          : 'bg-success/10 text-success'
                      }`}>
                        {result.verdict === 'THREAT_DETECTED' ? (
                          <AlertTriangle className="h-4 w-4" />
                        ) : (
                          <CheckCircle className="h-4 w-4" />
                        )}
                      </div>
                      <div>
                        <p className="font-medium">{result.source}</p>
                        <p className="text-sm text-muted-foreground">
                          {result.timestamp}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={result.verdict === 'THREAT_DETECTED' ? 'destructive' : 'outline'}>
                        {result.threats} threats
                      </Badge>
                      <Button variant="outline" size="sm">
                        <Download className="mr-2 h-4 w-4" />
                        Export
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </Layout>
  );
}