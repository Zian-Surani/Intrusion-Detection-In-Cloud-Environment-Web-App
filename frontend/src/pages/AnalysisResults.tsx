import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { type AnalysisResponse } from '@/lib/api';
import { 
  AlertTriangle, 
  CheckCircle, 
  ArrowLeft,
  Download,
  Shield,
  Brain,
  Activity
} from 'lucide-react';

export function AnalysisResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedLine, setSelectedLine] = useState<number | null>(null);

  const { result, timestamp } = location.state as { 
    result: AnalysisResponse; 
    timestamp: string; 
  } || {};

  useEffect(() => {
    if (!result) {
      navigate('/dashboard');
      return;
    }
  }, [result, navigate]);

  if (!result) {
    return (
      <Layout title="Analysis Results">
        <div className="container mx-auto p-4">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <Activity className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
              <p className="text-lg font-semibold">Loading results...</p>
              <p className="text-sm text-muted-foreground">Please wait while we load your analysis</p>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  const getSeverityColor = (score: number) => {
    if (score >= 0.8) return 'bg-destructive/20 text-destructive border-destructive/40';
    if (score >= 0.5) return 'bg-warning/20 text-warning border-warning/40';
    return 'bg-success/20 text-success border-success/40';
  };

  const getSeverityLevel = (score: number) => {
    if (score >= 0.8) return 'threat';
    if (score >= 0.5) return 'suspicious';
    return 'clean';
  };

  const getHeatmapColor = (score: number) => {
    if (score >= 0.8) return 'bg-destructive';
    if (score >= 0.5) return 'bg-warning';
    return 'bg-success/30';
  };

  const dfaMatches = Object.keys(result.dfa_detail).length;
  const annScore = result.ann_prob || 0;
  const totalLines = result.line_details.length;
  const highRiskLines = result.line_details.filter(line => line.score >= 0.8).length;

  const downloadResults = () => {
    const dataStr = JSON.stringify(result, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = `analysis_results_${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <Layout title="Analysis Results">
      <div className="container mx-auto p-4 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard')}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold">Analysis Results</h1>
            <p className="text-sm text-muted-foreground">
              Analysis completed • {new Date(result.timestamp).toLocaleString()}
            </p>
          </div>
        </div>

        {/* Verdict Banner */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Card className={`ios-card border-2 ${
            result.verdict.includes('ATTACK') 
              ? 'border-destructive/40 bg-destructive/5'
              : result.verdict.includes('SUSPICIOUS')
              ? 'border-warning/40 bg-warning/5'
              : 'border-success/40 bg-success/5'
          }`}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`p-4 rounded-full ${
                    result.verdict.includes('ATTACK') 
                      ? 'bg-destructive/10 text-destructive'
                      : result.verdict.includes('SUSPICIOUS')
                      ? 'bg-warning/10 text-warning'
                      : 'bg-success/10 text-success'
                  }`}>
                    {result.verdict.includes('ATTACK') || result.verdict.includes('SUSPICIOUS') ? (
                      <AlertTriangle className="h-8 w-8" />
                    ) : (
                      <CheckCircle className="h-8 w-8" />
                    )}
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">{result.verdict}</h2>
                    <p className="text-muted-foreground">
                      {highRiskLines} high-risk patterns found in {totalLines} lines
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold">
                    {result.ann_prob ? (result.ann_prob * 100).toFixed(1) + '%' : 'N/A'}
                  </p>
                  <p className="text-sm text-muted-foreground">Confidence</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Analysis Summary */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="ios-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Shield className="h-8 w-8 text-primary" />
                <div>
                  <p className="text-2xl font-bold text-primary">{dfaMatches}</p>
                  <p className="text-sm text-muted-foreground">DFA Matches</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="ios-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Brain className="h-8 w-8 text-accent" />
                <div>
                  <p className="text-2xl font-bold text-accent">
                    {result.method_used === 'ANN' ? '✓' : '✗'}
                  </p>
                  <p className="text-sm text-muted-foreground">Neural Network</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="ios-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Activity className="h-8 w-8 text-success" />
                <div>
                  <p className="text-2xl font-bold text-success">{result.method_used}</p>
                  <p className="text-sm text-muted-foreground">Detection Method</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Plain Explanation */}
        <Card className="ios-card">
          <CardHeader>
            <CardTitle>Analysis Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-muted/30 p-4 rounded-lg">
              <pre className="whitespace-pre-wrap text-sm">{result.plain_explanation}</pre>
            </div>
          </CardContent>
        </Card>

        {/* Heatmap and Line Analysis */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Heatmap Visualization */}
          <Card className="ios-card lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Threat Heatmap
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-10 gap-1">
                {result.line_details.map((line) => (
                  <div
                    key={line.line_number}
                    className={`aspect-square rounded cursor-pointer transition-all hover:scale-110 ${getHeatmapColor(line.score)}`}
                    style={{ 
                      opacity: Math.max(0.2, line.score)
                    }}
                    onClick={() => setSelectedLine(line.line_number)}
                    title={`Line ${line.line_number}: Score ${line.score.toFixed(2)}`}
                  />
                ))}
              </div>
              <div className="mt-4 space-y-2">
                <div className="flex items-center gap-2 text-xs">
                  <div className="w-3 h-3 bg-destructive rounded" />
                  <span>High Risk (≥0.8)</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <div className="w-3 h-3 bg-warning rounded" />
                  <span>Medium Risk (≥0.5)</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <div className="w-3 h-3 bg-success/50 rounded" />
                  <span>Low Risk (&lt;0.5)</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Line-wise Analysis */}
          <Card className="ios-card lg:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Line-wise Analysis</CardTitle>
              <Button variant="outline" size="sm" onClick={downloadResults}>
                <Download className="mr-2 h-4 w-4" />
                Export JSON
              </Button>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {result.line_details.map((line) => (
                  <motion.div
                    key={line.line_number}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: line.line_number * 0.02 }}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      selectedLine === line.line_number ? 'ring-2 ring-primary' : ''
                    } ${getSeverityColor(line.score)}`}
                    onClick={() => setSelectedLine(selectedLine === line.line_number ? null : line.line_number)}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-mono bg-muted px-2 py-1 rounded">
                            Line {line.line_number}
                          </span>
                          {line.flags.map((flag, index) => (
                            <Badge key={index} variant={flag === 'DFA' ? 'secondary' : 'outline'} className="text-xs">
                              {flag}
                            </Badge>
                          ))}
                          <Badge 
                            variant={getSeverityLevel(line.score) === 'threat' ? 'destructive' : 
                                   getSeverityLevel(line.score) === 'suspicious' ? 'secondary' : 'outline'} 
                            className="text-xs"
                          >
                            {getSeverityLevel(line.score).toUpperCase()}
                          </Badge>
                        </div>
                        <p className="font-mono text-sm break-all">{line.content}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs font-semibold">
                          {(line.score * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}