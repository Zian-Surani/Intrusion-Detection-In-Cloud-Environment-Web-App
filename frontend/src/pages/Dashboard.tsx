import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { useToast } from '@/hooks/use-toast';
import { apiService, type RecentAnalysis } from '@/lib/api';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Activity,
  Clock,
  Plus,
  Upload,
  Monitor,
  Loader2
} from 'lucide-react';

const stats = [
  {
    title: 'Active Threats',
    value: '3',
    change: '+12%',
    icon: AlertTriangle,
    color: 'text-destructive',
    bgColor: 'bg-destructive/10',
  },
  {
    title: 'Blocked Attacks',
    value: '127',
    change: '+23%',
    icon: Shield,
    color: 'text-success',
    bgColor: 'bg-success/10',
  },
  {
    title: 'System Health',
    value: '98%',
    change: '+2%',
    icon: Activity,
    color: 'text-primary',
    bgColor: 'bg-primary/10',
  },
  {
    title: 'Response Time',
    value: '0.3s',
    change: '-15%',
    icon: Clock,
    color: 'text-accent',
    bgColor: 'bg-accent/10',
  },
];

export function Dashboard() {
  const [logText, setLogText] = useState('');
  const [urlText, setUrlText] = useState('');
  const [analysisType, setAnalysisType] = useState<'upload' | 'url'>('upload');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [useNeuralNetwork, setUseNeuralNetwork] = useState(true);
  const [crawlEnabled, setCrawlEnabled] = useState(false);
  const [maxPages, setMaxPages] = useState(8);
  const [recentAnalyses, setRecentAnalyses] = useState<RecentAnalysis[]>([]);
  const [serverStatus, setServerStatus] = useState<string>('unknown');
  const navigate = useNavigate();
  const { toast } = useToast();

  // Fetch recent analyses and server status on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [recentData, serverInfo] = await Promise.all([
          apiService.getRecentAnalyses(),
          apiService.getServerInfo()
        ]);
        setRecentAnalyses(recentData.analyses);
        setServerStatus('connected');
      } catch (error) {
        console.error('Failed to fetch data:', error);
        setServerStatus('disconnected');
        toast({
          title: "Connection Error",
          description: "Failed to connect to backend server. Make sure it's running on port 8000.",
          variant: "destructive",
        });
      }
    };

    fetchData();
  }, [toast]);

  const handleRunAnalysis = async () => {
    if ((!logText.trim() && analysisType === 'upload') || (!urlText.trim() && analysisType === 'url')) {
      toast({
        title: "Invalid Input",
        description: "Please provide input data for analysis.",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      let result;
      if (analysisType === 'upload') {
        result = await apiService.analyzeLogs({
          logs: logText,
          use_neural_network: useNeuralNetwork,
          analysis_type: 'upload'
        });
      } else {
        result = await apiService.analyzeURL({
          url: urlText,
          crawl_enabled: crawlEnabled,
          max_pages: maxPages,
          use_neural_network: useNeuralNetwork
        });
      }

      // Navigate to results page with analysis data
      navigate('/analysis-results', { 
        state: { 
          result,
          timestamp: new Date().toISOString()
        } 
      });
      setIsDialogOpen(false);
      setLogText('');
      setUrlText('');
      
      toast({
        title: "Analysis Complete",
        description: `Analysis completed using ${result.method_used}`,
      });
      
    } catch (error) {
      console.error('Analysis failed:', error);
      toast({
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const openAnalysisDialog = (type: 'upload' | 'url') => {
    setAnalysisType(type);
    setIsDialogOpen(true);
  };

  return (
    <Layout title="Dashboard">
      <div className="container mx-auto p-4 space-y-6">
        {/* Stats Grid */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="ios-card">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">
                        {stat.title}
                      </p>
                      <div className="flex items-center gap-2">
                        <p className="text-2xl font-bold">{stat.value}</p>
                        <Badge variant="outline" className="text-xs">
                          {stat.change}
                        </Badge>
                      </div>
                    </div>
                    <div className={`${stat.bgColor} ${stat.color} p-3 rounded-lg`}>
                      <stat.icon className="h-6 w-6" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Server Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Card className="ios-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  serverStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-sm">
                  Backend Server: {serverStatus === 'connected' ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <Card className="ios-card">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <Button 
                  variant="glass" 
                  className="h-20 flex-col"
                  onClick={() => openAnalysisDialog('upload')}
                  disabled={serverStatus !== 'connected'}
                >
                  <Upload className="mb-2 h-6 w-6" />
                  Upload Logs
                </Button>
                <Button 
                  variant="glass" 
                  className="h-20 flex-col"
                  onClick={() => openAnalysisDialog('url')}
                  disabled={serverStatus !== 'connected'}
                >
                  <Monitor className="mb-2 h-6 w-6" />
                  Analyze URL
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Analysis Dialog */}
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogContent className="ios-card max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                {analysisType === 'upload' ? (
                  <>
                    <Upload className="h-5 w-5" />
                    Upload Logs for Analysis
                  </>
                ) : (
                  <>
                    <Monitor className="h-5 w-5" />
                    Analyze URL
                  </>
                )}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              {analysisType === 'upload' ? (
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-2 block">
                    Paste your log data below:
                  </label>
                  <Textarea
                    value={logText}
                    onChange={(e) => setLogText(e.target.value)}
                    placeholder={`Paste your server logs here...\nExample:\n192.168.1.100 - - [10/Oct/2023:14:32:15 +0000] "GET /admin/login.php HTTP/1.1" 200 1234`}
                    rows={8}
                    className="font-mono text-sm"
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground mb-2 block">
                      Enter URL to analyze:
                    </label>
                    <Input
                      value={urlText}
                      onChange={(e) => setUrlText(e.target.value)}
                      placeholder="https://example.com"
                      className="font-mono"
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Switch 
                        id="crawl-enabled" 
                        checked={crawlEnabled}
                        onCheckedChange={setCrawlEnabled}
                      />
                      <label htmlFor="crawl-enabled" className="text-sm font-medium">
                        Enable Crawling
                      </label>
                    </div>
                    {crawlEnabled && (
                      <div className="flex items-center space-x-2">
                        <label className="text-sm">Max pages:</label>
                        <Input
                          type="number"
                          value={maxPages}
                          onChange={(e) => setMaxPages(Number(e.target.value))}
                          min={1}
                          max={50}
                          className="w-20"
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              <div className="flex items-center space-x-2">
                <Switch 
                  id="use-ann" 
                  checked={useNeuralNetwork}
                  onCheckedChange={setUseNeuralNetwork}
                />
                <label htmlFor="use-ann" className="text-sm font-medium">
                  Use Neural Network
                </label>
              </div>
              
              <div className="flex gap-2 justify-end">
                <Button 
                  variant="outline" 
                  onClick={() => setIsDialogOpen(false)}
                  disabled={isAnalyzing}
                >
                  Cancel
                </Button>
                <Button 
                  variant="hero" 
                  onClick={handleRunAnalysis}
                  disabled={isAnalyzing || (analysisType === 'upload' ? !logText.trim() : !urlText.trim())}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Plus className="mr-2 h-4 w-4" />
                      Run Analysis
                    </>
                  )}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Recent Analyses */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <Card className="ios-card">
            <CardHeader>
              <CardTitle>Recent Analyses</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentAnalyses.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    No recent analyses found. Run your first analysis to see results here.
                  </p>
                ) : (
                  recentAnalyses.map((analysis) => (
                    <div
                      key={analysis.id}
                      className="flex items-center justify-between rounded-lg bg-muted/30 p-4"
                    >
                      <div className="flex items-center gap-4">
                        <div className={`p-2 rounded-full ${
                          analysis.verdict.includes('ATTACK') 
                            ? 'bg-destructive/10 text-destructive'
                            : analysis.verdict.includes('SUSPICIOUS')
                            ? 'bg-warning/10 text-warning'
                            : 'bg-success/10 text-success'
                        }`}>
                          {analysis.verdict.includes('ATTACK') ? (
                            <AlertTriangle className="h-4 w-4" />
                          ) : (
                            <CheckCircle className="h-4 w-4" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium">Analysis #{analysis.id}</p>
                          <p className="text-sm text-muted-foreground">
                            {new Date(analysis.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            analysis.verdict.includes('ATTACK') 
                              ? 'destructive' 
                              : analysis.verdict.includes('SUSPICIOUS')
                              ? 'secondary'
                              : 'outline'
                          }
                        >
                          {analysis.method_used}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {analysis.line_count} lines
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </Layout>
  );
}