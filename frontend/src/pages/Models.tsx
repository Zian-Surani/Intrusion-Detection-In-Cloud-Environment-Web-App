import React from 'react';
import { motion } from 'framer-motion';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Upload, 
  Download, 
  Settings, 
  Star,
  Activity,
  Calendar,
  Zap
} from 'lucide-react';

const models = [
  {
    id: 'ann-v2.1',
    name: 'Neural Network v2.1',
    type: 'ANN',
    status: 'active',
    accuracy: 94.2,
    lastTrained: '2023-10-08',
    size: '15.2 MB',
    isDefault: true,
  },
  {
    id: 'dfa-standard',
    name: 'Standard DFA Rules',
    type: 'DFA',
    status: 'active',
    accuracy: 98.7,
    lastTrained: '2023-10-01',
    size: '2.1 MB',
    isDefault: false,
  },
  {
    id: 'ann-experimental',
    name: 'Experimental Neural v3.0',
    type: 'ANN',
    status: 'training',
    accuracy: 0,
    lastTrained: 'In Progress',
    size: '22.8 MB',
    isDefault: false,
  },
];

export function Models() {
  return (
    <Layout title="Models">
      <div className="container mx-auto p-4 space-y-6">
        {/* Upload New Model */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="ios-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload New Model
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-border rounded-xl p-8 text-center">
                <Brain className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="font-semibold mb-2">Drop .joblib files here</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Support for scikit-learn models and custom neural networks
                </p>
                <Button variant="outline">Choose Model File</Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Model List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="ios-card">
            <CardHeader>
              <CardTitle>Installed Models</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {models.map((model, index) => (
                  <motion.div
                    key={model.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="rounded-lg border border-border/50 bg-card/30 p-4"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4">
                        <div className={`p-3 rounded-lg ${
                          model.type === 'ANN' 
                            ? 'bg-accent/10 text-accent'
                            : 'bg-primary/10 text-primary'
                        }`}>
                          {model.type === 'ANN' ? (
                            <Brain className="h-6 w-6" />
                          ) : (
                            <Zap className="h-6 w-6" />
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold">{model.name}</h3>
                            {model.isDefault && (
                              <Badge variant="default">
                                <Star className="mr-1 h-3 w-3" />
                                Default
                              </Badge>
                            )}
                            <Badge variant={
                              model.status === 'active' ? 'outline' :
                              model.status === 'training' ? 'secondary' : 'destructive'
                            }>
                              <Activity className="mr-1 h-3 w-3" />
                              {model.status}
                            </Badge>
                          </div>
                          <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                            <div>
                              <span className="font-medium">Accuracy:</span> {
                                model.accuracy > 0 ? `${model.accuracy}%` : 'Training...'
                              }
                            </div>
                            <div>
                              <span className="font-medium">Size:</span> {model.size}
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="h-3 w-3" />
                              <span className="font-medium">Last Updated:</span> {model.lastTrained}
                            </div>
                          </div>
                          {model.status === 'training' && (
                            <div className="mt-3">
                              <Progress value={67} className="h-2" />
                              <p className="text-xs text-muted-foreground mt-1">
                                Training progress: 67% complete
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm">
                          <Settings className="mr-2 h-4 w-4" />
                          Configure
                        </Button>
                        <Button variant="outline" size="sm">
                          <Download className="mr-2 h-4 w-4" />
                          Export
                        </Button>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </Layout>
  );
}