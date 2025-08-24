import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  User,
  Bell,
  Shield,
  Palette,
  Database,
  Key,
  Globe,
  Moon,
  Sun,
  Monitor,
  Save,
  Check,
} from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import Badge from '../components/ui/Badge';

const Settings: React.FC = () => {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('dark');
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    updates: true,
  });
  const [apiConfig, setApiConfig] = useState({
    endpoint: 'http://localhost:8000',
    timeout: 30,
  });
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-6xl mx-auto space-y-6 p-6"
    >
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display font-bold text-neutral-900 dark:text-neutral-100 mb-2">
          Settings
        </h1>
        <p className="text-neutral-600 dark:text-neutral-400">
          Manage your application preferences and configuration
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Profile Settings */}
          <Card variant="default">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900/30 rounded-xl flex items-center justify-center">
                  <User className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                    Profile
                  </h2>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Manage your account information
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                <Input
                  label="Display Name"
                  placeholder="John Doe"
                  defaultValue="User"
                />
                <Input
                  label="Email"
                  type="email"
                  placeholder="john@example.com"
                  defaultValue="user@example.com"
                />
                <Input
                  label="Organization"
                  placeholder="Acme Corp"
                  defaultValue="Free Plan"
                />
              </div>
            </div>
          </Card>

          {/* Appearance Settings */}
          <Card variant="default">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-accent-100 dark:bg-accent-900/30 rounded-xl flex items-center justify-center">
                  <Palette className="w-6 h-6 text-accent-600 dark:text-accent-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                    Appearance
                  </h2>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Customize the look and feel
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-3">
                    Theme
                  </label>
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      onClick={() => setTheme('light')}
                      className={`
                        p-3 rounded-xl border-2 transition-all
                        ${theme === 'light'
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-neutral-300 dark:border-neutral-600 hover:border-neutral-400 dark:hover:border-neutral-500'
                        }
                      `}
                    >
                      <Sun className="w-5 h-5 mx-auto mb-1 text-neutral-700 dark:text-neutral-300" />
                      <span className="text-xs text-neutral-700 dark:text-neutral-300">Light</span>
                    </button>
                    <button
                      onClick={() => setTheme('dark')}
                      className={`
                        p-3 rounded-xl border-2 transition-all
                        ${theme === 'dark'
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-neutral-300 dark:border-neutral-600 hover:border-neutral-400 dark:hover:border-neutral-500'
                        }
                      `}
                    >
                      <Moon className="w-5 h-5 mx-auto mb-1 text-neutral-700 dark:text-neutral-300" />
                      <span className="text-xs text-neutral-700 dark:text-neutral-300">Dark</span>
                    </button>
                    <button
                      onClick={() => setTheme('system')}
                      className={`
                        p-3 rounded-xl border-2 transition-all
                        ${theme === 'system'
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-neutral-300 dark:border-neutral-600 hover:border-neutral-400 dark:hover:border-neutral-500'
                        }
                      `}
                    >
                      <Monitor className="w-5 h-5 mx-auto mb-1 text-neutral-700 dark:text-neutral-300" />
                      <span className="text-xs text-neutral-700 dark:text-neutral-300">System</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* API Configuration */}
          <Card variant="default">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-success-100 dark:bg-success-900/30 rounded-xl flex items-center justify-center">
                  <Database className="w-6 h-6 text-success-600 dark:text-success-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                    API Configuration
                  </h2>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Configure backend connection
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                <Input
                  label="API Endpoint"
                  value={apiConfig.endpoint}
                  onChange={(e) => setApiConfig({ ...apiConfig, endpoint: e.target.value })}
                  leftIcon={<Globe className="w-4 h-4" />}
                />
                <Input
                  label="Request Timeout (seconds)"
                  type="number"
                  value={apiConfig.timeout}
                  onChange={(e) => setApiConfig({ ...apiConfig, timeout: parseInt(e.target.value) })}
                />
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Notifications */}
          <Card variant="default">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-warning-100 dark:bg-warning-900/30 rounded-xl flex items-center justify-center">
                  <Bell className="w-6 h-6 text-warning-600 dark:text-warning-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Notifications
                  </h2>
                </div>
              </div>

              <div className="space-y-3">
                <label className="flex items-center justify-between p-3 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 cursor-pointer">
                  <span className="text-sm text-neutral-700 dark:text-neutral-300">Email notifications</span>
                  <input
                    type="checkbox"
                    checked={notifications.email}
                    onChange={(e) => setNotifications({ ...notifications, email: e.target.checked })}
                    className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
                  />
                </label>
                <label className="flex items-center justify-between p-3 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 cursor-pointer">
                  <span className="text-sm text-neutral-700 dark:text-neutral-300">Push notifications</span>
                  <input
                    type="checkbox"
                    checked={notifications.push}
                    onChange={(e) => setNotifications({ ...notifications, push: e.target.checked })}
                    className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
                  />
                </label>
                <label className="flex items-center justify-between p-3 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 cursor-pointer">
                  <span className="text-sm text-neutral-700 dark:text-neutral-300">Product updates</span>
                  <input
                    type="checkbox"
                    checked={notifications.updates}
                    onChange={(e) => setNotifications({ ...notifications, updates: e.target.checked })}
                    className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
                  />
                </label>
              </div>
            </div>
          </Card>

          {/* Security */}
          <Card variant="default">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-error-100 dark:bg-error-900/30 rounded-xl flex items-center justify-center">
                  <Shield className="w-6 h-6 text-error-600 dark:text-error-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Security
                  </h2>
                </div>
              </div>

              <div className="space-y-3">
                <Button
                  variant="outline"
                  size="sm"
                  leftIcon={<Key className="w-4 h-4" />}
                  className="w-full"
                >
                  Change Password
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  leftIcon={<Shield className="w-4 h-4" />}
                  className="w-full"
                >
                  Enable 2FA
                </Button>
                <div className="pt-3 border-t border-neutral-200 dark:border-neutral-700">
                  <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-2">
                    Last login: Today at 10:30 AM
                  </p>
                  <p className="text-xs text-neutral-600 dark:text-neutral-400">
                    IP Address: 192.168.1.1
                  </p>
                </div>
              </div>
            </div>
          </Card>

          {/* Plan Info */}
          <Card variant="gradient">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Current Plan
                </h3>
                <Badge variant="primary">Free</Badge>
              </div>
              <div className="space-y-2 mb-4">
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-600 dark:text-neutral-400">Documents</span>
                  <span className="text-neutral-900 dark:text-neutral-100">10 / 100</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-600 dark:text-neutral-400">Queries</span>
                  <span className="text-neutral-900 dark:text-neutral-100">50 / 1000</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-600 dark:text-neutral-400">Storage</span>
                  <span className="text-neutral-900 dark:text-neutral-100">100 MB / 1 GB</span>
                </div>
              </div>
              <Button variant="primary" size="sm" className="w-full">
                Upgrade Plan
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <Button
          variant="primary"
          size="lg"
          leftIcon={saved ? <Check className="w-5 h-5" /> : <Save className="w-5 h-5" />}
          onClick={handleSave}
          className="min-w-[150px]"
        >
          {saved ? 'Saved!' : 'Save Changes'}
        </Button>
      </div>
    </motion.div>
  );
};

export default Settings;