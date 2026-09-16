import React from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export type DetectionMode = 'calibrated' | 'pixel';

export interface Settings {
  apiBaseUrl: string;
  apiKey: string;
  detectionMode: DetectionMode;
  cameraId: string;
  conf: number;
  gapThresholdM: number;
  minGapRatio: number;
  rowToleranceRatio: number;
  enableRecheck: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  apiBaseUrl: 'http://localhost:8000',
  apiKey: '',
  detectionMode: 'pixel',
  cameraId: '',
  conf: 0.35,
  gapThresholdM: 4.5,
  minGapRatio: 0.85,
  rowToleranceRatio: 0.6,
  enableRecheck: true,
};

const STORAGE_KEY = 'park-vision-settings';

interface SettingsContextValue {
  settings: Settings;
  updateSettings: (partial: Partial<Settings>) => void;
  loaded: boolean;
}

const SettingsContext = React.createContext<SettingsContextValue | undefined>(undefined);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = React.useState<Settings>(DEFAULT_SETTINGS);
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY)
      .then((stored) => {
        if (stored) {
          setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(stored) });
        }
      })
      .finally(() => setLoaded(true));
  }, []);

  const updateSettings = (partial: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...partial };
      AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next)).catch(() => {
        // Best-effort persistence; in-memory state still updates either way.
      });
      return next;
    });
  };

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, loaded }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const context = React.useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}
