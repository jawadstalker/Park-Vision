import React from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Pressable,
  ScrollView,
  Switch,
  SafeAreaView,
  Alert,
} from 'react-native';
import { useSettings, DetectionMode } from '../context/SettingsContext';
import { fetchStatus } from '../api/parkVision';

interface Props {
  onClose: () => void;
}

export default function SettingsScreen({ onClose }: Props) {
  const { settings, updateSettings } = useSettings();
  const [calibratedCameras, setCalibratedCameras] = React.useState<string[]>([]);
  const [checkingConnection, setCheckingConnection] = React.useState(false);

  const checkConnection = async () => {
    setCheckingConnection(true);
    try {
      const status = await fetchStatus(settings);
      setCalibratedCameras(status.cameras_calibrated);
      Alert.alert(
        'Connected',
        `Model loaded: ${status.model_loaded}\nCalibrated cameras: ${
          status.cameras_calibrated.join(', ') || 'none'
        }`
      );
    } catch (error) {
      Alert.alert('Connection failed', error instanceof Error ? error.message : String(error));
    } finally {
      setCheckingConnection(false);
    }
  };

  const setMode = (mode: DetectionMode) => updateSettings({ detectionMode: mode });

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.headerRow}>
          <Text style={styles.title}>Settings</Text>
          <Pressable onPress={onClose}>
            <Text style={styles.close}>Done</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionLabel}>Backend</Text>
        <TextInput
          style={styles.input}
          value={settings.apiBaseUrl}
          onChangeText={(text) => updateSettings({ apiBaseUrl: text })}
          placeholder="http://192.168.1.20:8000"
          autoCapitalize="none"
          autoCorrect={false}
        />
        <TextInput
          style={styles.input}
          value={settings.apiKey}
          onChangeText={(text) => updateSettings({ apiKey: text })}
          placeholder="API key (leave blank if auth is disabled)"
          autoCapitalize="none"
          autoCorrect={false}
          secureTextEntry
        />
        <Pressable style={styles.secondaryButton} onPress={checkConnection} disabled={checkingConnection}>
          <Text style={styles.secondaryButtonText}>
            {checkingConnection ? 'Checking…' : 'Test Connection'}
          </Text>
        </Pressable>

        <Text style={styles.sectionLabel}>Detection mode</Text>
        <View style={styles.modeRow}>
          <Pressable
            style={[styles.modeButton, settings.detectionMode === 'calibrated' && styles.modeButtonActive]}
            onPress={() => setMode('calibrated')}
          >
            <Text
              style={[
                styles.modeButtonText,
                settings.detectionMode === 'calibrated' && styles.modeButtonTextActive,
              ]}
            >
              Calibrated (meters)
            </Text>
          </Pressable>
          <Pressable
            style={[styles.modeButton, settings.detectionMode === 'pixel' && styles.modeButtonActive]}
            onPress={() => setMode('pixel')}
          >
            <Text
              style={[
                styles.modeButtonText,
                settings.detectionMode === 'pixel' && styles.modeButtonTextActive,
              ]}
            >
              No calibration (pixel-based)
            </Text>
          </Pressable>
        </View>

        {settings.detectionMode === 'calibrated' ? (
          <>
            <Text style={styles.sectionLabel}>Camera ID (must already be calibrated)</Text>
            <TextInput
              style={styles.input}
              value={settings.cameraId}
              onChangeText={(text) => updateSettings({ cameraId: text })}
              placeholder="street-01"
              autoCapitalize="none"
              autoCorrect={false}
            />
            {calibratedCameras.length > 0 && (
              <Text style={styles.hint}>Known cameras: {calibratedCameras.join(', ')}</Text>
            )}

            <Text style={styles.sectionLabel}>
              Empty spot threshold: {settings.gapThresholdM.toFixed(1)} m
            </Text>
            <StepperRow
              value={settings.gapThresholdM}
              step={0.5}
              min={2}
              max={8}
              onChange={(v) => updateSettings({ gapThresholdM: v })}
            />
          </>
        ) : (
          <>
            <Text style={styles.sectionLabel}>Camera ID (optional, for history)</Text>
            <TextInput
              style={styles.input}
              value={settings.cameraId}
              onChangeText={(text) => updateSettings({ cameraId: text })}
              placeholder="cam-1 (optional)"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <Text style={styles.sectionLabel}>
              Minimum gap ratio: {settings.minGapRatio.toFixed(2)}
            </Text>
            <StepperRow
              value={settings.minGapRatio}
              step={0.05}
              min={0.5}
              max={2.0}
              onChange={(v) => updateSettings({ minGapRatio: v })}
            />

            <Text style={styles.sectionLabel}>
              Row grouping tolerance: {settings.rowToleranceRatio.toFixed(2)}
            </Text>
            <StepperRow
              value={settings.rowToleranceRatio}
              step={0.05}
              min={0.2}
              max={1.5}
              onChange={(v) => updateSettings({ rowToleranceRatio: v })}
            />

            <View style={styles.switchRow}>
              <Text style={styles.sectionLabel}>Recheck large gaps (slower, more accurate)</Text>
              <Switch
                value={settings.enableRecheck}
                onValueChange={(v) => updateSettings({ enableRecheck: v })}
              />
            </View>
          </>
        )}

        <Text style={styles.sectionLabel}>
          Detection confidence threshold: {settings.conf.toFixed(2)}
        </Text>
        <StepperRow
          value={settings.conf}
          step={0.05}
          min={0.1}
          max={0.9}
          onChange={(v) => updateSettings({ conf: v })}
        />
      </ScrollView>
    </SafeAreaView>
  );
}

function StepperRow({
  value,
  step,
  min,
  max,
  onChange,
}: {
  value: number;
  step: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  return (
    <View style={styles.stepperRow}>
      <Pressable
        style={styles.stepperButton}
        onPress={() => onChange(clamp(Math.round((value - step) * 100) / 100))}
      >
        <Text style={styles.stepperButtonText}>−</Text>
      </Pressable>
      <Pressable
        style={styles.stepperButton}
        onPress={() => onChange(clamp(Math.round((value + step) * 100) / 100))}
      >
        <Text style={styles.stepperButtonText}>+</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 16 },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: { fontSize: 22, fontWeight: '700' },
  close: { fontSize: 16, color: '#2563eb', fontWeight: '600' },
  sectionLabel: { fontSize: 14, fontWeight: '600', marginTop: 16, marginBottom: 6, color: '#333' },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 15,
  },
  hint: { fontSize: 12, color: '#666', marginTop: 4 },
  secondaryButton: {
    marginTop: 10,
    alignSelf: 'flex-start',
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#eef2ff',
    borderRadius: 8,
  },
  secondaryButtonText: { color: '#2563eb', fontWeight: '600' },
  modeRow: { flexDirection: 'row', gap: 8 },
  modeButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    alignItems: 'center',
  },
  modeButtonActive: { backgroundColor: '#2563eb', borderColor: '#2563eb' },
  modeButtonText: { fontSize: 13, fontWeight: '600', color: '#333', textAlign: 'center' },
  modeButtonTextActive: { color: '#fff' },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
  },
  stepperRow: { flexDirection: 'row', gap: 12 },
  stepperButton: {
    width: 44,
    height: 36,
    borderRadius: 8,
    backgroundColor: '#eee',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepperButtonText: { fontSize: 20, fontWeight: '700' },
});
