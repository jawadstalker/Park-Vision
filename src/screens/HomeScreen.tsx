import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
  ActivityIndicator,
  SafeAreaView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import DetectionOverlay from '../components/DetectionOverlay';
import { detectPixel, PixelDetectResponse } from '../api/parkVision';

export default function HomeScreen() {
  const [imageUri, setImageUri] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<PixelDetectResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const pickAndDetect = async (fromCamera: boolean) => {
    setError(null);
    const permission = fromCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      setError('Please allow camera/gallery access to continue.');
      return;
    }

    const pickerResult = fromCamera
      ? await ImagePicker.launchCameraAsync({ quality: 0.8 })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });

    if (pickerResult.canceled || !pickerResult.assets?.length) {
      return;
    }

    const uri = pickerResult.assets[0].uri;
    setImageUri(uri);
    setResult(null);
    setLoading(true);

    try {
      const response = await detectPixel(uri, { enableRecheck: true });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const emptyCount = result?.spots.filter((s) => s.status === 'empty').length ?? 0;
  const occupiedCount = result?.spots.filter((s) => s.status === 'occupied').length ?? 0;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Park Vision</Text>
          <Text style={styles.subtitle}>Snap a street photo to find open spots</Text>
        </View>

        <View style={styles.buttonRow}>
          <Pressable
            style={({ pressed }) => [styles.button, styles.primaryButton, pressed && styles.buttonPressed]}
            onPress={() => pickAndDetect(true)}
          >
            <Text style={styles.buttonIcon}>📷</Text>
            <Text style={styles.buttonText}>Take Photo</Text>
          </Pressable>
          <Pressable
            style={({ pressed }) => [styles.button, styles.secondaryButton, pressed && styles.buttonPressed]}
            onPress={() => pickAndDetect(false)}
          >
            <Text style={styles.buttonIcon}>🖼️</Text>
            <Text style={[styles.buttonText, styles.secondaryButtonText]}>Gallery</Text>
          </Pressable>
        </View>

        {loading && (
          <View style={styles.card}>
            <ActivityIndicator size="large" color="#2563eb" />
            <Text style={styles.loadingText}>Analyzing photo…</Text>
          </View>
        )}

        {error && !loading && (
          <View style={[styles.card, styles.errorCard]}>
            <Text style={styles.errorTitle}>Detection failed</Text>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {imageUri && result && !loading && !error && (
          <View style={styles.card}>
            <DetectionOverlay imageUri={imageUri} result={result} />

            <View style={styles.metrics}>
              <View style={styles.metricPill}>
                <Text style={styles.metricValue}>{result.vehicles.length}</Text>
                <Text style={styles.metricLabel}>Vehicles</Text>
              </View>
              <View style={[styles.metricPill, styles.metricPillEmpty]}>
                <Text style={[styles.metricValue, styles.empty]}>{emptyCount}</Text>
                <Text style={styles.metricLabel}>Empty</Text>
              </View>
              <View style={[styles.metricPill, styles.metricPillOccupied]}>
                <Text style={[styles.metricValue, styles.occupied]}>{occupiedCount}</Text>
                <Text style={styles.metricLabel}>Occupied</Text>
              </View>
            </View>

            <Text style={styles.timing}>Processed in {result.processing_time_ms.toFixed(0)} ms</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#f4f6fb' },
  content: { padding: 20, paddingBottom: 40, alignItems: 'center' },
  header: { alignItems: 'center', marginBottom: 24, marginTop: 8 },
  title: { fontSize: 28, fontWeight: '800', color: '#0f172a', letterSpacing: -0.5 },
  subtitle: { fontSize: 14, color: '#64748b', marginTop: 4 },
  buttonRow: { flexDirection: 'row', gap: 12, marginBottom: 20, width: '100%', maxWidth: 420 },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    borderRadius: 14,
    gap: 8,
  },
  primaryButton: {
    backgroundColor: '#2563eb',
    shadowColor: '#2563eb',
    shadowOpacity: 0.3,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 3,
  },
  secondaryButton: {
    backgroundColor: '#fff',
    borderWidth: 1.5,
    borderColor: '#e2e8f0',
  },
  buttonPressed: { opacity: 0.8, transform: [{ scale: 0.98 }] },
  buttonIcon: { fontSize: 16 },
  buttonText: { color: '#fff', fontWeight: '700', fontSize: 15 },
  secondaryButtonText: { color: '#334155' },
  card: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#0f172a',
    shadowOpacity: 0.08,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 6 },
    elevation: 2,
  },
  loadingText: { marginTop: 12, color: '#64748b', fontWeight: '500' },
  errorCard: { backgroundColor: '#fef2f2', borderWidth: 1, borderColor: '#fecaca' },
  errorTitle: { color: '#b91c1c', fontWeight: '700', fontSize: 15, marginBottom: 4 },
  errorText: { color: '#991b1b', fontSize: 13, textAlign: 'center' },
  metrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginTop: 16,
    gap: 8,
  },
  metricPill: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: '#f1f5f9',
    borderRadius: 14,
    paddingVertical: 12,
  },
  metricPillEmpty: { backgroundColor: '#f0fdf4' },
  metricPillOccupied: { backgroundColor: '#fef2f2' },
  metricValue: { fontSize: 20, fontWeight: '800', color: '#0f172a' },
  metricLabel: { fontSize: 11, color: '#64748b', marginTop: 2, fontWeight: '600' },
  empty: { color: '#16a34a' },
  occupied: { color: '#dc2626' },
  timing: { marginTop: 12, color: '#94a3b8', fontSize: 12 },
});
