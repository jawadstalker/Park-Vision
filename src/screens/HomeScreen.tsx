import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
  ActivityIndicator,
  Alert,
  SafeAreaView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import DetectionOverlay from '../components/DetectionOverlay';
import { detectPixel, PixelDetectResponse } from '../api/parkVision';

export default function HomeScreen() {
  const [imageUri, setImageUri] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<PixelDetectResponse | null>(null);
  const [loading, setLoading] = React.useState(false);

  const pickAndDetect = async (fromCamera: boolean) => {
    const permission = fromCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert('Permission needed', 'Please allow access to continue.');
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
    } catch (error) {
      Alert.alert('Detection failed', error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  };

  const emptyCount = result?.spots.filter((s) => s.status === 'empty').length ?? 0;
  const occupiedCount = result?.spots.filter((s) => s.status === 'occupied').length ?? 0;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Park-Vision</Text>

        <View style={styles.buttonRow}>
          <Pressable style={styles.button} onPress={() => pickAndDetect(true)}>
            <Text style={styles.buttonText}>Take Photo</Text>
          </Pressable>
          <Pressable style={styles.button} onPress={() => pickAndDetect(false)}>
            <Text style={styles.buttonText}>Choose From Gallery</Text>
          </Pressable>
        </View>

        {loading && <ActivityIndicator size="large" style={styles.spinner} />}

        {imageUri && result && !loading && (
          <>
            <DetectionOverlay imageUri={imageUri} result={result} />
            <View style={styles.metrics}>
              <Text style={styles.metricText}>Vehicles: {result.vehicles.length}</Text>
              <Text style={[styles.metricText, styles.empty]}>Empty: {emptyCount}</Text>
              <Text style={[styles.metricText, styles.occupied]}>Occupied: {occupiedCount}</Text>
            </View>
            <Text style={styles.timing}>
              Processed in {result.processing_time_ms.toFixed(0)} ms
            </Text>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 16, alignItems: 'center' },
  title: { fontSize: 24, fontWeight: '700', marginBottom: 16 },
  buttonRow: { flexDirection: 'row', gap: 12, marginBottom: 16 },
  button: {
    backgroundColor: '#2563eb',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  buttonText: { color: '#fff', fontWeight: '600' },
  spinner: { marginTop: 24 },
  metrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginTop: 16,
  },
  metricText: { fontSize: 16, fontWeight: '600' },
  empty: { color: '#22c55e' },
  occupied: { color: '#ef4444' },
  timing: { marginTop: 8, color: '#666' },
});
