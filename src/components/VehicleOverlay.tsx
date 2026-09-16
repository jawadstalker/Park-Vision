import React from 'react';
import { Image, View, StyleSheet, LayoutChangeEvent } from 'react-native';
import Svg, { Rect } from 'react-native-svg';

interface VehicleBox {
  bbox: [number, number, number, number];
}

interface Props {
  imageUri: string;
  frameWidth: number;
  frameHeight: number;
  vehicles: VehicleBox[];
}

export default function VehicleOverlay({ imageUri, frameWidth, frameHeight, vehicles }: Props) {
  const [displaySize, setDisplaySize] = React.useState({ width: 0, height: 0 });

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setDisplaySize({ width, height });
  };

  const scaleX = displaySize.width / frameWidth;
  const scaleY = displaySize.height / frameHeight;

  return (
    <View style={[styles.container, { aspectRatio: frameWidth / frameHeight }]} onLayout={onLayout}>
      <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
      {displaySize.width > 0 && (
        <Svg style={StyleSheet.absoluteFill}>
          {vehicles.map((vehicle, index) => {
            const [x1, y1, x2, y2] = vehicle.bbox;
            return (
              <Rect
                key={`vehicle-${index}`}
                x={x1 * scaleX}
                y={y1 * scaleY}
                width={(x2 - x1) * scaleX}
                height={(y2 - y1) * scaleY}
                stroke="#22c55e"
                strokeWidth={2}
                fill="none"
              />
            );
          })}
        </Svg>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: '100%', backgroundColor: '#111' },
  image: { width: '100%', height: '100%' },
});
