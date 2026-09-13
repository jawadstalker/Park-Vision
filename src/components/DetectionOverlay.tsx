import React from 'react';
import { Image, View, StyleSheet, LayoutChangeEvent } from 'react-native';
import Svg, { Rect } from 'react-native-svg';
import type { PixelDetectResponse } from '../api/parkVision';

interface Props {
  imageUri: string;
  result: PixelDetectResponse;
}

export default function DetectionOverlay({ imageUri, result }: Props) {
  const [displaySize, setDisplaySize] = React.useState({ width: 0, height: 0 });

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setDisplaySize({ width, height });
  };

  const scaleX = displaySize.width / result.frame_width;
  const scaleY = displaySize.height / result.frame_height;

  return (
    <View
      style={[styles.container, { aspectRatio: result.frame_width / result.frame_height }]}
      onLayout={onLayout}
    >
      <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
      {displaySize.width > 0 && (
        <Svg style={StyleSheet.absoluteFill}>
          {result.spots.map((spot) => {
            const [x1, y1, x2, y2] = spot.bbox;
            const color = spot.status === 'empty' ? '#22c55e' : '#ef4444';
            return (
              <Rect
                key={`spot-${spot.id}`}
                x={x1 * scaleX}
                y={y1 * scaleY}
                width={(x2 - x1) * scaleX}
                height={(y2 - y1) * scaleY}
                stroke={color}
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
  container: {
    width: '100%',
    backgroundColor: '#111',
  },
  image: {
    width: '100%',
    height: '100%',
  },
});
