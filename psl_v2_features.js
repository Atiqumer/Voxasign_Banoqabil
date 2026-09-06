// Exact browser implementation of training/psl/feature_engineering.py.
// The input landmarks must be MediaPipe's 21 landmarks in its documented order.

function distance(points, first, second) {
  const dx = points[first][0] - points[second][0];
  const dy = points[first][1] - points[second][1];
  const dz = points[first][2] - points[second][2];
  return Math.hypot(dx, dy, dz);
}

function normalizedAngle(points, first, middle, last) {
  const ba = [
    points[first][0] - points[middle][0],
    points[first][1] - points[middle][1],
    points[first][2] - points[middle][2],
  ];
  const bc = [
    points[last][0] - points[middle][0],
    points[last][1] - points[middle][1],
    points[last][2] - points[middle][2],
  ];
  const baLength = Math.max(Math.hypot(...ba), 1e-8);
  const bcLength = Math.max(Math.hypot(...bc), 1e-8);
  const cosine = Math.max(
    -1,
    Math.min(1, (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / (baLength * bcLength)),
  );
  return Math.acos(cosine) / Math.PI;
}

export function normalizePslLandmarks(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) {
    throw new Error("PSL V2 requires exactly 21 MediaPipe hand landmarks.");
  }

  const wrist = landmarks[0];
  const points = landmarks.map((landmark) => [
    landmark.x - wrist.x,
    landmark.y - wrist.y,
    landmark.z - wrist.z,
  ]);
  const maximum = Math.max(...points.flat().map(Math.abs), 0);
  const scale = maximum > 0 ? maximum : 1;
  return points.map((point) => point.map((value) => value / scale));
}

export function extractPslV2Features(landmarks) {
  const points = normalizePslLandmarks(landmarks);
  const features = points.flat(); // Original 63 features.

  const bonePairs = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
  ];
  bonePairs.forEach(([first, second]) => features.push(distance(points, first, second)));

  const angleTriplets = [
    [1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8], [9, 10, 11],
    [10, 11, 12], [13, 14, 15], [14, 15, 16], [17, 18, 19], [18, 19, 20],
  ];
  angleTriplets.forEach(([first, middle, last]) => {
    features.push(normalizedAngle(points, first, middle, last));
  });

  [4, 8, 12, 16, 20].forEach((tip) => features.push(distance(points, 0, tip)));

  const fingertipPairs = [
    [4, 8], [4, 12], [4, 16], [4, 20], [8, 12], [8, 16], [8, 20],
    [12, 16], [12, 20], [16, 20],
  ];
  fingertipPairs.forEach(([first, second]) => features.push(distance(points, first, second)));

  const palmPairs = [[0, 5], [0, 9], [0, 13], [0, 17], [5, 9], [9, 13], [13, 17]];
  palmPairs.forEach(([first, second]) => features.push(distance(points, first, second)));

  if (features.length !== 115) {
    throw new Error(`PSL V2 feature extraction produced ${features.length}, expected 115.`);
  }
  return Float32Array.from(features);
}

export function scalePslV2Features(features, scaler) {
  if (features.length !== 115 || scaler.mean.length !== 115 || scaler.scale.length !== 115) {
    throw new Error("PSL V2 scaler and feature vector must each contain 115 values.");
  }
  return Float32Array.from(features, (value, index) => {
    const scale = scaler.scale[index] === 0 ? 1 : scaler.scale[index];
    return (value - scaler.mean[index]) / scale;
  });
}
