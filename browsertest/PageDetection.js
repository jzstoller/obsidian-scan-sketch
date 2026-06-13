// Services/PageDetection.ts
var DETECTION_MAX_DIM = 800;
function detectPageCorners(imageData) {
  const result = detectPageCornersDebug(imageData);
  return result.corners;
}
function detectPageCornersDebug(imageData) {
  const { data: scaled, scale, width, height } = downscale(imageData, DETECTION_MAX_DIM);
  const { saturation, value } = toHSVChannels(scaled, width, height);
  const satMask = thresholdBelow(saturation, width, height, 100);
  const valMask = thresholdAbove(value, width, height, 120);
  const paperMask = bitwiseAnd(satMask, valMask, width, height);
  const gray = toGrayscale(scaled, width, height);
  const denoised = medianFilter5x5(gray, width, height);
  const blurred = gaussianBlur5x5(denoised, width, height);
  const { magnitude, direction } = sobel(blurred, width, height);
  const suppressed = nonMaxSuppression(magnitude, direction, width, height);
  let edges = hysteresisThreshold(suppressed, width, height, 50, 100);
  edges = dilate3x3(edges, width, height);
  const contours = findContours(edges, width, height);
  const debug = { paperMask, edges, combined: edges, width, height };
  if (contours.length === 0)
    return { corners: null, hull: null, debug };
  const candidates = contours.map((c) => ({ contour: c, area: contourArea(c) })).sort((a, b) => b.area - a.area);
  let firstHull = null;
  for (const { contour } of candidates) {
    if (contour.length < 4)
      continue;
    const hull = convexHull(contour);
    if (hull.length < 4)
      continue;
    if (!firstHull)
      firstHull = hull.map((p) => ({ x: p.x / scale, y: p.y / scale }));
    let quad = approxPolyDP(hull, 0.02 * perimeter(hull));
    if (quad.length !== 4) {
      quad = boundingQuadFromHull(hull);
    }
    if (quad.length !== 4)
      continue;
    const ordered = orderCorners(quad);
    if (!isValidQuad(ordered) || !quadOverlapsPaper(ordered, paperMask, width, height)) {
      continue;
    }
    const corners = ordered.map((p) => ({ x: p.x / scale, y: p.y / scale, isDragging: false }));
    return { corners, hull: firstHull, debug };
  }
  return { corners: null, hull: firstHull, debug };
}
function downscale(imageData, maxDim) {
  const { width: w, height: h, data } = imageData;
  const scale = Math.min(1, maxDim / Math.max(w, h));
  if (scale === 1)
    return { data, scale: 1, width: w, height: h };
  const width = Math.round(w * scale);
  const height = Math.round(h * scale);
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcX = Math.min(w - 1, Math.floor(x / scale));
      const srcY = Math.min(h - 1, Math.floor(y / scale));
      const srcIdx = (srcY * w + srcX) * 4;
      const dstIdx = (y * width + x) * 4;
      out[dstIdx] = data[srcIdx];
      out[dstIdx + 1] = data[srcIdx + 1];
      out[dstIdx + 2] = data[srcIdx + 2];
      out[dstIdx + 3] = data[srcIdx + 3];
    }
  }
  return { data: out, scale, width, height };
}
function toGrayscale(data, width, height) {
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return gray;
}
function toHSVChannels(data, width, height) {
  const saturation = new Uint8Array(width * height);
  const value = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const v = max;
    const s = max === 0 ? 0 : (max - min) / max;
    saturation[i] = Math.round(s * 255);
    value[i] = Math.round(v * 255);
  }
  return { saturation, value };
}
function medianFilter5x5(src, width, height) {
  const out = new Float32Array(width * height);
  const window = new Float32Array(25);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let n = 0;
      for (let ky = -2; ky <= 2; ky++) {
        for (let kx = -2; kx <= 2; kx++) {
          const sx = clamp(x + kx, 0, width - 1);
          const sy = clamp(y + ky, 0, height - 1);
          window[n++] = src[sy * width + sx];
        }
      }
      const sorted = Array.from(window).sort((a, b) => a - b);
      out[y * width + x] = sorted[12];
    }
  }
  return out;
}
function gaussianBlur5x5(src, width, height) {
  const kernel = [2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2];
  const kernelSum = 159;
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let k = 0;
      for (let ky = -2; ky <= 2; ky++) {
        for (let kx = -2; kx <= 2; kx++) {
          const sx = clamp(x + kx, 0, width - 1);
          const sy = clamp(y + ky, 0, height - 1);
          sum += src[sy * width + sx] * kernel[k++];
        }
      }
      out[y * width + x] = sum / kernelSum;
    }
  }
  return out;
}
function thresholdAbove(src, width, height, value) {
  const out = new Uint8Array(width * height);
  for (let i = 0; i < src.length; i++) {
    out[i] = src[i] >= value ? 255 : 0;
  }
  return out;
}
function thresholdBelow(src, width, height, value) {
  const out = new Uint8Array(width * height);
  for (let i = 0; i < src.length; i++) {
    out[i] = src[i] < value ? 255 : 0;
  }
  return out;
}
function bitwiseAnd(a, b, width, height) {
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i] !== 0 && b[i] !== 0 ? 255 : 0;
  }
  return out;
}
function dilate3x3(src, width, height) {
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let any = false;
      for (let ky = -1; ky <= 1 && !any; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const sx = x + kx;
          const sy = y + ky;
          if (sx < 0 || sx >= width || sy < 0 || sy >= height)
            continue;
          if (src[sy * width + sx] !== 0) {
            any = true;
            break;
          }
        }
      }
      out[y * width + x] = any ? 255 : 0;
    }
  }
  return out;
}
function sobel(src, width, height) {
  const magnitude = new Float32Array(width * height);
  const direction = new Float32Array(width * height);
  const gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sx = 0;
      let sy = 0;
      let k = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const px = clamp(x + kx, 0, width - 1);
          const py = clamp(y + ky, 0, height - 1);
          const val = src[py * width + px];
          sx += val * gx[k];
          sy += val * gy[k];
          k++;
        }
      }
      const idx = y * width + x;
      magnitude[idx] = Math.sqrt(sx * sx + sy * sy);
      direction[idx] = Math.atan2(sy, sx);
    }
  }
  return { magnitude, direction };
}
function nonMaxSuppression(magnitude, direction, width, height) {
  const out = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      let angle = direction[idx] * (180 / Math.PI);
      if (angle < 0)
        angle += 180;
      let n1;
      let n2;
      if (angle >= 0 && angle < 22.5 || angle >= 157.5 && angle <= 180) {
        n1 = magnitude[idx - 1];
        n2 = magnitude[idx + 1];
      } else if (angle >= 22.5 && angle < 67.5) {
        n1 = magnitude[idx - width + 1];
        n2 = magnitude[idx + width - 1];
      } else if (angle >= 67.5 && angle < 112.5) {
        n1 = magnitude[idx - width];
        n2 = magnitude[idx + width];
      } else {
        n1 = magnitude[idx - width - 1];
        n2 = magnitude[idx + width + 1];
      }
      const m = magnitude[idx];
      out[idx] = m >= n1 && m >= n2 ? m : 0;
    }
  }
  return out;
}
function hysteresisThreshold(src, width, height, low, high) {
  const strong = 255;
  const weak = 75;
  const out = new Uint8Array(width * height);
  for (let i = 0; i < src.length; i++) {
    if (src[i] >= high)
      out[i] = strong;
    else if (src[i] >= low)
      out[i] = weak;
    else
      out[i] = 0;
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        if (out[idx] !== weak)
          continue;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            if (out[(y + ky) * width + (x + kx)] === strong) {
              out[idx] = strong;
              changed = true;
              break;
            }
          }
          if (out[idx] === strong)
            break;
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) {
    if (out[i] === weak)
      out[i] = 0;
  }
  return out;
}
function findContours(edges, width, height) {
  const visited = new Uint8Array(width * height);
  const contours = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (edges[idx] === 0 || visited[idx])
        continue;
      const component = [];
      const stack = [idx];
      visited[idx] = 1;
      while (stack.length > 0) {
        const cur = stack.pop();
        const cx = cur % width;
        const cy = Math.floor(cur / width);
        component.push({ x: cx, y: cy });
        for (let ny = -1; ny <= 1; ny++) {
          for (let nx = -1; nx <= 1; nx++) {
            if (nx === 0 && ny === 0)
              continue;
            const px = cx + nx;
            const py = cy + ny;
            if (px < 0 || px >= width || py < 0 || py >= height)
              continue;
            const pidx = py * width + px;
            if (edges[pidx] !== 0 && !visited[pidx]) {
              visited[pidx] = 1;
              stack.push(pidx);
            }
          }
        }
      }
      if (component.length > 60)
        contours.push(component);
    }
  }
  return contours;
}
function contourArea(points) {
  const hull = convexHull(points);
  if (hull.length < 3)
    return 0;
  let area = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    area += a.x * b.y - b.x * a.y;
  }
  return Math.abs(area) / 2;
}
function perimeter(points) {
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    total += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return total;
}
function approxPolyDP(points, epsilon) {
  if (points.length < 3)
    return points;
  const dmaxInfo = findFarthestPoint(points, points[0], points[points.length - 1]);
  if (dmaxInfo.distance > epsilon) {
    const left = approxPolyDP(points.slice(0, dmaxInfo.index + 1), epsilon);
    const right = approxPolyDP(points.slice(dmaxInfo.index), epsilon);
    return [...left.slice(0, -1), ...right];
  }
  return [points[0], points[points.length - 1]];
}
function findFarthestPoint(points, lineStart, lineEnd) {
  let maxDist = 0;
  let maxIdx = 0;
  for (let i = 1; i < points.length - 1; i++) {
    const d = pointToLineDistance(points[i], lineStart, lineEnd);
    if (d > maxDist) {
      maxDist = d;
      maxIdx = i;
    }
  }
  return { index: maxIdx, distance: maxDist };
}
function pointToLineDistance(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  if (len === 0)
    return Math.hypot(p.x - a.x, p.y - a.y);
  return Math.abs((p.x - a.x) * dy - (p.y - a.y) * dx) / len;
}
function convexHull(points) {
  const sorted = [...points].sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
  if (sorted.length <= 2)
    return sorted;
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return [...lower, ...upper];
}
function boundingQuadFromHull(hull) {
  if (hull.length < 4)
    return [];
  let tl = hull[0];
  let br = hull[0];
  let tr = hull[0];
  let bl = hull[0];
  for (const p of hull) {
    if (p.x + p.y < tl.x + tl.y)
      tl = p;
    if (p.x + p.y > br.x + br.y)
      br = p;
    if (p.x - p.y > tr.x - tr.y)
      tr = p;
    if (p.x - p.y < bl.x - bl.y)
      bl = p;
  }
  return [tl, tr, br, bl];
}
function orderCorners(points) {
  let tl = points[0];
  let br = points[0];
  let tr = points[0];
  let bl = points[0];
  for (const p of points) {
    if (p.x + p.y < tl.x + tl.y)
      tl = p;
    if (p.x + p.y > br.x + br.y)
      br = p;
    if (p.x - p.y > tr.x - tr.y)
      tr = p;
    if (p.x - p.y < bl.x - bl.y)
      bl = p;
  }
  return [tl, tr, br, bl];
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function isValidQuad(ordered) {
  for (let i = 0; i < ordered.length; i++) {
    for (let j = i + 1; j < ordered.length; j++) {
      if (dist(ordered[i], ordered[j]) < 10)
        return false;
    }
  }
  const [tl, tr, br, bl] = ordered;
  const topDist = dist(tl, tr);
  const bottomDist = dist(bl, br);
  const leftDist = dist(tl, bl);
  const rightDist = dist(tr, br);
  const widthAvg = (topDist + bottomDist) / 2;
  const heightAvg = (leftDist + rightDist) / 2;
  const aspectRatio = widthAvg / heightAvg;
  if (aspectRatio < 0.3 || aspectRatio > 3)
    return false;
  const topBottomRatio = Math.max(topDist, bottomDist) / Math.min(topDist, bottomDist);
  if (topBottomRatio > 1.5)
    return false;
  const leftRightRatio = Math.max(leftDist, rightDist) / Math.min(leftDist, rightDist);
  if (leftRightRatio > 1.5)
    return false;
  const minSideLength = Math.min(topDist, bottomDist, leftDist, rightDist);
  if (minSideLength < 20)
    return false;
  return true;
}
function quadOverlapsPaper(points, paperMask, width, height) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const x0 = Math.max(0, Math.min(...xs));
  const y0 = Math.max(0, Math.min(...ys));
  const x1 = Math.min(width - 1, Math.max(...xs));
  const y1 = Math.min(height - 1, Math.max(...ys));
  let paperPixels = 0;
  let total = 0;
  const step = 10;
  for (let y = y0; y <= y1; y += step) {
    for (let x = x0; x <= x1; x += step) {
      if (paperMask[Math.round(y) * width + Math.round(x)] > 0)
        paperPixels++;
      total++;
    }
  }
  return total > 0 && paperPixels / total > 0.15;
}
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
export {
  detectPageCorners,
  detectPageCornersDebug
};
