#version 330

in vec2 v_uv; 
out vec4 fragColor;
uniform sampler2D webcam;
uniform sampler2D histogram;
uniform float time;
uniform vec2 resolution;

const int bayerMatrix[16] = int[](
    0, 8, 2, 10,
    12, 4, 14, 6,
    3, 11, 1, 9,
    15, 7, 13, 5
);

const vec3 palette[5] = vec3[](
    vec3(0.00, 0.00, 0.00),
    vec3(0.05, 0.00, 0.18),
    vec3(0.30, 0.00, 0.70),
    vec3(0.55, 0.10, 1.00),
    vec3(0.85, 0.80, 1.00)
);

float hash(float n) { return fract(sin(n) * 43758.5453123); }

float sampleLuma(vec2 uv, vec2 texSize, float pixelSize) {
    vec2 snapped = floor(uv * texSize / pixelSize) * pixelSize / texSize;
    vec3 c = texture(webcam, snapped).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

float glyphBit(int[5] rows, int x, int y) {
    int row = rows[4 - y];
    return float((row >> (3 - x)) & 1);
}

int[5] getGlyph(int ch) {
    if (ch ==  0) return int[5](1, 1, 1, 9, 6);
    if (ch ==  1) return int[5](9, 9, 15, 9, 9);
    if (ch ==  2) return int[5](14, 8, 12, 8, 14);
    if (ch ==  3) return int[5](14, 9, 14, 10, 9);
    if (ch ==  4) return int[5](9, 15, 9, 9, 9);
    if (ch ==  5) return int[5](6, 9, 9, 9, 6);
    if (ch ==  6) return int[5](7, 8, 6, 1, 14);
    if (ch ==  7) return int[5](7, 2, 2, 2, 7);
    if (ch ==  8) return int[5](8, 8, 8, 8, 14);
    if (ch ==  9) return int[5](6, 9, 15, 9, 9);
    if (ch == 10) return int[5](14, 9, 9, 9, 14);
    if (ch == 12) return int[5](14, 9, 14, 10, 9);
    if (ch == 13) return int[5](14, 8, 12, 8, 14);
    if (ch == 14) return int[5](6, 8, 8, 8, 6);
    return int[5](0, 0, 0, 0, 0);
}

float drawText(float fx, float fy, float startX, float startY, float S, int text[12], int len) {
    float GLYPH_W   = 4.0 * S;
    float GAP       = 1.0 * S;
    float CHAR_STEP = GLYPH_W + GAP;
    float lx = fx - startX;
    float ly = fy - startY;
    if (lx < 0.0 || ly < 0.0 || ly >= 5.0 * S) return 0.0;
    int charPos = int(lx / CHAR_STEP);
    if (charPos < 0 || charPos >= len) return 0.0;
    float cx = lx - float(charPos) * CHAR_STEP;
    if (cx >= GLYPH_W) return 0.0;
    int px = int(cx / S);
    int py = int(ly / S);
    if (px < 0 || px > 3 || py < 0 || py > 4) return 0.0;
    int ci = text[charPos];
    if (ci < 0) return 0.0;
    return glyphBit(getGlyph(ci), px, py);
}

float recDot(vec2 fc, vec2 center, float r) {
    return 1.0 - step(r, length(fc - center));
}

// Histogram panel
const float PANEL_W  = 160.0;
const float PANEL_H  = 80.0;
const float PANEL_X  = 12.0;
const float PANEL_Y  = 12.0;
const float BORDER   = 1.0;

float drawHistogram(float fx, float fy, float W, float H) {
    float lx = fx - PANEL_X;
    float ly = fy - PANEL_Y;

    if (lx < 0.0 || lx >= PANEL_W || ly < 0.0 || ly >= PANEL_H) return -2.0;

    if (lx < BORDER || lx >= PANEL_W - BORDER ||
        ly < BORDER || ly >= PANEL_H - BORDER) return 2.0;

    // inner area
    float innerW = PANEL_W - BORDER * 2.0;
    float innerH = PANEL_H - BORDER * 2.0;
    float ix = lx - BORDER;
    float iy = ly - BORDER;

    float binF   = (ix / innerW) * 255.0;
    float barHeight = texture(histogram, vec2(binF / 255.0, 0.5)).r;
    float barPx = barHeight * innerH;
    int bx = int(fx) % 4;
    int by = int(fy) % 4;
    float dither = (float(bayerMatrix[by * 4 + bx]) + 0.5) / 16.0;

    if (iy < barPx) {
        float barLuma = (iy / innerH);
        return barLuma;
    }

    return -1.0;
}

void main() {
    vec2 uv = v_uv;
    vec2 texSize = vec2(textureSize(webcam, 0));
    float pixelSize = 10.0;
    vec2 px = 1.0 / texSize;

    // Glitch
    float sliceY     = floor(uv.y * 40.0) / 40.0;
    float glitchSeed = floor(time * 8.0);
    float glitchRand = hash(sliceY * 100.0 + glitchSeed);
    float glitchStr  = step(0.92, glitchRand);
    float jitter     = (hash(glitchSeed + sliceY) - 0.5) * 0.03;
    uv.x += jitter * glitchStr;

    // Chromatic aberration
    float aberration = 0.006 + 0.004 * sin(time * 1.3);
    vec2 chunky_uv = floor(uv * texSize / pixelSize) * pixelSize / texSize;
    vec2 chunky_r  = floor((uv + vec2(aberration, 0.0)) * texSize / pixelSize) * pixelSize / texSize;
    vec2 chunky_b  = floor((uv - vec2(aberration, 0.0)) * texSize / pixelSize) * pixelSize / texSize;

    float luma_r = texture(webcam, chunky_r).r;
    float luma_g = texture(webcam, chunky_uv).g;
    float luma_b = texture(webcam, chunky_b).b;
    float luminance = dot(vec3(luma_r, luma_g, luma_b), vec3(0.299, 0.587, 0.114));

    // Bayer dithering
    int bx = int(gl_FragCoord.x / pixelSize) % 4;
    int by = int(gl_FragCoord.y / pixelSize) % 4;
    float threshold = (float(bayerMatrix[by * 4 + bx]) + 0.5) / 16.0;
    float scaledLuma = luminance * 4.0;
    int index = clamp(int(scaledLuma), 0, 3);
    if (fract(scaledLuma) > threshold) index += 1;

    vec3 col = palette[index];

    // Depth fog
    vec2 centered = v_uv - vec2(0.5, 0.5);
    centered.x *= texSize.x / texSize.y;
    float dist = length(centered);

    float breathe  = 0.5 + 0.5 * sin(time * 0.7);
    float fogCore  = exp(-dist * dist * 6.0);
    float fogMid   = exp(-dist * dist * 2.2) * 0.4;
    float fogPulse = fogCore * 0.15 * breathe;

    vec3 fogColor = vec3(0.01, 0.00, 0.05) * fogCore
                  + vec3(0.00, 0.00, 0.03) * fogMid
                  + vec3(0.03, 0.00, 0.08) * fogPulse;

    float isBg = step(luminance, 0.001);
    col = mix(col, fogColor, isBg);

    // Bloom
    float bloom = 0.0;
    float bloomRadius = pixelSize * 5.0;
    vec2 step1 = px * bloomRadius;

    vec2 offsets[12] = vec2[](
        vec2( 1.0,  0.0), vec2(-1.0,  0.0),
        vec2( 0.0,  1.0), vec2( 0.0, -1.0),
        vec2( 0.7,  0.7), vec2(-0.7,  0.7),
        vec2( 0.7, -0.7), vec2(-0.7, -0.7),
        vec2( 1.5,  0.0), vec2(-1.5,  0.0),
        vec2( 0.0,  1.5), vec2( 0.0, -1.5)
    );
    for (int i = 0; i < 12; i++) {
        float s = sampleLuma(uv + offsets[i] * step1, texSize, pixelSize);
        bloom += max(0.0, s - 0.35);
    }
    bloom /= 12.0;
    float pulse = 1.0 + 0.25 * sin(time * 2.1);
    bloom *= 2.8 * pulse;

    vec3 bloomColor = vec3(0.70, 0.40, 1.00) * bloom;
    col = col + bloomColor;

    // Scanline flicker
    float scanline = sin(gl_FragCoord.y * 3.14159 * 1.5 + time * 6.0) * 0.04 + 0.96;
    col *= scanline;

    col = col / (col + 0.5);
    col = clamp(col, 0.0, 1.0);

    // Screen
    float W    = resolution.x;
    float H    = resolution.y;
    float fx   = gl_FragCoord.x;
    float fy   = gl_FragCoord.y;
    float fy_tl = H - fy;

    // Histogram panel
    float histVal = drawHistogram(fx, fy, W, H);
    if (histVal >= 2.0) {
        col = vec3(0.30, 0.10, 0.60);
    } else if (histVal >= 0.0) {
        float barScaled = histVal * 4.0;
        int barIndex = clamp(int(barScaled), 0, 3);
        int bxh = int(fx) % 4;
        int byh = int(fy) % 4;
        float dither = (float(bayerMatrix[byh * 4 + bxh]) + 0.5) / 16.0;
        if (fract(barScaled) > dither) barIndex += 1;
        col = palette[barIndex];
        col += vec3(0.20, 0.05, 0.40) * histVal * 0.5;
        col = clamp(col, 0.0, 1.0);
    } else if (histVal == -1.0) {
        col = vec3(0.02, 0.00, 0.06);
    }

    // REC indicator
    float dot_ = recDot(vec2(fx, fy_tl), vec2(22.0, 22.0), 7.0);
    float blink = step(0.30, fract(time * 0.9));
    dot_ *= blink;

    int strREC[12] = int[](12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    float recTextMask = drawText(fx, fy, 38.0, H - 27.0, 2.0, strREC, 3);
    float recAlpha = clamp(dot_ + recTextMask * 0.9, 0.0, 1.0);
    col = mix(col, vec3(1.0, 0.15, 0.15), recAlpha);

    // Tag
    int strName[12] = int[](0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10);
    float S    = 2.0;
    float tagW = 12.0 * (4.0 * S + 1.0 * S);
    float tagX = W - tagW - 10.0;
    float tagY = 14.0;
    float tagMask  = drawText(fx, fy, tagX, tagY, S, strName, 12);
    float tagAlpha = clamp(tagMask * 0.9, 0.0, 1.0);
    col = mix(col, vec3(0.55, 0.30, 1.00), tagAlpha);

    fragColor = vec4(col, 1.0);
}
