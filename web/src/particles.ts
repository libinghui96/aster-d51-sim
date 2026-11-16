// web/src/particles.ts
// Aster Hobby JNR D51 — 干度耦合三维 FBM 体积凝结白烟粒子系统
// 还原过热蒸汽「烟囱口近透明热气核 -> 升空绝热冷却凝结翻滚白雾」与四拍乏汽脉动、双安全阀及排液旋塞喷射

import * as THREE from "three";

interface Puff {
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  age: number;
  life: number;
  dryness: number;
  baseSize: number;
  seed: number;
}

export class LiveSteamParticles {
  private readonly maxCount = 1800;
  private puffs: Puff[] = [];
  private cursor = 0;
  private geom: THREE.BufferGeometry;
  private positions: Float32Array;
  private aNormAge: Float32Array;
  private aDryness: Float32Array;
  private aSize: Float32Array;
  private aSeed: Float32Array;
  private mat: THREE.ShaderMaterial;
  private avgExhaust = 0.0;
  private elapsed = 0.0;
  readonly points: THREE.Points;

  constructor() {
    this.positions = new Float32Array(this.maxCount * 3);
    this.aNormAge = new Float32Array(this.maxCount);
    this.aDryness = new Float32Array(this.maxCount);
    this.aSize = new Float32Array(this.maxCount);
    this.aSeed = new Float32Array(this.maxCount);

    for (let i = 0; i < this.maxCount; i++) {
      const seed = Math.random();
      this.puffs.push({
        pos: new THREE.Vector3(0, 0, -20),
        vel: new THREE.Vector3(),
        age: 99,
        life: 1,
        dryness: 0.9,
        baseSize: 0.024,
        seed,
      });
      this.positions[3 * i + 2] = -20;
      this.aNormAge[i] = 1.0;
      this.aDryness[i] = 0.9;
      this.aSize[i] = 0.0;
      this.aSeed[i] = seed;
    }

    this.geom = new THREE.BufferGeometry();
    this.geom.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geom.setAttribute("aNormAge", new THREE.BufferAttribute(this.aNormAge, 1));
    this.geom.setAttribute("aDryness", new THREE.BufferAttribute(this.aDryness, 1));
    this.geom.setAttribute("aSize", new THREE.BufferAttribute(this.aSize, 1));
    this.geom.setAttribute("aSeed", new THREE.BufferAttribute(this.aSeed, 1));

    this.mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: { value: 0.0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
      },
      vertexShader: /* glsl */ `
        attribute float aNormAge;
        attribute float aDryness;
        attribute float aSize;
        attribute float aSeed;
        uniform float uPixelRatio;
        varying float vNormAge;
        varying float vDryness;
        varying float vSeed;
        void main() {
          vNormAge = aNormAge;
          vDryness = aDryness;
          vSeed = aSeed;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          float grow = 0.65 + 2.65 * pow(clamp(aNormAge, 0.0, 1.0), 0.68);
          float px = aSize * grow * uPixelRatio * (560.0 / max(0.12, -mvPos.z));
          gl_PointSize = aNormAge >= 1.0 ? 0.0 : clamp(px, 1.5, 175.0);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        varying float vNormAge;
        varying float vDryness;
        varying float vSeed;

        float hash21(vec2 p) {
          p = fract(p * vec2(234.34, 435.345));
          p += dot(p, p + 34.23);
          return fract(p.x * p.y);
        }
        float noise2(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
            mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x),
            f.y
          );
        }
        void main() {
          if (vNormAge >= 1.0) discard;
          vec2 uv = gl_PointCoord * 2.0 - 1.0;
          float r = length(uv);
          if (r > 1.0) discard;

          float ang = vSeed * 6.283 + vNormAge * (vSeed > 0.5 ? 1.8 : -1.8);
          mat2 rot = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
          vec2 q = rot * uv * 2.2 + vec2(vSeed * 11.3, uTime * 0.65);
          float fbm = 0.62 * noise2(q) + 0.38 * noise2(q * 2.3 + 4.1);

          float mask = smoothstep(1.0, 0.14, r + 0.28 * (fbm - 0.5));
          // 过热高干度乏汽在喷口处呈半透明热气核，离开喷口与环境冷空气掺混绝热膨胀冷却后迅速凝结成翻滚浓白雾团
          float condDelay = clamp((vDryness - 0.86) * 0.85, 0.0, 0.11);
          float condenseIn = smoothstep(0.0, 0.03 + condDelay, vNormAge);
          float fadeOut = pow(1.0 - clamp(vNormAge, 0.0, 1.0), 1.22);
          float wetBoost = mix(1.38, 0.88, clamp((vDryness - 0.82) / 0.16, 0.0, 1.0));

          float alpha = mask * condenseIn * fadeOut * 0.58 * wetBoost;
          if (alpha < 0.008) discard;

          vec3 lightTop = vec3(0.97, 0.99, 1.00);
          vec3 shadowBot = vec3(0.76, 0.82, 0.90);
          vec3 col = mix(shadowBot, lightTop, clamp(0.52 - 0.45 * uv.y + 0.38 * fbm, 0.0, 1.0));
          gl_FragColor = vec4(col, clamp(alpha, 0.0, 0.86));
        }
      `,
    });

    this.points = new THREE.Points(this.geom, this.mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 6;
  }

  private spawn(
    worldOrigin: THREE.Vector3,
    worldVel: THREE.Vector3,
    life: number,
    dryness: number,
    baseSize: number
  ): void {
    const p = this.puffs[this.cursor];
    this.cursor = (this.cursor + 1) % this.maxCount;
    p.pos.copy(worldOrigin);
    p.vel.copy(worldVel);
    p.age = 0.0;
    p.life = life;
    p.dryness = dryness;
    p.baseSize = baseSize;
  }

  update(
    dt: number,
    locoMatrix: THREE.Matrix4,
    exhaustIntensity: number,
    safetyIntensity: number,
    drainIntensity: number,
    exhaustDryness = 0.91,
    whistleIntensity = 0.0,
    sightPurgeIntensity = 0.0,
    blowdownIntensity = 0.0
  ): void {
    this.elapsed += dt;
    this.mat.uniforms.uTime.value = this.elapsed;

    const tmpPos = new THREE.Vector3();
    const tmpVel = new THREE.Vector3();
    const rotOnly = new THREE.Matrix4().extractRotation(locoMatrix);

    // 四拍乏汽脉动包络检波 (每转 4 次交替排气脉冲团簇 + 连续引风底烟)
    const chuffPulse = Math.max(0.0, exhaustIntensity - this.avgExhaust * 0.86);
    this.avgExhaust += Math.min(1.0, dt * 8.0) * (exhaustIntensity - this.avgExhaust);

    // 1. 18-25 烟囱口主排汽四拍翻滚白烟羽流 (X=0.130, Y=0.0, Z=0.093)
    if (exhaustIntensity > 0.015) {
      const burstCount = chuffPulse > 0.05 ? 6 : Math.random() < Math.min(1.0, 0.28 + exhaustIntensity * 1.15) ? 3 : 1;
      for (let b = 0; b < burstCount; b++) {
        tmpPos
          .set(0.130 + (Math.random() - 0.5) * 0.006, (Math.random() - 0.5) * 0.006, 0.093)
          .applyMatrix4(locoMatrix);
        const wJet = 0.56 + 0.92 * exhaustIntensity + 0.78 * chuffPulse;
        tmpVel
          .set(-0.11 + (Math.random() - 0.5) * 0.15, (Math.random() - 0.5) * 0.15, wJet)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 1.15 + Math.random() * 0.65, exhaustDryness, 0.025 + 0.013 * chuffPulse);
      }
    }

    // 2. 28-6 x2 锅筒顶中线纵列双安全阀超压泄放白雾柱 (イラスト 12/28: X=-0.046 / -0.033, Y=0.0, Z=0.0805)
    if (safetyIntensity > 0.1) {
      for (const sx of [-0.046, -0.033]) {
        tmpPos.set(sx + (Math.random() - 0.5) * 0.002, (Math.random() - 0.5) * 0.002, 0.0805).applyMatrix4(locoMatrix);
        tmpVel
          .set(-0.04 + (Math.random() - 0.5) * 0.07, (Math.random() - 0.5) * 0.06, 0.96 + Math.random() * 0.32)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.62, 0.89, 0.019);
      }
    }

    // 3. 1-16 x4 左右气缸前后排液嘴浓白湿汽水平喷流 (イラスト 1: X=0.121/0.131, Y=±0.0308, Z=-0.0135)
    if (drainIntensity > 0.025 && Math.random() < Math.min(1.0, 0.25 + drainIntensity * 1.10)) {
      const xNoz = Math.random() < 0.5 ? 0.121 : 0.131;
      for (const sgn of [-1, 1]) {
        tmpPos.set(xNoz, sgn * 0.0308, -0.0135).applyMatrix4(locoMatrix);
        tmpVel
          .set(0.10 + (Math.random() - 0.5) * 0.09, sgn * (0.40 + Math.random() * 0.18), -0.04 + Math.random() * 0.07)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.54, 0.80, 0.018);
      }
    }

    // 4. 11-8 左火室颊下前伸真实黄铜谐振汽笛喷流 (イラスト 11/12/13: X=-0.102, Y=+0.0204, Z=+0.0232)
    if (whistleIntensity > 0.04) {
      for (let w = 0; w < 2; w++) {
        tmpPos
          .set(-0.102 + (Math.random() - 0.5) * 0.004, +0.0204 + (Math.random() - 0.5) * 0.003, 0.0232)
          .applyMatrix4(locoMatrix);
        tmpVel
          .set(0.16 + Math.random() * 0.12, +0.28 + Math.random() * 0.14, -0.10 + Math.random() * 0.08)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.50, 0.86, 0.019);
      }
    }

    // 5. 1-1/1-2 驾驶室右侧外缘 (-Y=-31.3mm) 11-7 水面计底塞冲洗白汽喷流 (X=-0.1185, Y=-0.0313, Z=+0.0260)
    if (sightPurgeIntensity > 0.03) {
      tmpPos
        .set(-0.1185 + (Math.random() - 0.5) * 0.003, -0.0313 + (Math.random() - 0.5) * 0.003, 0.0260)
        .applyMatrix4(locoMatrix);
      tmpVel
        .set(-0.06 + (Math.random() - 0.5) * 0.08, -0.08 - Math.random() * 0.06, -0.36 - Math.random() * 0.14)
        .applyMatrix4(rotOnly);
      this.spawn(tmpPos, tmpVel, 0.44, 0.78, 0.016);
    }

    // 6. 11-16 驾驶室左床下 (+Y=+18.5mm) 锅炉排水阀高温饱和闪蒸排污水雾 (X=-0.1152, Y=+0.0185, Z=+0.0105)
    if (blowdownIntensity > 0.03) {
      for (let b = 0; b < 2; b++) {
        tmpPos
          .set(-0.1152 + (Math.random() - 0.5) * 0.004, +0.0185 + (Math.random() - 0.5) * 0.004, 0.0105)
          .applyMatrix4(locoMatrix);
        tmpVel
          .set(-0.14 - Math.random() * 0.10, +0.20 + Math.random() * 0.12, -0.30 - Math.random() * 0.12)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.54, 0.74, 0.020);
      }
    }

    for (let i = 0; i < this.maxCount; i++) {
      const p = this.puffs[i];
      if (p.age < p.life) {
        p.age += dt;
        const swirl = 0.06 * Math.sin(this.elapsed * 4.5 + p.seed * 12.5);
        p.pos.addScaledVector(p.vel, dt);
        p.pos.y += swirl * dt;
        p.vel.multiplyScalar(0.962);
        p.vel.z += 0.20 * dt;
        this.positions[3 * i] = p.pos.x;
        this.positions[3 * i + 1] = p.pos.y;
        this.positions[3 * i + 2] = p.pos.z;
        this.aNormAge[i] = p.age / p.life;
        this.aDryness[i] = p.dryness;
        this.aSize[i] = p.baseSize;
      } else {
        this.positions[3 * i + 2] = -20.0;
        this.aNormAge[i] = 1.0;
        this.aSize[i] = 0.0;
      }
    }

    this.geom.attributes.position.needsUpdate = true;
    this.geom.attributes.aNormAge.needsUpdate = true;
    this.geom.attributes.aDryness.needsUpdate = true;
    this.geom.attributes.aSize.needsUpdate = true;
  }
}

export class BrassWhistleAudio {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private osc1: OscillatorNode | null = null;
  private osc2: OscillatorNode | null = null;
  private noiseGain: GainNode | null = null;

  private ensureInit(): void {
    if (this.ctx) {
      if (this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
      }
      return;
    }
    const AudioCtor = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtor) return;
    const ctx: AudioContext = new AudioCtor();
    this.ctx = ctx;

    const master = ctx.createGain();
    master.gain.value = 0.0;
    master.connect(ctx.destination);
    this.masterGain = master;

    // JNR D51 5音阶/双腔黄铜蒸汽笛谐波合成
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 1850;
    filter.connect(master);

    const o1 = ctx.createOscillator();
    o1.type = "sawtooth";
    o1.frequency.value = 349.2; // F4
    const g1 = ctx.createGain();
    g1.gain.value = 0.34;
    o1.connect(g1).connect(filter);
    o1.start();
    this.osc1 = o1;

    const o2 = ctx.createOscillator();
    o2.type = "triangle";
    o2.frequency.value = 440.0; // A4
    const g2 = ctx.createGain();
    g2.gain.value = 0.28;
    o2.connect(g2).connect(filter);
    o2.start();
    this.osc2 = o2;

    // 蒸汽喷嘴湍流嘶声
    const bufSize = ctx.sampleRate;
    const nBuf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const ch = nBuf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) ch[i] = Math.random() * 2 - 1;
    const nSrc = ctx.createBufferSource();
    nSrc.buffer = nBuf;
    nSrc.loop = true;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = 2200;
    bp.Q.value = 1.4;
    const ng = ctx.createGain();
    ng.gain.value = 0.16;
    nSrc.connect(bp).connect(ng).connect(master);
    nSrc.start();
    this.noiseGain = ng;
  }

  update(whistleValve: number, boilerPressureKgf: number): void {
    const active = whistleValve > 0.03 && boilerPressureKgf > 0.18;
    if (active) this.ensureInit();
    if (!this.ctx || !this.masterGain || !this.osc1 || !this.osc2) return;
    const pFactor = Math.min(1.12, Math.max(0.4, boilerPressureKgf / 3.2));
    const targetGain = active ? Math.min(0.22, 0.20 * whistleValve * pFactor) : 0.0;
    const pitchBend = 0.94 + 0.08 * pFactor;
    const now = this.ctx.currentTime;
    this.masterGain.gain.setTargetAtTime(targetGain, now, 0.035);
    this.osc1.frequency.setTargetAtTime(349.2 * pitchBend, now, 0.05);
    this.osc2.frequency.setTargetAtTime(440.0 * pitchBend, now, 0.05);
    if (this.noiseGain) {
      this.noiseGain.gain.setTargetAtTime(0.14 * pFactor, now, 0.04);
    }
  }
}
