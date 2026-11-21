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
    blowdownIntensity = 0.0,
    flueGasIntensity = 0.0,
    backdraftActive = false
  ): void {
    this.elapsed += dt;
    this.mat.uniforms.uTime.value = this.elapsed;

    const tmpPos = new THREE.Vector3();
    const tmpVel = new THREE.Vector3();
    const rotOnly = new THREE.Matrix4().extractRotation(locoMatrix);

    // 四拍乏汽脉动包络检波 (每转 4 次交替排气脉冲团簇 + 连续引风底烟)
    const chuffPulse = Math.max(0.0, exhaustIntensity - this.avgExhaust * 0.86);
    this.avgExhaust += Math.min(1.0, dt * 8.0) * (exhaustIntensity - this.avgExhaust);

    // 1. 18-25 烟囱口主排汽四拍翻滚白烟羽流 + 电动抽风/通风喷嘴高温湿烟气排出 (X=0.130, Y=0.0, Z=0.093)
    const stackDrive = Math.max(exhaustIntensity, 0.42 * flueGasIntensity);
    if (stackDrive > 0.015) {
      const burstCount = chuffPulse > 0.05 ? 6 : Math.random() < Math.min(1.0, 0.25 + stackDrive * 1.15) ? (exhaustIntensity > 0.02 ? 3 : 1) : 0;
      const effDry = exhaustIntensity > 0.02 ? exhaustDryness : 0.965;
      for (let b = 0; b < burstCount; b++) {
        tmpPos
          .set(0.130 + (Math.random() - 0.5) * 0.006, (Math.random() - 0.5) * 0.006, 0.093)
          .applyMatrix4(locoMatrix);
        const wJet = 0.48 + 0.94 * stackDrive + 0.78 * chuffPulse;
        tmpVel
          .set(-0.11 + (Math.random() - 0.5) * 0.15, (Math.random() - 0.5) * 0.15, wJet)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 1.05 + Math.random() * 0.65, effDry, 0.022 + 0.013 * chuffPulse);
      }
    }

    // 1b. 零引风漫油/倒火时自 12-1 焚口门与炉底倒窜出的热气团 (Back-draft Puff)
    if (backdraftActive && Math.random() < 0.45) {
      tmpPos.set(-0.114, (Math.random() - 0.5) * 0.006, 0.0285).applyMatrix4(locoMatrix);
      tmpVel.set(-0.22 - Math.random() * 0.12, (Math.random() - 0.5) * 0.08, 0.12 + Math.random() * 0.10).applyMatrix4(rotOnly);
      this.spawn(tmpPos, tmpVel, 0.48, 0.92, 0.018);
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

export interface WhistleAcousticInput {
  whistleValve: number;
  boilerPressureKgf: number;
  freqHz?: number;
  splDb?: number;
  drynessX?: number;
  wallTempC?: number;
  sputter?: number;
  jetMach?: number;
  soundSpeedMS?: number;
  distanceM?: number;
  dopplerFactor?: number;
}

/**
 * Aster Hobby JNR D51 11-8 車底隠蔵式・閉端円筒真鍮汽笛 (ø6.2 OD / ø4.30mm ID × 44.4mm, 閉端共鳴長 L_cav=29.2mm,
 * -Z 下向き凝結水排出窓 h_mouth=3.20mm, 有効音響長 L_eff=39.0mm, aster-manual.pdf p.17 Fig.16)
 * 第一原理・二相蒸気気動音響リアルタイム WebAudio シンセサイザ:
 *   1. 四分の一波長閉管奇次定在波モード (1f_1, 3f_1, 5f_1) ＋ 高マッハ数ラビウム非対称ジェットによる微弱 2f_1 偶次倍音；
 *   2. 初吹時の冷温真鍮管壁 (T_wall < T_sat) における蒸気凝結 → Wood-Kieffer 二相湿蒸気音速低下 c_mix(T, x_wh) による
 *      ハスキーな低め立ち上がり (~2.38 kHz) から暖管・下向き窓排液に伴う澄んだ上昇グライド (~3.14 kHz) を忠実再現；
 *   3. 凝結水滴スパッタリング (whistleSputter) による確率的気液二相 AM/FM 変調 ＋ Strouhal エッジトーン帯域通過乱流ジェット音；
 *   4. 3D カメラ距離減衰・空気高域吸収・列車走行時の実ドップラーシフト f_obs = f_1 * dopplerFactor！
 */
export class BrassWhistleAudio {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private distLowpass: BiquadFilterNode | null = null;
  private bodyFormant: BiquadFilterNode | null = null;
  private modeOscs: OscillatorNode[] = [];
  private modeGains: GainNode[] = [];
  private sputterOsc: OscillatorNode | null = null;
  private sputterDepth: GainNode | null = null;
  private sputterFmDepth: GainNode | null = null;
  private edgeBandpass: BiquadFilterNode | null = null;
  private edgeNoiseGain: GainNode | null = null;
  private hissHighpass: BiquadFilterNode | null = null;
  private hissNoiseGain: GainNode | null = null;

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

    // 3D 距離に応じた空気吸収ローパスフィルタ
    const distLp = ctx.createBiquadFilter();
    distLp.type = "lowpass";
    distLp.frequency.value = 11500;
    distLp.Q.value = 0.65;

    // t=0.95mm 真鍮円筒シェル放射フォルマント
    const brassPeak = ctx.createBiquadFilter();
    brassPeak.type = "peaking";
    brassPeak.frequency.value = 3150;
    brassPeak.Q.value = 2.4;
    brassPeak.gain.value = 4.2;

    brassPeak.connect(distLp).connect(master).connect(ctx.destination);
    this.masterGain = master;
    this.distLowpass = distLp;
    this.bodyFormant = brassPeak;

    // 閉端円筒管モード [1f_1 (基本波), 2f_1 (高圧ジェット非線形偶次), 3f_1 (第3奇次倍音), 5f_1 (第5奇次倍音)]
    const modeRatios = [1.0, 2.0, 3.0, 5.0];
    for (let i = 0; i < modeRatios.length; i++) {
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.value = 2850 * modeRatios[i];
      const g = ctx.createGain();
      g.gain.value = 0.0;
      osc.connect(g).connect(brassPeak);
      osc.start();
      this.modeOscs.push(osc);
      this.modeGains.push(g);
    }

    // 初吹二相凝結水滴によるゴロゴロ・スパッタリング AM/FM 変調器 (~29 Hz 気液スラグ脈動)
    const spOsc = ctx.createOscillator();
    spOsc.type = "sawtooth";
    spOsc.frequency.value = 29.5;
    const spAm = ctx.createGain();
    spAm.gain.value = 0.0;
    const spFm = ctx.createGain();
    spFm.gain.value = 0.0;
    spOsc.connect(spAm);
    spOsc.connect(spFm);
    spAm.connect(this.modeGains[0].gain);
    spFm.connect(this.modeOscs[0].frequency);
    spOsc.start();
    this.sputterOsc = spOsc;
    this.sputterDepth = spAm;
    this.sputterFmDepth = spFm;

    // 白色乱流音源 -> (a) Strouhal 哨口エッジトーン狭帯域ジェット音 + (b) 11-1/13-3 蒸気スリット高域ヒス音
    const bufSize = ctx.sampleRate * 2;
    const nBuf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const ch = nBuf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) ch[i] = Math.random() * 2 - 1;
    const nSrc = ctx.createBufferSource();
    nSrc.buffer = nBuf;
    nSrc.loop = true;

    const edgeBp = ctx.createBiquadFilter();
    edgeBp.type = "bandpass";
    edgeBp.frequency.value = 2900;
    edgeBp.Q.value = 3.8;
    const edgeG = ctx.createGain();
    edgeG.gain.value = 0.0;
    nSrc.connect(edgeBp).connect(edgeG).connect(brassPeak);

    const hissHp = ctx.createBiquadFilter();
    hissHp.type = "highpass";
    hissHp.frequency.value = 3800;
    const hissG = ctx.createGain();
    hissG.gain.value = 0.0;
    nSrc.connect(hissHp).connect(hissG).connect(distLp);

    nSrc.start();
    this.edgeBandpass = edgeBp;
    this.edgeNoiseGain = edgeG;
    this.hissHighpass = hissHp;
    this.hissNoiseGain = hissG;
  }

  update(input: WhistleAcousticInput | number, legacyPressureKgf = 0.0): void {
    const inp: WhistleAcousticInput =
      typeof input === "number"
        ? { whistleValve: input, boilerPressureKgf: legacyPressureKgf }
        : input;

    const uVal = Math.max(0.0, Math.min(1.0, inp.whistleValve));
    const pKgf = Math.max(0.0, inp.boilerPressureKgf);
    const anyBlow = uVal > 0.02 && pKgf > 0.015;
    if (anyBlow) this.ensureInit();
    if (!this.ctx || !this.masterGain || this.modeOscs.length < 4) return;

    const now = this.ctx.currentTime;
    if (!anyBlow) {
      this.masterGain.gain.setTargetAtTime(0.0, now, 0.018);
      for (const mg of this.modeGains) mg.gain.setTargetAtTime(0.0, now, 0.015);
      this.edgeNoiseGain?.gain.setTargetAtTime(0.0, now, 0.018);
      this.hissNoiseGain?.gain.setTargetAtTime(0.0, now, 0.018);
      return;
    }

    // 物理テレメトリが渡されていれば 100% 物理ソルバ値を使用、未指定時は同一閉管二相音響式で算出
    const dryX = Math.max(0.52, Math.min(0.995, inp.drynessX ?? 0.94));
    const sput = Math.max(0.0, Math.min(1.0, inp.sputter ?? Math.max(0.0, 1.5 * (0.95 - dryX))));
    const gammaTp = 1.13 + 0.195 * dryX;
    const pr = 1.0 + (pKgf * uVal * 98066.5) / 101325.0;
    const mJet =
      inp.jetMach ??
      Math.min(1.0, Math.sqrt(Math.max(0.0, (2.0 / (gammaTp - 1.0)) * (Math.pow(pr, (gammaTp - 1.0) / gammaTp) - 1.0))));
    const cEff = inp.soundSpeedMS ?? Math.sqrt(dryX * gammaTp * 461.52 * (273.15 + 115.0 + 8.5 * pKgf));
    const f1Raw =
      inp.freqHz !== undefined
        ? inp.freqHz
        : mJet >= 0.085
          ? (cEff / (4.0 * 0.039)) * (0.965 + 0.065 * mJet)
          : 0.0;
    const doppler = Math.max(0.85, Math.min(1.18, inp.dopplerFactor ?? 1.0));
    const f1 = f1Raw * doppler;

    const distM = Math.max(0.18, inp.distanceM ?? 0.45);
    const distAtten = Math.min(1.35, 0.45 / Math.pow(distM, 0.85));
    const splDb = inp.splDb ?? (f1 > 100 ? 50.0 + 38.5 * Math.log10(1.0 + 9.5 * Math.pow(mJet, 2.4)) : 32.0 * mJet);
    const splAmp = Math.min(1.0, Math.max(0.0, Math.pow(10.0, (splDb - 88.0) / 28.0)));
    const targetMaster = Math.min(0.24, 0.21 * splAmp * distAtten);

    this.masterGain.gain.setTargetAtTime(targetMaster, now, 0.022);
    this.distLowpass?.frequency.setTargetAtTime(Math.max(2800, 13500 - 1650 * distM), now, 0.04);

    if (f1 > 150.0) {
      // 閉管共鳴ロックイン状態: 1f_1 主音 + 高圧ラビウム非線形 2f_1 + 3f_1/5f_1 奇次定在波
      const ratios = [1.0, 2.0, 3.0, 5.0];
      const dampWet = 1.0 - 0.55 * sput; // 湿り水膜は高次倍音を減衰させくぐもった初吹音を作る
      const amps = [
        0.52 * (1.0 - 0.22 * sput),
        0.11 * mJet * mJet * dampWet,
        0.22 * (0.45 + 0.55 * mJet) * dampWet,
        0.07 * mJet * dampWet * dampWet,
      ];
      for (let i = 0; i < 4; i++) {
        this.modeOscs[i].frequency.setTargetAtTime(Math.min(18000, f1 * ratios[i]), now, 0.016);
        this.modeGains[i].gain.setTargetAtTime(amps[i], now, 0.020);
      }
      this.bodyFormant?.frequency.setTargetAtTime(Math.min(6000, Math.max(1800, f1 * 1.02)), now, 0.03);
      this.sputterDepth?.gain.setTargetAtTime(0.20 * sput, now, 0.025);
      this.sputterFmDepth?.gain.setTargetAtTime(68.0 * sput, now, 0.025);
      if (this.sputterOsc) {
        this.sputterOsc.frequency.setTargetAtTime(24.0 + 18.0 * mJet, now, 0.04);
      }
      this.edgeBandpass?.frequency.setTargetAtTime(Math.min(9000, f1), now, 0.025);
      this.edgeNoiseGain?.gain.setTargetAtTime(0.16 * mJet * (1.0 + 0.85 * sput), now, 0.025);
      this.hissNoiseGain?.gain.setTargetAtTime(0.07 * mJet + 0.10 * sput, now, 0.025);
    } else {
      // 低圧 (<0.07 kgf/cm², M_jet < 0.085) 閾値未満: 音響発振ロックイン前の純蒸気シュー音のみ!
      for (const mg of this.modeGains) mg.gain.setTargetAtTime(0.0, now, 0.015);
      this.sputterDepth?.gain.setTargetAtTime(0.0, now, 0.02);
      this.sputterFmDepth?.gain.setTargetAtTime(0.0, now, 0.02);
      const fEdgeSub = Math.max(450, Math.min(3500, (0.032 * (cEff * mJet)) / 0.0032));
      this.edgeBandpass?.frequency.setTargetAtTime(fEdgeSub, now, 0.03);
      this.edgeNoiseGain?.gain.setTargetAtTime(0.18 * (0.25 + mJet), now, 0.03);
      this.hissNoiseGain?.gain.setTargetAtTime(0.14 * (0.25 + mJet), now, 0.03);
    }
  }
}
