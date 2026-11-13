// web/src/particles.ts
import * as THREE from "three";

interface Puff {
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  age: number;
  life: number;
}

export class LiveSteamParticles {
  private readonly maxCount = 900;
  private puffs: Puff[] = [];
  private cursor = 0;
  private geom: THREE.BufferGeometry;
  private positions: Float32Array;
  private avgExhaust = 0.0;
  readonly points: THREE.Points;

  constructor() {
    this.positions = new Float32Array(this.maxCount * 3);
    for (let i = 0; i < this.maxCount; i++) {
      this.puffs.push({ pos: new THREE.Vector3(0, 0, -10), vel: new THREE.Vector3(), age: 99, life: 1 });
      this.positions[3 * i + 2] = -10;
    }
    this.geom = new THREE.BufferGeometry();
    this.geom.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));

    const c = document.createElement("canvas");
    c.width = c.height = 64;
    const ctx = c.getContext("2d")!;
    const grad = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
    grad.addColorStop(0, "rgba(248, 250, 255, 0.62)");
    grad.addColorStop(0.55, "rgba(230, 236, 246, 0.28)");
    grad.addColorStop(1, "rgba(215, 224, 238, 0.0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const mat = new THREE.PointsMaterial({
      size: 0.029,
      map: new THREE.CanvasTexture(c),
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });
    this.points = new THREE.Points(this.geom, mat);
    this.points.frustumCulled = false;
  }

  private spawn(worldOrigin: THREE.Vector3, worldVel: THREE.Vector3, life: number): void {
    const p = this.puffs[this.cursor];
    this.cursor = (this.cursor + 1) % this.maxCount;
    p.pos.copy(worldOrigin);
    p.vel.copy(worldVel);
    p.age = 0;
    p.life = life;
  }

  update(
    dt: number,
    locoMatrix: THREE.Matrix4,
    exhaustIntensity: number,
    safetyIntensity: number,
    drainIntensity: number
  ): void {
    const tmpPos = new THREE.Vector3();
    const tmpVel = new THREE.Vector3();
    const rotOnly = new THREE.Matrix4().extractRotation(locoMatrix);

    // 四拍乏汽脉动包络检波 (每转 4 次交替排气脉冲团簇 + 连续引风底烟)
    const chuffPulse = Math.max(0.0, exhaustIntensity - this.avgExhaust * 0.88);
    this.avgExhaust += Math.min(1.0, dt * 8.0) * (exhaustIntensity - this.avgExhaust);

    if (exhaustIntensity > 0.03) {
      const burstCount = chuffPulse > 0.08 ? 3 : Math.random() < exhaustIntensity * 0.85 ? 1 : 0;
      for (let b = 0; b < burstCount; b++) {
        tmpPos.set(0.130 + (Math.random() - 0.5) * 0.005, (Math.random() - 0.5) * 0.006, 0.096).applyMatrix4(locoMatrix);
        const wJet = 0.50 + 0.72 * exhaustIntensity + 0.55 * chuffPulse;
        tmpVel
          .set(-0.08 + (Math.random() - 0.5) * 0.14, (Math.random() - 0.5) * 0.14, wJet)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.95 + Math.random() * 0.55);
      }
    }

    if (safetyIntensity > 0.1) {
      for (const sgn of [-1, 1]) {
        tmpPos.set(-0.036, sgn * 0.004, 0.084).applyMatrix4(locoMatrix);
        tmpVel.set((Math.random() - 0.5) * 0.08, sgn * 0.05, 0.88 + Math.random() * 0.25).applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.48);
      }
    }

    if (drainIntensity > 0.04 && Math.random() < Math.min(1.0, drainIntensity * 0.95)) {
      for (const sgn of [-1, 1]) {
        tmpPos.set(0.126, sgn * 0.035, -0.010).applyMatrix4(locoMatrix);
        tmpVel.set(0.14, sgn * (0.36 + Math.random() * 0.16), -0.04 + Math.random() * 0.08).applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.42);
      }
    }

    for (let i = 0; i < this.maxCount; i++) {
      const p = this.puffs[i];
      if (p.age < p.life) {
        p.age += dt;
        p.pos.addScaledVector(p.vel, dt);
        p.vel.multiplyScalar(0.962);
        p.vel.z += 0.14 * dt;
        this.positions[3 * i] = p.pos.x;
        this.positions[3 * i + 1] = p.pos.y;
        this.positions[3 * i + 2] = p.pos.z;
      } else {
        this.positions[3 * i + 2] = -20.0;
      }
    }
    this.geom.attributes.position.needsUpdate = true;
  }
}
