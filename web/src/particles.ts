// web/src/particles.ts
import * as THREE from "three";

interface Puff {
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  age: number;
  life: number;
}

export class LiveSteamParticles {
  private readonly maxCount = 600;
  private puffs: Puff[] = [];
  private cursor = 0;
  private geom: THREE.BufferGeometry;
  private positions: Float32Array;
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
    grad.addColorStop(0, "rgba(245, 248, 255, 0.55)");
    grad.addColorStop(1, "rgba(220, 228, 240, 0.0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const mat = new THREE.PointsMaterial({
      size: 0.028,
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

    if (exhaustIntensity > 0.04 && Math.random() < exhaustIntensity * 0.95) {
      tmpPos.set(0.130, (Math.random() - 0.5) * 0.006, 0.096).applyMatrix4(locoMatrix);
      tmpVel.set(-0.08 + (Math.random() - 0.5) * 0.12, (Math.random() - 0.5) * 0.12, 0.55 + 0.65 * exhaustIntensity).applyMatrix4(rotOnly);
      this.spawn(tmpPos, tmpVel, 0.9 + Math.random() * 0.5);
    }

    if (safetyIntensity > 0.1) {
      for (const sgn of [-1, 1]) {
        tmpPos.set(-0.036, sgn * 0.004, 0.084).applyMatrix4(locoMatrix);
        tmpVel.set((Math.random() - 0.5) * 0.08, sgn * 0.05, 0.85 + Math.random() * 0.25).applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.45);
      }
    }

    if (drainIntensity > 0.05 && Math.random() < drainIntensity * 0.85) {
      for (const sgn of [-1, 1]) {
        tmpPos.set(0.126, sgn * 0.035, -0.010).applyMatrix4(locoMatrix);
        tmpVel.set(0.12, sgn * (0.35 + Math.random() * 0.15), -0.05 + Math.random() * 0.08).applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.40);
      }
    }

    for (let i = 0; i < this.maxCount; i++) {
      const p = this.puffs[i];
      if (p.age < p.life) {
        p.age += dt;
        p.pos.addScaledVector(p.vel, dt);
        p.vel.multiplyScalar(0.96);
        p.vel.z += 0.12 * dt;
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