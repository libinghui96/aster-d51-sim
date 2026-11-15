// web/src/thermo1d.ts
// Aster Hobby JNR D51 一维热流体—平面滑阀配气—极限热损伤—RTX 5080 CFD 查找表—AI 自动司炉 (100% 同步 sim/thermo_1d.py)

const PA_PER_KGF_CM2 = 98066.5;
const P_ATM = 101325.0;
const R_STEAM = 461.5;
const GAMMA_STEAM = 1.30;
const CP_WATER = 4186.0;
const CP_COPPER = 385.0;
const RHO_LIQ_WATER = 980.0;

export interface CabControls {
  fuelValve: number;
  burnerLit: boolean;
  chimneyFan: boolean;
  blowerValve: number;
  regulator: number;
  reverser: number;
  drainCock: boolean;
  bypassValve: number;
  whistleValve: number;
  blowdownValve: number;
  fireboxDoor: number;
  fuelCapOpen: boolean;
  waterLidOpen: boolean;
  waterFillValve: number;
  tenderDrainOpen: boolean;
  sightDrainValve: boolean;
  cabRoofOpen: boolean;
  coalCoverOpen: boolean;
  thermalWarp: number;
  gradeDeg: number;
}

export interface CylinderCalibration {
  qFdcL: number;
  qFdcR: number;
  stroke: number;
  qValveCenterL: number;
  qValveCenterR: number;
  steamLap: number;
  exhaustLap: number;
  portWidth: number;
}

export interface SmokeboxLutJson {
  mdot_exhaust_grid_g_s: number[];
  mdot_blower_grid_g_s: number[];
  smokebox_vacuum_pa_table: number[][];
  draft_eff_table: number[][];
  eta_boil_table: number[][];
  superheat_delta_c_table: number[][];
}

export interface SpatialFluidState {
  gpuActive: boolean;
  pinoActive?: boolean;
  gpuName: string;
  totalVoxels: number;
  gpuStepMs: number;
  gpuHz: number;
  stlPassagesVerified: boolean;
  openFlueCount: number;
  airFuelPhi: number;
  flameTiltDeg: number;
  wickTempsC: number[];
  flueGasProfileC: number[];
  steamDrynessDome: number;
  boilingIntensity: number;
  waterFreeZMm: number;
  waterSloshPitchDeg: number;
  waterLayerTempsC: number[];
  crownHotspotsC: number[];
  shProfileTempC: number[];
  shProfileDryness: number[];
  cylLPKgf: number[];
  cylRPKgf: number[];
  valvePortOpenL: number[];
  valvePortOpenR: number[];
  exhaustDryness: number;
  oilFilmUm: number;
  feedwaterVelMS: number;
  pumpChamberPKgf: number;
  checkBallLiftsMm: number[];
  checkBall5LiftsMm?: number[];
  lubeWaterFrac: number;
  pino4OpActive?: boolean;
  tenderBayTempC?: number;
  tenderWaveRmsMm?: number;
  tenderWaveB64?: string;
  midplaneTempB64: string;
}

export interface StepOutput {
  simTime: number;
  boilingProgressPct: number;
  forceL: number;
  forceR: number;
  forcePump: number;
  valveFrictionL: number;
  valveFrictionR: number;
  boilerPressureKgf: number;
  pgBourdonKgf: number;
  turretPressureKgf: number;
  steamChestKgf: number;
  boilerTempC: number;
  superheatTempC: number;
  crownTempC: number;
  crownSubmergedMm: number;
  copperAnnealDamage: number;
  sealHealth: number;
  waterLevelPct: number;
  trueWaterLevelPct: number;
  sightGlassLevelPct: number;
  cylWallTempC: number;
  condensateG: number;
  smokeboxVacuumPa: number;
  lutActive: boolean;
  steamGenGS: number;
  feedwaterGS: number;
  alcoholMl: number;
  tenderWaterMl: number;
  tenderWaterTempC: number;
  returnWaterMlS: number;
  waterFillMlS: number;
  tenderDrainMlS: number;
  tenderOverflowMlS: number;
  mariotteHeadspaceKpa: number;
  mariotteBubbling: boolean;
  mariotteFloodAlarm: boolean;
  whistleSteamGS: number;
  blowdownGS: number;
  sightPurgeGS: number;
  lubricatorOilPct: number;
  wheelRpm: number;
  scaleSpeedKmh: number;
  safetyValveOpen: boolean;
  dryFireAlarm: boolean;
  flashSteamSpike: boolean;
  hydrolockL: boolean;
  hydrolockR: boolean;
  exhaustPlume: number;
  safetyPlume: number;
  drainPlume: number;
  whistlePlume: number;
  sightPurgePlume: number;
  blowdownPlume: number;
  spatial: SpatialFluidState;
}

function satPressurePa(tempC: number): number {
  const tk = Math.max(273.16, Math.min(tempC + 273.15, 640.0));
  const theta = tk / 647.096;
  const tau = 1.0 - theta;
  const lnPi =
    (647.096 / tk) *
    (-7.85951783 * tau +
      1.84408259 * Math.pow(tau, 1.5) -
      11.7866497 * Math.pow(tau, 3.0) +
      22.6807411 * Math.pow(tau, 3.5) -
      15.9618719 * Math.pow(tau, 4.0) +
      1.80122502 * Math.pow(tau, 7.5));
  return 22.064e6 * Math.exp(lnPi);
}

function satTempC(pPa: number): number {
  const pBar = Math.max(0.2, pPa * 1e-5);
  return 1730.63 / (8.07131 - Math.log10(pBar * 750.062)) - 233.426;
}

function latentHeatJKg(tempC: number): number {
  return Math.max(1.5e6, (2501.0 - 2.36 * tempC) * 1e3);
}

function orificeMassFlow(pUp: number, pDown: number, tempK: number, area: number, cd = 0.68): number {
  if (area <= 1e-12 || pUp <= pDown) return 0.0;
  const dp = pUp - pDown;
  const rCrit = Math.pow(2.0 / (GAMMA_STEAM + 1.0), GAMMA_STEAM / (GAMMA_STEAM - 1.0));
  const pEff = Math.max(pDown, pUp * rCrit);
  const ratio = pEff / pUp;
  const term = Math.max(0.0, Math.pow(ratio, 2.0 / GAMMA_STEAM) - Math.pow(ratio, (GAMMA_STEAM + 1.0) / GAMMA_STEAM));
  const psi = Math.sqrt((2.0 * GAMMA_STEAM / (GAMMA_STEAM - 1.0)) * term);
  const smooth = (dp / Math.pow(dp * dp + 250.0 * 250.0, 0.25)) * (dp > 1e-6 ? Math.pow(dp, -0.5) : 0.0);
  const mdot = (cd * area * pUp * psi) / Math.sqrt(R_STEAM * Math.max(290.0, tempK));
  return dp < 1500.0 ? mdot * Math.min(1.0, smooth) : mdot;
}

function bilinearInterp(xGrid: number[], yGrid: number[], table: number[][], x: number, y: number): number {
  const xc = Math.max(xGrid[0], Math.min(xGrid[xGrid.length - 1], x));
  const yc = Math.max(yGrid[0], Math.min(yGrid[yGrid.length - 1], y));
  let ix = 0;
  while (ix < xGrid.length - 2 && xGrid[ix + 1] < xc) ix++;
  let iy = 0;
  while (iy < yGrid.length - 2 && yGrid[iy + 1] < yc) iy++;
  const tx = (xc - xGrid[ix]) / Math.max(1e-12, xGrid[ix + 1] - xGrid[ix]);
  const ty = (yc - yGrid[iy]) / Math.max(1e-12, yGrid[iy + 1] - yGrid[iy]);
  const v00 = table[ix][iy];
  const v10 = table[ix + 1][iy];
  const v01 = table[ix][iy + 1];
  const v11 = table[ix + 1][iy + 1];
  return (1.0 - tx) * ((1.0 - ty) * v00 + ty * v01) + tx * ((1.0 - ty) * v10 + ty * v11);
}

export class AsterD51Thermo1D {
  readonly A_head = Math.PI * Math.pow(0.015 / 2.0, 2);
  readonly A_crank = (Math.PI * (0.015 * 0.015 - 0.003 * 0.003)) / 4.0;
  readonly A_valveTop = 4.6e-5;
  readonly A_valvePortRelief = 1.2e-5;
  readonly A_axlePump = Math.PI * Math.pow(0.0032 / 2.0, 2);
  readonly A_handPump = Math.PI * Math.pow(0.007 / 2.0, 2);
  readonly V_clear: number;
  readonly portBreadth = 0.0055;
  readonly V_boilerTotal = 3.8e-4;
  readonly M_copperShell = 0.395;
  readonly M_copperCrown = 0.065;
  readonly waterMinMl = 140.0;
  readonly waterMaxMl = 280.0;

  T_boilerC = 25.0;
  T_crownC = 25.0;
  M_waterKg = 0.21;
  P_boiler = P_ATM;
  P_steamChest = P_ATM;
  T_superheatC = 25.0;
  T_shWallC = 25.0;
  safetyValveOpen = false;

  alcoholMl = 130.0;
  sumpFuelMl = 0.0;
  readonly tenderWaterCapMl = 340.0;
  tenderWaterMl = 272.0; // Manual p.39: 8分目 (80% of 340ml)
  T_tenderWaterC = 25.0;
  simTime = 0.0;
  readonly lubeTankCapMl = 3.2;
  lubeCondensateMl = 0.64;
  lubricatorOilPct = 80.0;

  P_cylL = [P_ATM, P_ATM];
  P_cylR = [P_ATM, P_ATM];
  T_cylWallC = 25.0;
  M_condL = [1.5e-5, 1.5e-5];
  M_condR = [1.5e-5, 1.5e-5];
  condensateKg = 6.0e-5;

  copperAnnealDamage = 0.0;
  sealHealth = 1.0;
  private sloshPitchDeg = 0.0;
  private sloshPitchVel = 0.0;
  private crownZonesC = [25.0, 25.0, 25.0, 25.0, 25.0, 25.0];
  private flashOverpressurePa = 0.0;
  private handPumpPendingKg = 0.0;
  private pgBourdonKgf = 0.0;
  private sgMeniscusPct = 50.0;
  private lut: SmokeboxLutJson | null = null;

  constructor(public calib: CylinderCalibration, lut?: SmokeboxLutJson | null) {
    this.V_clear = 0.12 * this.A_head * calib.stroke;
    if (lut) this.lut = lut;
    this.resetHotReady();
  }

  setSmokeboxLut(lut: SmokeboxLutJson | null): void {
    this.lut = lut;
  }

  resetColdPrepared(): void {
    this.T_boilerC = 25.0;
    this.T_crownC = 25.0;
    this.M_waterKg = 0.21;
    this.P_boiler = P_ATM;
    this.P_steamChest = P_ATM;
    this.T_superheatC = 25.0;
    this.T_shWallC = 25.0;
    this.safetyValveOpen = false;
    this.alcoholMl = 130.0;
    this.sumpFuelMl = 0.0;
    this.tenderWaterMl = 272.0;
    this.T_tenderWaterC = 25.0;
    this.simTime = 0.0;
    this.lubeCondensateMl = 0.2 * this.lubeTankCapMl;
    this.lubricatorOilPct = 80.0;
    this.P_cylL = [P_ATM, P_ATM];
    this.P_cylR = [P_ATM, P_ATM];
    this.T_cylWallC = 25.0;
    this.M_condL = [8.0e-5, 8.0e-5];
    this.M_condR = [8.0e-5, 8.0e-5];
    this.condensateKg = 3.2e-4;
    this.copperAnnealDamage = 0.0;
    this.sealHealth = 1.0;
    this.sloshPitchDeg = 0.0;
    this.sloshPitchVel = 0.0;
    this.crownZonesC = [25.0, 25.0, 25.0, 25.0, 25.0, 25.0];
    this.flashOverpressurePa = 0.0;
    this.handPumpPendingKg = 0.0;
    this.pgBourdonKgf = 0.0;
    this.sgMeniscusPct = 50.0;
  }

  resetHotReady(): void {
    this.resetColdPrepared();
    this.T_boilerC = 149.6;
    this.T_crownC = 156.0;
    this.crownZonesC = [157.5, 156.8, 156.2, 155.8, 155.2, 154.5];
    this.P_boiler = satPressurePa(this.T_boilerC);
    this.pgBourdonKgf = Math.max(0.0, (this.P_boiler - P_ATM) / PA_PER_KGF_CM2);
    this.T_superheatC = 165.0;
    this.T_shWallC = 172.0;
    this.sumpFuelMl = 3.5;
    this.T_cylWallC = 132.0;
    this.M_condL = [1.5e-5, 1.5e-5];
    this.M_condR = [1.5e-5, 1.5e-5];
    this.condensateKg = 6.0e-5;
  }

  triggerHandPump(strokes = 5): void {
    const volMl = strokes * (this.A_handPump * 0.016 * 1e6) * 1.15;
    const used = Math.min(volMl, this.tenderWaterMl);
    this.tenderWaterMl -= used;
    this.handPumpPendingKg += used * 1e-3;
  }

  private valveAreas(qVal: number, qCenter: number): [number, number, number, number] {
    const xv = qVal - qCenter;
    const { steamLap: es, exhaustLap: ee, portWidth: wp } = this.calib;
    const b = this.portBreadth;
    return [
      b * Math.min(wp, Math.max(0.0, -xv - es)),
      b * Math.min(wp, Math.max(0.0, +xv - ee)),
      b * Math.min(wp, Math.max(0.0, +xv - es)),
      b * Math.min(wp, Math.max(0.0, -xv - ee)),
    ];
  }

  private slideValveFriction(pPair: number[], vValve: number): number {
    const dpChest = Math.max(0.0, this.P_steamChest - P_ATM);
    const dpPorts = 0.5 * (Math.max(0.0, pPair[0] - P_ATM) + Math.max(0.0, pPair[1] - P_ATM));
    const normalLoad = Math.max(0.0, dpChest * this.A_valveTop - dpPorts * this.A_valvePortRelief);
    const muFilm = this.lubricatorOilPct > 2.0 ? 0.042 : 0.135;
    const muEff = muFilm * (2.0 - this.sealHealth);
    return -muEff * normalLoad * Math.tanh(vValve / 0.015) - 0.10 * vValve;
  }

  private stepCylSide(
    dtSub: number,
    pPair: number[],
    mCondPair: number[],
    qPiston: number,
    vPiston: number,
    qFdc: number,
    areas: [number, number, number, number],
    drainOpen: boolean
  ): [number, number, number, number, number, boolean] {
    const [aAdmH, aExhH, aAdmC, aExhC] = areas;
    const tShK = this.T_superheatC + 273.15;
    const aDrain = drainOpen ? 3.2e-7 : 0.0;
    const aBlowby = 2.2e-7 * (1.0 - this.sealHealth);

    const sP = Math.min(this.calib.stroke, Math.max(0.0, qFdc - qPiston));
    const dsDt = -vPiston;
    const vHTot = this.V_clear + this.A_head * sP;
    const vCTot = this.V_clear + this.A_crank * (this.calib.stroke - sP);

    const volWH = mCondPair[0] / RHO_LIQ_WATER;
    const volWC = mCondPair[1] / RHO_LIQ_WATER;
    const vHGas = Math.max(0.08 * this.V_clear, vHTot - volWH);
    const vCGas = Math.max(0.08 * this.V_clear, vCTot - volWC);

    const exhChokeH = !drainOpen && volWH > 0.48 * this.V_clear ? 0.14 : 1.0;
    const exhChokeC = !drainOpen && volWC > 0.48 * this.V_clear ? 0.14 : 1.0;

    const mInH = orificeMassFlow(this.P_steamChest, pPair[0], tShK, aAdmH);
    const mOutH = orificeMassFlow(pPair[0], P_ATM, tShK, aExhH * exhChokeH);
    const mDrnH = orificeMassFlow(pPair[0], P_ATM, tShK, aDrain);

    const mInC = orificeMassFlow(this.P_steamChest, pPair[1], tShK, aAdmC);
    const mOutC = orificeMassFlow(pPair[1], P_ATM, tShK, aExhC * exhChokeC);
    const mDrnC = orificeMassFlow(pPair[1], P_ATM, tShK, aDrain);

    const mBlow =
      pPair[0] >= pPair[1]
        ? orificeMassFlow(pPair[0], pPair[1], tShK, aBlowby)
        : -orificeMassFlow(pPair[1], pPair[0], tShK, aBlowby);

    const subcoolC = Math.max(0.0, satTempC(this.P_steamChest) - this.T_cylWallC - 6.0);
    const condFrac = Math.min(0.88, 0.024 * subcoolC);
    const mdotCondGen = 3.2 * condFrac * (mInH + mInC);
    let dmWH = 3.2 * condFrac * mInH;
    let dmWC = 3.2 * condFrac * mInC;
    let purgedLiqRate = 0.0;
    if (drainOpen) {
      const dpHBar = Math.max(0.25, (pPair[0] - P_ATM) * 1e-5);
      const dpCBar = Math.max(0.25, (pPair[1] - P_ATM) * 1e-5);
      const purgeH = mCondPair[0] > 0 ? 0.006 * dpHBar : 0.0;
      const purgeC = mCondPair[1] > 0 ? 0.006 * dpCBar : 0.0;
      dmWH -= purgeH;
      dmWC -= purgeC;
      purgedLiqRate = purgeH + purgeC;
    } else {
      if (aExhH > 1e-8 && volWH > 0.85 * this.V_clear) dmWH -= 0.000015 * (mOutH / Math.max(1e-6, mOutH + 2e-4));
      if (aExhC > 1e-8 && volWC > 0.85 * this.V_clear) dmWC -= 0.000015 * (mOutC / Math.max(1e-6, mOutC + 2e-4));
    }

    const maxCond = 3.2 * this.V_clear * RHO_LIQ_WATER;
    mCondPair[0] = Math.max(0.0, Math.min(maxCond, mCondPair[0] + dmWH * dtSub));
    mCondPair[1] = Math.max(0.0, Math.min(maxCond, mCondPair[1] + dmWC * dtSub));

    const dpH =
      ((GAMMA_STEAM * R_STEAM * tShK) / vHGas) * ((1.0 - 0.5 * condFrac) * mInH - mOutH - mDrnH - mBlow) -
      ((GAMMA_STEAM * pPair[0]) / vHGas) * (this.A_head * dsDt);
    const dpC =
      ((GAMMA_STEAM * R_STEAM * tShK) / vCGas) * ((1.0 - 0.5 * condFrac) * mInC - mOutC - mDrnC + mBlow) -
      ((GAMMA_STEAM * pPair[1]) / vCGas) * (-this.A_crank * dsDt);

    const pRelief = Math.max(this.P_steamChest + 25000.0, P_ATM + 35000.0);
    pPair[0] = Math.min(pRelief, Math.max(0.85 * P_ATM, pPair[0] + dpH * dtSub));
    pPair[1] = Math.min(pRelief, Math.max(0.85 * P_ATM, pPair[1] + dpC * dtSub));

    let fHydroX = 0.0;
    let hydrolock = false;
    if (!drainOpen && volWH > 0.48 * this.V_clear) {
      const overH = volWH - 0.34 * vHTot;
      if (overH > 0.0) {
        hydrolock = true;
        if (vPiston > -0.002) {
          fHydroX -= Math.min(420.0, 3.5e8 * overH * (vPiston >= 0.0 ? 1.0 : 0.15) + 3200.0 * Math.max(0.0, vPiston));
        }
      }
    }
    if (!drainOpen && volWC > 0.48 * this.V_clear) {
      const overC = volWC - 0.34 * vCTot;
      if (overC > 0.0) {
        hydrolock = true;
        if (vPiston < 0.002) {
          fHydroX += Math.min(420.0, 3.5e8 * overC * (vPiston <= 0.0 ? 1.0 : 0.15) + 3200.0 * Math.max(0.0, -vPiston));
        }
      }
    }

    return [mInH + mInC, mOutH + mOutC, mDrnH + mDrnC + 0.15 * purgedLiqRate, mdotCondGen, fHydroX, hydrolock];
  }

  step(
    dt: number,
    qPL: number, vPL: number, qVL: number,
    qPR: number, vPR: number, qVR: number,
    omega3: number, vPumpRam: number,
    ctrl: CabControls,
    vVL = 0.0,
    vVR = 0.0,
    pitchRad = 0.0,
    accelX = 0.0
  ): StepOutput {
    const dtTh = dt * Math.max(1.0, ctrl.thermalWarp);
    this.simTime += dt;

    let mariotteBubbling = false;
    let mariotteFloodAlarm = false;
    if (ctrl.fuelValve > 0.01 && this.alcoholMl > 0) {
      if (ctrl.fuelCapOpen && ctrl.fuelValve > 0.05) {
        // 36-5 气密旋盖在开阀时拧开: 马氏瓶负压破坏，酒精静压溢流灌满 34-2 サンプ
        mariotteFloodAlarm = true;
        if (this.sumpFuelMl < 6.8) {
          const flood = Math.min(this.alcoholMl, 1.85 * ctrl.fuelValve * dtTh, 6.8 - this.sumpFuelMl);
          this.alcoholMl -= flood;
          this.sumpFuelMl += flood;
        }
      } else if (this.sumpFuelMl < 3.5) {
        const refill = Math.min(this.alcoholMl, 0.45 * ctrl.fuelValve * dtTh, 3.5 - this.sumpFuelMl);
        this.alcoholMl -= refill;
        this.sumpFuelMl += refill;
        mariotteBubbling = refill > 1e-6;
      }
    }
    if (this.sumpFuelMl <= 1e-4) ctrl.burnerLit = false;

    const aReg = 3.2e-6 * Math.pow(ctrl.regulator, 1.35);
    const tBoilK = this.T_boilerC + 273.15;
    const hasSteam = this.M_waterKg > 0.012 && this.P_boiler > P_ATM;
    const mdotReg = hasSteam ? orificeMassFlow(this.P_boiler, this.P_steamChest, tBoilK, aReg) : 0.0;

    const areasL = this.valveAreas(qVL, this.calib.qValveCenterL);
    const areasR = this.valveAreas(qVR, this.calib.qValveCenterR);

    const nSub = 4;
    const dtSub = dt / nSub;
    let mdotIn = 0.0, mdotExh = 0.0, mdotDrn = 0.0, mdotCondTot = 0.0, fHydroLAcc = 0.0, fHydroRAcc = 0.0;
    let hydrolockL = false, hydrolockR = false;

    for (let i = 0; i < nSub; i++) {
      const [inL, exhL, drnL, cgL, fhL, hlL] = this.stepCylSide(
        dtSub, this.P_cylL, this.M_condL, qPL, vPL, this.calib.qFdcL, areasL, ctrl.drainCock
      );
      const [inR, exhR, drnR, cgR, fhR, hlR] = this.stepCylSide(
        dtSub, this.P_cylR, this.M_condR, qPR, vPR, this.calib.qFdcR, areasR, ctrl.drainCock
      );
      const inTot = inL + inR;
      const mdotGlandLeak = orificeMassFlow(
        this.P_steamChest, P_ATM, this.T_superheatC + 273.15, 1.5e-7 * (1.0 - this.sealHealth)
      );
      const dpSc = ((R_STEAM * (this.T_superheatC + 273.15)) / 2.8e-5) * (mdotReg - inTot - mdotGlandLeak);
      this.P_steamChest = Math.min(this.P_boiler, Math.max(P_ATM, this.P_steamChest + dpSc * dtSub));
      mdotIn += inTot / nSub;
      mdotExh += (exhL + exhR) / nSub;
      mdotDrn += (drnL + drnR) / nSub;
      mdotCondTot += (cgL + cgR) / nSub;
      fHydroLAcc += fhL / nSub;
      fHydroRAcc += fhR / nSub;
      hydrolockL = hydrolockL || hlL;
      hydrolockR = hydrolockR || hlR;
    }
    this.condensateKg = this.M_condL[0] + this.M_condL[1] + this.M_condR[0] + this.M_condR[1];

    if (mdotIn > 1e-6 || mdotCondTot > 1e-7) {
      const qLatCyl = Math.min(920.0, mdotCondTot * 2.1e6);
      const qConvCyl = Math.max(0.0, (this.T_superheatC - this.T_cylWallC) * 2.2);
      this.T_cylWallC = Math.min(this.T_superheatC, this.T_cylWallC + ((qLatCyl + qConvCyl) / 48.0) * dtTh);
    } else {
      this.T_cylWallC += (25.0 - this.T_cylWallC) * 0.002 * dtTh;
    }

    // 4-2 卧式蒸汽置换式注油器: 蒸汽受前车架冷气流冷却凝结沉水，等体积顶出蒸汽油汇入 13-4/7-9 给汽三通
    if (this.P_steamChest > P_ATM + 2500.0 && this.lubeCondensateMl < this.lubeTankCapMl) {
      const tSatLubeC = satTempC(this.P_steamChest);
      const airCool = 1.0 + 0.022 * Math.min(40.0, Math.abs(omega3));
      const pulse = 0.35 + 0.65 * Math.min(1.0, mdotIn / 1.5e-4);
      const qLubeW = 0.0115 * airCool * pulse * Math.max(0.0, tSatLubeC - 25.0);
      const mdotLubeCond = qLubeW / latentHeatJKg(tSatLubeC);
      const dvDispMl = (mdotLubeCond / 985.0) * 1e6 * dtTh;
      this.lubeCondensateMl = Math.min(this.lubeTankCapMl, this.lubeCondensateMl + dvDispMl);
      this.lubricatorOilPct = Math.max(0.0, 100.0 * (1.0 - this.lubeCondensateMl / this.lubeTankCapMl));
    }

    let sealWear = 0.0;
    if (this.T_superheatC > 225.0) sealWear += 0.0008 * (this.T_superheatC - 225.0);
    if (this.lubricatorOilPct < 1.0 && Math.abs(vPL) + Math.abs(vPR) > 0.08) sealWear += 0.0025;
    if (hydrolockL || hydrolockR) sealWear += 0.015;
    this.sealHealth = Math.max(0.25, this.sealHealth - sealWear * dtTh);

    const lube = this.lubricatorOilPct > 2.0 ? 1.0 : 2.8;
    const waterDrag = !ctrl.drainCock
      ? 115.0 * Math.min(1.0, Math.max(0.0, this.condensateKg - 0.00045) / 0.00085)
      : 0.8;
    const calcForce = (p: number[], v: number, fHydro: number) =>
      (p[1] - P_ATM) * this.A_crank - (p[0] - P_ATM) * this.A_head - lube * (1.35 * Math.tanh(v * 80.0) + waterDrag * v) + fHydro;

    const forceL = calcForce(this.P_cylL, vPL, fHydroLAcc);
    const forceR = calcForce(this.P_cylR, vPR, fHydroRAcc);
    const valveFrictionL = this.slideValveFriction(this.P_cylL, vVL);
    const valveFrictionR = this.slideValveFriction(this.P_cylR, vVR);

    const pGauge = Math.max(0.0, (this.P_boiler - P_ATM) / PA_PER_KGF_CM2);
    if (pGauge >= 4.0) this.safetyValveOpen = true;
    else if (pGauge <= 3.6) this.safetyValveOpen = false;

    const aBrazeLeak = 0.9e-6 * Math.max(0.0, this.copperAnnealDamage - 0.55);
    const mdotBlw = hasSteam ? orificeMassFlow(this.P_boiler, P_ATM, tBoilK, 0.16e-6 * ctrl.blowerValve) : 0.0;
    const mdotWhistle = hasSteam ? orificeMassFlow(this.P_boiler, P_ATM, tBoilK, 0.26e-6 * Math.max(0.0, ctrl.whistleValve)) : 0.0;
    const mdotSv = hasSteam && this.safetyValveOpen ? orificeMassFlow(this.P_boiler, P_ATM, tBoilK, 3.6e-6) : 0.0;
    const mdotBraze = hasSteam ? orificeMassFlow(this.P_boiler, P_ATM, tBoilK, aBrazeLeak) : 0.0;
    let mdotBlowdownLiq = 0.0;
    if (ctrl.blowdownValve > 0.01 && this.M_waterKg > 0.02) {
      const dpBd = Math.max(6000.0, this.P_boiler - P_ATM);
      mdotBlowdownLiq = 0.62 * (0.45e-6 * ctrl.blowdownValve) * Math.sqrt(2.0 * RHO_LIQ_WATER * dpBd);
    }
    let mdotSightPurge = 0.0;
    if (ctrl.sightDrainValve && this.M_waterKg > 0.02) {
      const dpSg = Math.max(5000.0, this.P_boiler - P_ATM);
      mdotSightPurge = 0.58 * 0.18e-6 * Math.sqrt(2.0 * RHO_LIQ_WATER * dpSg);
    }

    let draftEff: number;
    let etaBoilEff: number;
    let shDeltaC: number;
    let smokeboxVacPa: number;
    if (this.lut) {
      const exhGs = mdotExh * 1000.0;
      const blwGs = mdotBlw * 1000.0;
      const gE = this.lut.mdot_exhaust_grid_g_s;
      const gB = this.lut.mdot_blower_grid_g_s;
      const lutDraft = bilinearInterp(gE, gB, this.lut.draft_eff_table, exhGs, blwGs);
      const lutEta = bilinearInterp(gE, gB, this.lut.eta_boil_table, exhGs, blwGs);
      const lutSh = bilinearInterp(gE, gB, this.lut.superheat_delta_c_table, exhGs, blwGs);
      smokeboxVacPa = bilinearInterp(gE, gB, this.lut.smokebox_vacuum_pa_table, exhGs, blwGs);
      if (ctrl.chimneyFan) {
        draftEff = Math.min(1.32, Math.max(lutDraft, 0.96));
        etaBoilEff = Math.max(lutEta, 0.48);
        shDeltaC = Math.max(lutSh, 26.0);
        smokeboxVacPa = Math.max(smokeboxVacPa, 78.0);
      } else {
        draftEff = lutDraft;
        etaBoilEff = lutEta;
        shDeltaC = lutSh;
      }
    } else {
      const draftIdx = (ctrl.chimneyFan ? 1.15 : 0.0) + 18000.0 * mdotBlw + 2600.0 * mdotExh;
      draftEff = Math.min(1.3, 0.12 + 0.92 * Math.tanh(draftIdx));
      etaBoilEff = (ctrl.chimneyFan ? 0.50 : 0.38) * draftEff;
      shDeltaC = 28.0 * draftEff;
      smokeboxVacPa = 115.0 * Math.max(0.0, draftEff - 0.12);
    }

    // 12-2 火室门开启时冷二次风短路进入火室，降低烟室负压与换热效率
    if (ctrl.fireboxDoor > 0.01) {
      const fd = Math.min(1.0, ctrl.fireboxDoor);
      etaBoilEff *= 1.0 - 0.34 * fd;
      smokeboxVacPa *= 1.0 - 0.42 * fd;
    }

    const thetaGradeOnly = Math.max(-0.30, Math.min(0.30, pitchRad + (ctrl.gradeDeg * Math.PI) / 180.0));
    const wickHeadFactor = Math.max(0.75, Math.min(1.22, 1.0 - 0.32 * Math.tan(thetaGradeOnly)));
    const floodMult = mariotteFloodAlarm || this.sumpFuelMl > 4.2 ? 1.38 : 1.0;

    let qComb = 0.0;
    if (ctrl.burnerLit && this.sumpFuelMl > 0) {
      const usedMl = Math.min(
        this.sumpFuelMl,
        0.106 * (0.58 + 0.42 * draftEff) * wickHeadFactor * floodMult * dtTh
      );
      this.sumpFuelMl -= usedMl;
      const effPenalty = floodMult > 1.05 ? 0.80 : 1.0;
      qComb = etaBoilEff * effPenalty * 26.8e6 * ((usedMl / dtTh) * 789e-6);
    }

    // 17-5 双 U 四程过热器 (上排 2 根 ø11.5mm 烟管内 504mm/47.5cm² SUS304 管壁对流热平衡)
    const tFlueGasC = this.T_boilerC + (ctrl.burnerLit ? 1.32 * shDeltaC : 0.0);
    const gGasToSh = 4.2 * (0.25 + 0.75 * draftEff);
    const mdotCpSteam = mdotReg * 2080.0;
    let qShToSteam: number;
    let targetShC: number;
    if (mdotCpSteam > 1e-5) {
      const uShIn = 1.8 + 4200.0 * mdotReg;
      const epsSh = 1.0 - Math.exp(-Math.min(12.0, uShIn / mdotCpSteam));
      qShToSteam = epsSh * mdotCpSteam * (this.T_shWallC - this.T_boilerC);
      targetShC = this.T_boilerC + epsSh * (this.T_shWallC - this.T_boilerC);
    } else {
      qShToSteam = 0.25 * (this.T_shWallC - this.T_boilerC);
      targetShC = 0.35 * this.T_boilerC + 0.65 * this.T_shWallC;
    }
    const dTShWall = (gGasToSh * (tFlueGasC - this.T_shWallC) - qShToSteam) / 7.0;
    this.T_shWallC = Math.max(25.0, Math.min(360.0, this.T_shWallC + dTShWall * dtTh));
    this.T_superheatC += Math.min(1.0, 2.5 * dtTh) * (targetShC - this.T_superheatC);

    // イラスト 4 / 6 / 31 / 34: 共管双泵给水与 4-17 旁通阀 -> 31-14 鹅颈回水立管耦合
    const vPump = this.tenderWaterMl > 0.5 ? 0.85 * this.A_axlePump * Math.max(0.0, -vPumpRam) : 0.0;
    const dpCheck = Math.max(8000.0, this.P_boiler - P_ATM);
    const vBypCap = 0.65 * (1.6e-7 * Math.pow(ctrl.bypassValve, 1.4)) * Math.sqrt((2.0 * dpCheck) / 1000.0);
    let vToBoil = Math.max(0.0, vPump - vBypCap);
    const usedWaterMl = Math.min(this.tenderWaterMl, vToBoil * 1e6 * dtTh);
    this.tenderWaterMl -= usedWaterMl;
    vToBoil = (usedWaterMl * 1e-6) / dtTh;
    const returnAxleMlS = Math.max(0.0, vPump - vToBoil) * 1e6;

    let mdotFwHand = 0.0;
    let returnHandMlS = 0.0;
    if (this.handPumpPendingKg > 1e-7) {
      const injKg = Math.min(this.handPumpPendingKg, Math.max(2.5e-5, this.handPumpPendingKg * Math.min(1.0, dtTh / 0.35)));
      this.handPumpPendingKg -= injKg;
      const bypOpen = ctrl.bypassValve > 0.06 ? Math.pow(ctrl.bypassValve, 1.15) * 1.25 : 0.0;
      const divertFrac = bypOpen > 0.02 ? (0.94 * bypOpen) / (0.08 + bypOpen) : 0.0;
      const retKg = injKg * divertFrac;
      const boilKg = injKg - retKg;
      this.tenderWaterMl = Math.min(this.tenderWaterCapMl, this.tenderWaterMl + retKg * 1000.0);
      mdotFwHand = boilKg / dtTh;
      returnHandMlS = (retKg * 1000.0) / dtTh;
    } else {
      this.handPumpPendingKg = 0.0;
    }
    const returnWaterMlS = returnAxleMlS + returnHandMlS;
    // 35-1 常开注水口 加水 (Water Pouring) 与 31-11/31-9 底部排水阀 放水 (Torricelli Drain)
    const waterFillMlS = 18.0 * Math.max(0.0, Math.min(1.0, ctrl.waterFillValve || 0.0));
    if (waterFillMlS > 1e-4) {
      const dFill = waterFillMlS * dtTh;
      const prevW = Math.max(1.0, this.tenderWaterMl);
      this.tenderWaterMl += dFill;
      this.T_tenderWaterC = (prevW * this.T_tenderWaterC + dFill * 22.0) / (prevW + dFill);
    }
    let tenderDrainMlS = 0.0;
    if (ctrl.tenderDrainOpen && this.tenderWaterMl > 0.05) {
      const hWaterM = Math.max(0.002, 0.042 * Math.min(1.0, this.tenderWaterMl / this.tenderWaterCapMl));
      tenderDrainMlS = 0.64 * (Math.PI * Math.pow(0.00175, 2)) * Math.sqrt(2.0 * 9.81 * hWaterM) * 1e6;
      const dDrain = Math.min(this.tenderWaterMl, tenderDrainMlS * dtTh);
      this.tenderWaterMl -= dDrain;
      tenderDrainMlS = dDrain / dtTh;
    }
    let tenderOverflowMlS = 0.0;
    if (this.tenderWaterMl > this.tenderWaterCapMl) {
      const excess = this.tenderWaterMl - this.tenderWaterCapMl;
      this.tenderWaterMl = this.tenderWaterCapMl;
      tenderOverflowMlS = excess / dtTh;
    }

    if (returnWaterMlS > 1e-4 && this.tenderWaterMl > 1.0) {
      const tRetC = Math.min(72.0, this.T_tenderWaterC + 0.22 * Math.max(0.0, this.T_boilerC - this.T_tenderWaterC));
      const mixFrac = Math.min(0.25, (returnWaterMlS * dtTh) / Math.max(5.0, this.tenderWaterMl));
      this.T_tenderWaterC += mixFrac * (tRetC - this.T_tenderWaterC);
    }
    this.T_tenderWaterC = Math.max(22.0, this.T_tenderWaterC - 0.0012 * (this.T_tenderWaterC - 25.0) * dtTh);
    const mdotFw = vToBoil * 1000.0 + mdotFwHand;

    let forcePump = 0.0;
    if (vPumpRam < -1e-4 && this.tenderWaterMl > 0.5) {
      const pLine = vToBoil > 0.0 ? dpCheck : dpCheck * Math.pow(vPump / Math.max(1e-12, vBypCap), 2);
      forcePump = pLine * this.A_axlePump;
    }

    // O4-1 / O4-2: 坡道假水位 + 紫铜火室顶板干烧退火与冷水闪蒸
    const thetaEff = Math.max(-0.30, Math.min(0.30, pitchRad + (ctrl.gradeDeg * Math.PI) / 180.0 + Math.atan2(accelX, 9.81)));
    const tanTh = Math.tan(thetaEff);
    const waterMl = this.M_waterKg * 1000.0;
    const trueLevelPct = (100.0 * (waterMl - this.waterMinMl)) / (this.waterMaxMl - this.waterMinMl);
    const apparentLevelPct = trueLevelPct + ((68.0 * tanTh) / 14.7) * 100.0;

    const hCrownStaticMm = (waterMl - 145.0) * (11.5 / 65.0);
    const crownSubmergedMm = hCrownStaticMm - 24.0 * Math.abs(tanTh);
    const fWet = Math.max(0.0, Math.min(1.0, crownSubmergedMm / 4.0));

    const qFireCrown = 0.58 * qComb;
    const qFireTubes = 0.42 * qComb;
    const gCrownWater = 22.0 * fWet + 0.36 * (1.0 - fWet);
    const qCrownToWater = gCrownWater * (this.T_crownC - this.T_boilerC);
    const qCrownRadLoss = 0.08 * Math.max(0.0, this.T_crownC - 25.0);

    let flashSpike = false;
    let qFlashW = 0.0;
    if (this.T_crownC > 215.0 && fWet < 0.45 && mdotFw > 1.5e-4) {
      flashSpike = true;
      qFlashW = Math.min(2800.0, 26.0 * (this.T_crownC - this.T_boilerC) * Math.min(4.0, mdotFw / 0.0012));
      const mdotFlashVap = qFlashW / latentHeatJKg(this.T_boilerC);
      const vVapNow = Math.max(4e-5, this.V_boilerTotal - this.M_waterKg / 960.0);
      const dpFlash = ((R_STEAM * (this.T_boilerC + 273.15)) / vVapNow) * (mdotFlashVap * dtTh);
      this.flashOverpressurePa = Math.min(1.8 * PA_PER_KGF_CM2, this.flashOverpressurePa + dpFlash);
    } else {
      this.flashOverpressurePa *= Math.exp(-1.2 * dtTh);
    }

    const cCrown = this.M_copperCrown * CP_COPPER;
    const dTCrown = (qFireCrown - qCrownToWater - qCrownRadLoss - qFlashW) / cCrown;
    this.T_crownC = Math.max(25.0, Math.min(780.0, this.T_crownC + dTCrown * dtTh));
    if (this.T_crownC > 340.0) {
      const dmgRate = 0.0018 * Math.pow((this.T_crownC - 340.0) / 80.0, 1.4);
      this.copperAnnealDamage = Math.min(1.0, this.copperAnnealDamage + dmgRate * dtTh);
    }

    const qNetWater = qFireTubes * (0.25 + 0.75 * fWet) + qCrownToWater + qFlashW - 0.42 * (this.T_boilerC - 25.0);
    const mdotSteamOut = mdotReg + mdotBlw + mdotWhistle + mdotSv + mdotBraze;
    const hFg = latentHeatJKg(this.T_boilerC);
    const vGas = Math.max(2e-5, this.V_boilerTotal - this.M_waterKg / 960.0);

    if (this.T_boilerC < 99.9 && !flashSpike) {
      const cTot = this.M_waterKg * CP_WATER + this.M_copperShell * CP_COPPER;
      this.T_boilerC += ((qNetWater - mdotFw * CP_WATER * (this.T_boilerC - this.T_tenderWaterC)) / Math.max(60.0, cTot)) * dtTh;
      this.M_waterKg = Math.max(0.0, Math.min(0.35, this.M_waterKg + (mdotFw - mdotBlowdownLiq - mdotSightPurge) * dtTh));
      this.P_boiler = P_ATM;
    } else {
      const p1 = satPressurePa(this.T_boilerC);
      const p2 = satPressurePa(this.T_boilerC + 0.1);
      const drhoDt = (p2 / (R_STEAM * (tBoilK + 0.1)) - p1 / (R_STEAM * tBoilK)) / 0.1;
      const cEff = this.M_waterKg * CP_WATER + this.M_copperShell * CP_COPPER + vGas * hFg * drhoDt;
      const dTdt =
        (qNetWater - mdotSteamOut * hFg - mdotFw * CP_WATER * (this.T_boilerC - this.T_tenderWaterC)) /
        Math.max(85.0, cEff);
      this.T_boilerC = Math.max(25.0, Math.min(198.0, this.T_boilerC + dTdt * dtTh));
      this.M_waterKg = Math.max(
        0.0,
        Math.min(0.35, this.M_waterKg + (mdotFw - mdotBlowdownLiq - mdotSightPurge - mdotSteamOut - vGas * drhoDt * dTdt) * dtTh)
      );
      this.P_boiler = this.M_waterKg > 0.012 ? Math.max(P_ATM, satPressurePa(this.T_boilerC) + this.flashOverpressurePa) : P_ATM;
    }

    const rpm = (omega3 / (2.0 * Math.PI)) * 60.0;
    const speedKmh = Math.abs(omega3) * (0.5 * 0.0438) * 32.0 * 3.6;
    const dryAlarm = (crownSubmergedMm <= 0.5 && ctrl.burnerLit) || this.T_crownC >= 240.0;

    // -------------------------------------------------------------------------
    // 五大零件级三维流体与相变空间场 (与本地 RTX 5080 thermo_3d_gpu.py 1:1 契约同步)
    // -------------------------------------------------------------------------
    const sloshTargetDeg = (thetaEff * 180.0) / Math.PI;
    const omegaSlosh = 13.5;
    const sloshAcc = omegaSlosh * omegaSlosh * (sloshTargetDeg - this.sloshPitchDeg) - 3.6 * this.sloshPitchVel;
    this.sloshPitchVel += sloshAcc * dt;
    this.sloshPitchDeg = Math.max(-18.0, Math.min(18.0, this.sloshPitchDeg + this.sloshPitchVel * dt));

    const litActive = ctrl.burnerLit && this.sumpFuelMl > 1e-4;
    const airSupply = Math.max(0.18, 0.24 + 0.88 * draftEff);
    const fuelDemand = litActive ? (0.58 + 0.42 * draftEff) * wickHeadFactor * floodMult : 0.0;
    const airFuelPhi = litActive ? Math.max(0.55, Math.min(1.65, fuelDemand / airSupply)) : 0.0;
    const flameTiltDeg = litActive ? Math.min(52.0, 5.0 + 38.0 * Math.max(0.0, draftEff - 0.10)) : 0.0;
    const tFlameBase = litActive
      ? 670.0 + 195.0 * Math.min(1.18, 1.0 / Math.max(0.68, airFuelPhi)) * Math.min(1.15, 0.65 + 0.45 * draftEff)
      : 28.0;
    const wickTempsC = [tFlameBase - 14.0, tFlameBase + 2.0, tFlameBase + 14.0];

    const flueGasProfileC: number[] = [];
    const tSmokeboxGas = this.T_boilerC + (litActive ? 0.68 * shDeltaC : 0.0);
    for (let k = 0; k < 8; k++) {
      const s = k / 7.0;
      flueGasProfileC.push(litActive ? tSmokeboxGas + (tFlameBase - tSmokeboxGas) * Math.exp(-2.35 * s) : this.T_boilerC);
    }

    const boilingIntensity =
      this.T_boilerC >= 99.4 ? Math.max(0.0, Math.min(1.0, Math.max(0.0, qNetWater) / 680.0)) : 0.0;
    const carryOver = Math.max(0.0, (trueLevelPct - 66.0) / 34.0);
    const steamDrynessDome = Math.max(
      0.76,
      Math.min(0.992, 0.986 - 0.092 * boilingIntensity - 0.11 * carryOver)
    );
    const waterFreeZMm = Math.max(32.0, Math.min(68.5, 55.5 + (trueLevelPct - 50.0) * 0.23));
    const coldFeedCool = Math.min(18.0, mdotFw * 14500.0);
    const waterLayerTempsC = [
      Math.max(25.0, this.T_boilerC - 5.5 - 0.85 * coldFeedCool),
      Math.max(25.0, this.T_boilerC - 2.6 - 0.40 * coldFeedCool),
      Math.max(25.0, this.T_boilerC - 0.8 - 0.12 * coldFeedCool),
      this.T_boilerC,
    ];

    // 火室紫铜顶板 6 分区轴向局部浸没与局部空焚き红热场 (X ∈ [-106.5, -47.0] mm)
    const tanSlosh = Math.tan((this.sloshPitchDeg * Math.PI) / 180.0);
    const cZone = (this.M_copperCrown / 6.0) * CP_COPPER;
    for (let zIdx = 0; zIdx < 6; zIdx++) {
      const xRelMm = -25.0 + 10.0 * zIdx;
      const localDepthMm = hCrownStaticMm + xRelMm * tanSlosh;
      const fZoneWet = Math.max(0.0, Math.min(1.0, localDepthMm / 3.8));
      const qZoneIn = (qFireCrown / 6.0) * (0.92 + 0.16 * (zIdx / 5.0));
      const gZone = (22.0 * fZoneWet + 0.34 * (1.0 - fZoneWet)) / 6.0;
      const dTz = (qZoneIn - gZone * (this.crownZonesC[zIdx] - this.T_boilerC)) / Math.max(1.0, cZone);
      this.crownZonesC[zIdx] = Math.max(25.0, Math.min(790.0, this.crownZonesC[zIdx] + dTz * dtTh));
    }

    // 13-9 -> 17-6 -> 17-5 四程过热管 -> 13-4 -> 7-9 沿程干度 x(s) 与过热升温 T(s)
    const shProfileDryness: number[] = [];
    const shProfileTempC: number[] = [];
    const latentRem = 1.0 - steamDrynessDome;
    const superheatGain = Math.max(0.0, this.T_superheatC - this.T_boilerC);
    for (let k = 0; k < 8; k++) {
      const s = k / 7.0;
      const dryFrac = Math.min(1.0, steamDrynessDome + latentRem * Math.min(1.0, s / 0.34));
      const shFrac = s <= 0.28 ? 0.0 : Math.pow((s - 0.28) / 0.72, 0.88);
      shProfileDryness.push(dryFrac);
      shProfileTempC.push(this.T_boilerC + superheatGain * shFrac);
    }

    const pGaugeBoil = Math.max(0.0, (this.P_boiler - P_ATM) / PA_PER_KGF_CM2);
    const pGaugeChest = Math.max(0.0, (this.P_steamChest - P_ATM) / PA_PER_KGF_CM2);
    const cylLPKgf = [
      Math.max(0.0, (this.P_cylL[0] - P_ATM) / PA_PER_KGF_CM2),
      Math.max(0.0, (this.P_cylL[1] - P_ATM) / PA_PER_KGF_CM2),
    ];
    const cylRPKgf = [
      Math.max(0.0, (this.P_cylR[0] - P_ATM) / PA_PER_KGF_CM2),
      Math.max(0.0, (this.P_cylR[1] - P_ATM) / PA_PER_KGF_CM2),
    ];
    const maxPortArea = Math.max(1e-9, this.portBreadth * this.calib.portWidth);
    const valvePortOpenL = [
      Math.max(-1.0, Math.min(1.0, (areasL[0] - areasL[1]) / maxPortArea)),
      Math.max(-1.0, Math.min(1.0, (areasL[2] - areasL[3]) / maxPortArea)),
    ];
    const valvePortOpenR = [
      Math.max(-1.0, Math.min(1.0, (areasR[0] - areasR[1]) / maxPortArea)),
      Math.max(-1.0, Math.min(1.0, (areasR[2] - areasR[3]) / maxPortArea)),
    ];

    const exhaustDryness = Math.max(
      0.82,
      Math.min(0.99, 0.875 + 0.0016 * Math.max(0.0, this.T_superheatC - 115.0) - 0.014 * pGaugeChest)
    );
    const oilFilmUm = 0.22 + 3.58 * Math.min(1.0, this.lubricatorOilPct / 20.0) * this.sealHealth;
    const pipeArea66 = Math.PI * Math.pow(0.0011, 2);
    const feedwaterVelMS = Math.min(2.6, (vToBoil + mdotFwHand * 1e-3) / pipeArea66);
    const pumpDelivActive = vPumpRam < -1e-4 && this.tenderWaterMl > 0.5;
    const pumpSuctActive = vPumpRam > 1e-4 && this.tenderWaterMl > 0.5;
    const pumpChamberPKgf = pumpDelivActive ? pGaugeBoil + 0.35 : pumpSuctActive ? -0.18 : 0.05;
    const checkBallLiftsMm = [
      pumpSuctActive ? Math.min(0.85, 16.0 * Math.abs(vPumpRam)) : 0.0,
      pumpDelivActive && vToBoil > 1e-9 ? Math.min(0.85, 16.0 * Math.abs(vPumpRam)) : 0.0,
    ];
    const lubeWaterFrac = Math.max(0.0, Math.min(1.0, this.lubeCondensateMl / this.lubeTankCapMl));

    const boilingProgressPct = Math.max(0.0, Math.min(100.0, ((this.T_boilerC - 25.0) / 75.0) * 100.0));
    const turretPressureKgf = Math.max(0.0, pGaugeBoil - 0.06 * Math.min(2.5, mdotSteamOut * 1000.0));
    this.pgBourdonKgf += Math.min(1.0, dt / 0.12) * (turretPressureKgf - this.pgBourdonKgf);
    const appClamped = Math.max(0.0, Math.min(100.0, apparentLevelPct));
    const sgTarget = ctrl.sightDrainValve ? Math.max(4.0, appClamped * (0.18 + 0.12 * Math.exp(-pGaugeBoil))) : appClamped;
    this.sgMeniscusPct += Math.min(1.0, dt * 11.0) * (sgTarget - this.sgMeniscusPct);
    const mariotteHeadspaceKpa = ctrl.fuelCapOpen
      ? 0.0
      : Math.max(0.0, 0.789 * 9.81 * (0.032 * (this.alcoholMl / 130.0)) * 1e-3 * 1000.0);

    return {
      simTime: this.simTime,
      boilingProgressPct,
      forceL,
      forceR,
      forcePump,
      valveFrictionL,
      valveFrictionR,
      boilerPressureKgf: pGaugeBoil,
      pgBourdonKgf: this.pgBourdonKgf,
      turretPressureKgf,
      steamChestKgf: pGaugeChest,
      boilerTempC: this.T_boilerC,
      superheatTempC: this.T_superheatC,
      crownTempC: Math.max(this.T_crownC, ...this.crownZonesC),
      crownSubmergedMm,
      copperAnnealDamage: this.copperAnnealDamage,
      sealHealth: this.sealHealth,
      waterLevelPct: appClamped,
      trueWaterLevelPct: Math.max(0.0, Math.min(100.0, trueLevelPct)),
      sightGlassLevelPct: Math.max(0.0, Math.min(100.0, this.sgMeniscusPct)),
      cylWallTempC: this.T_cylWallC,
      condensateG: this.condensateKg * 1000.0,
      smokeboxVacuumPa: smokeboxVacPa,
      lutActive: this.lut !== null,
      steamGenGS: mdotSteamOut * 1000.0,
      feedwaterGS: mdotFw * 1000.0,
      alcoholMl: this.alcoholMl + this.sumpFuelMl,
      tenderWaterMl: this.tenderWaterMl,
      tenderWaterTempC: this.T_tenderWaterC,
      returnWaterMlS,
      waterFillMlS,
      tenderDrainMlS,
      tenderOverflowMlS,
      mariotteHeadspaceKpa,
      mariotteBubbling,
      mariotteFloodAlarm,
      whistleSteamGS: mdotWhistle * 1000.0,
      blowdownGS: mdotBlowdownLiq * 1000.0,
      sightPurgeGS: mdotSightPurge * 1000.0,
      lubricatorOilPct: this.lubricatorOilPct,
      wheelRpm: rpm,
      scaleSpeedKmh: speedKmh,
      safetyValveOpen: this.safetyValveOpen,
      dryFireAlarm: dryAlarm,
      flashSteamSpike: flashSpike,
      hydrolockL,
      hydrolockR,
      exhaustPlume: Math.min(1.0, (mdotExh + 0.4 * mdotBlw) * 1800.0),
      safetyPlume: this.safetyValveOpen ? 1.0 : 0.0,
      drainPlume: Math.min(1.0, mdotDrn * 2400.0),
      whistlePlume: Math.min(1.0, mdotWhistle * 3800.0),
      sightPurgePlume: ctrl.sightDrainValve ? Math.min(1.0, 0.35 + mdotSightPurge * 650.0) : 0.0,
      blowdownPlume: ctrl.blowdownValve > 0.02 ? Math.min(1.0, 0.30 + mdotBlowdownLiq * 420.0) : 0.0,
      spatial: {
        gpuActive: false,
        gpuName: "WASM-1D+Spatial",
        totalVoxels: 0,
        gpuStepMs: 0.0,
        gpuHz: 2000.0,
        stlPassagesVerified: true,
        openFlueCount: 5,
        airFuelPhi,
        flameTiltDeg,
        wickTempsC,
        flueGasProfileC,
        steamDrynessDome,
        boilingIntensity,
        waterFreeZMm,
        waterSloshPitchDeg: this.sloshPitchDeg,
        waterLayerTempsC,
        crownHotspotsC: [...this.crownZonesC],
        shProfileTempC,
        shProfileDryness,
        cylLPKgf,
        cylRPKgf,
        valvePortOpenL,
        valvePortOpenR,
        exhaustDryness,
        oilFilmUm,
        feedwaterVelMS,
        pumpChamberPKgf,
        checkBallLiftsMm,
        lubeWaterFrac,
        midplaneTempB64: "",
      },
    };
  }
}

export interface AutoFiremanPolicy {
  target_speed_kmh: number;
  p_target_kgf: number;
  cutoff_start: number;
  cutoff_cruise: number;
  cutoff_speed_span_kmh: number;
  kp_reg: number;
  ki_reg: number;
  kp_blower: number;
  kp_bypass: number;
  ff_bypass_base: number;
  warmup_wall_temp_c: number;
}

export class AutoFiremanAgent {
  policy: AutoFiremanPolicy = {
    target_speed_kmh: 50.0,
    p_target_kgf: 3.72,
    cutoff_start: 0.85,
    cutoff_cruise: 0.58,
    cutoff_speed_span_kmh: 42.0,
    kp_reg: 0.018,
    ki_reg: 0.0045,
    kp_blower: 0.95,
    kp_bypass: 0.065,
    ff_bypass_base: 0.42,
    warmup_wall_temp_c: 102.0,
  };
  private speedInt = 0.0;
  private direction = 1.0;

  constructor(rawJson?: any) {
    if (rawJson) {
      const p = rawJson.optimal_policy || rawJson;
      Object.assign(this.policy, p);
    }
  }

  reset(direction = 1.0): void {
    this.speedInt = 0.0;
    this.direction = direction >= 0.0 ? 1.0 : -1.0;
  }

  update(dt: number, out: StepOutput, ctrl: CabControls): void {
    const p = this.policy;
    if (Math.abs(ctrl.reverser) > 0.05) {
      this.direction = ctrl.reverser >= 0.0 ? 1.0 : -1.0;
    }

    ctrl.burnerLit = out.alcoholMl > 0.2;
    ctrl.fuelValve = out.boilerPressureKgf > 3.90 ? Math.max(0.35, 1.0 - 2.5 * (out.boilerPressureKgf - 3.90)) : 1.0;

    const pErr = p.p_target_kgf - out.boilerPressureKgf;
    const exhRelief = Math.min(0.35, out.scaleSpeedKmh * 0.006);
    ctrl.blowerValve = Math.max(0.0, Math.min(1.0, p.kp_blower * pErr - exhRelief + 0.12));
    ctrl.chimneyFan = out.boilerPressureKgf < 0.8 && Math.abs(out.wheelRpm) < 5.0;

    const needDrain =
      out.hydrolockL || out.hydrolockR || out.cylWallTempC < p.warmup_wall_temp_c || out.condensateG > 0.28;
    ctrl.drainCock = needDrain;

    if (out.boilerPressureKgf < 1.6) {
      ctrl.regulator = 0.0;
      ctrl.reverser = this.direction * p.cutoff_start;
      return;
    }

    const spdRatio = Math.max(0.0, Math.min(1.0, out.scaleSpeedKmh / Math.max(10.0, p.cutoff_speed_span_kmh)));
    const magRev = p.cutoff_start - (p.cutoff_start - p.cutoff_cruise) * spdRatio;
    ctrl.reverser = this.direction * magRev;

    const vErr = p.target_speed_kmh - out.scaleSpeedKmh;
    this.speedInt = Math.max(-18.0, Math.min(35.0, this.speedInt + vErr * dt));
    let regCmd = 0.28 + p.kp_reg * vErr + p.ki_reg * this.speedInt;
    if (needDrain) regCmd = Math.min(regCmd, 0.42);
    ctrl.regulator = Math.max(0.0, Math.min(0.92, regCmd));

    const wErr = out.trueWaterLevelPct - 50.0;
    let bypCmd = p.ff_bypass_base + p.kp_bypass * wErr - 0.18 * out.steamGenGS;
    if (out.crownTempC > 215.0 && out.crownSubmergedMm < 1.5) {
      bypCmd = Math.max(0.42, bypCmd);
    }
    if (out.trueWaterLevelPct < 42.0 && Math.abs(out.wheelRpm) < 25.0 && out.crownTempC < 210.0) {
      bypCmd = 0.0;
    }
    ctrl.bypassValve = Math.max(0.0, Math.min(1.0, bypCmd));
  }
}
